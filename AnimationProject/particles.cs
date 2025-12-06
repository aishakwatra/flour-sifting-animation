#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// Physics Constants
const vec3 GRAVITY = vec3(0.0, -9.8, 0.0);
const float PARTICLE_INV_MASS = 1.0 / 0.1;
const float DELTA_T = 0.005;

// Spawn variables from main 
uniform vec3 spawnCenter;
uniform float spawnRangeXZ;

// Table / heightmap info main 
uniform float tableMinX;
uniform float tableMaxX;
uniform float tableMinZ;
uniform float tableMaxZ;
uniform float tableY;           
uniform int hmWidth;
uniform int hmHeight;
uniform float flourUnitHeight;

uniform vec3  sifterCenter;
uniform vec3 sifterCenterPrev;

uniform float sifterRadius;
uniform float sifterY;

uniform float sifterCellSize;      // spacing between mesh lines
uniform float sifterBarThickness;  // thickness of the bars

const float TABLE_Y = -2.3;
const float KILL_Y = TABLE_Y - 2.0;

// --- Flour heightmap (integer image) ---
layout(r32ui, binding = 0) uniform uimage2D flourHeightmap;

// --- Buffers ---
layout(std430, binding = 0) buffer Pos
{
    vec4 Position [];
};


layout(std430, binding = 1) buffer Vel
{

    vec4 Velocity [];
};

float rand(float n)
{
    return fract(sin(n) * 43758.5453123);
}


// Returns true if hitPosWorld is on a mesh bar (solid), false if it's a hole (flour passes through)
bool sifterBarAt(vec3 hitPosWorld)
{

    vec2 localXZ = hitPosWorld.xz - sifterCenter.xz;

    // check if within the circular sift
    float r = length(localXZ);
    if (r > sifterRadius)
        return false;  // outside the sift

    float cell = sifterCellSize;
    float halfBar = 0.5 * sifterBarThickness;

    // Offset inside current cell along X and Z
    float gx = abs(mod(localXZ.x + 0.5 * cell, cell) - 0.5 * cell);
    float gz = abs(mod(localXZ.y + 0.5 * cell, cell) - 0.5 * cell);

    // Cross-shaped mesh in each cell (a vertical and horizontal bar)
    bool onVerticalBar = gx < halfBar;
    bool onHorizontalBar = gz < halfBar;

    return (onVerticalBar || onHorizontalBar);
}

bool handleSifterBottomCollision(vec3 p, inout vec3 p_new, inout vec3 v_new)
{
    // Only consider particles moving downward through the bottom plane
    if (p.y > sifterY && p_new.y <= sifterY)
    {
        // Find intersection with y = sifterY along segment p - p_new
        float t = (sifterY - p.y) / (p_new.y - p.y);
        t = clamp(t, 0.0, 1.0);

        vec3 hitPos = mix(p, p_new, t);

        // If this position is on a bar, we collide
        // if it's a hole, we pass through
        if (!sifterBarAt(hitPos))
        {
            // Hole = no collision, let flour keep falling
            return false;
        }

        // Collision with a mesh bar: treat as flat surface with normal straight up
        vec3 n = vec3(0.0, 1.0, 0.0);

        float vn = dot(v_new, n);
        vec3 vt = v_new - vn * n;

        if (vn < 0.0)
        {
            float restitution = 0.4; // bounciness
            float friction = 0.4; // damp sideways a bit

            vec3 v_reflected =
                -restitution * vn * n +    // bounce upward
                (1.0 - friction) * vt;     // keep some horizontal motion

            v_new = v_reflected;
        }

        // Sit just above the mesh to avoid jitter
        p_new = hitPos;
        p_new.y = sifterY + 0.0005;

        return true;
    }

    return false;
}

const float SIFTER_WALL_HEIGHT = 4.0;


bool handleSifterSideCollision(vec3 p, inout vec3 p_new, inout vec3 v_new)
{
    float yMin = sifterY;
    float yMax = sifterY + SIFTER_WALL_HEIGHT;

    //we’re completely below or above the cylindrical wall
    if ((p.y < yMin && p_new.y < yMin) || (p.y > yMax && p_new.y > yMax))
        return false;

    // distance from sifter center in XZ plane
    vec2 d_old = p.xz - sifterCenter.xz;
    vec2 d_new = p_new.xz - sifterCenter.xz;

    float r_old = length(d_old);
    float r_new = length(d_new);

    bool wasInside = (r_old <= sifterRadius);
    bool isOutside = (r_new > sifterRadius);

    //transitions from inside -> outside
    if (wasInside && isOutside)
    {
        // Snap the position back onto the cylinder surface
        if (r_new > 0.0)
        {
            vec2 dir = normalize(d_new); 
            p_new.xz = sifterCenter.xz + dir * (sifterRadius - 0.001);
        }

        // outward normal on the cylinder wall
        vec3 n;
        if (r_new > 0.0)
        {
            vec2 dir = normalize(d_new);
            n = normalize(vec3(dir.x, 0.0, dir.y)); 
        }
        else
        {
            n = vec3(1.0, 0.0, 0.0); // fallback
        }

        // decompose velocity into normal + tangential components
        float vn = dot(v_new, n);      
        vec3 vt = v_new - vn * n;       

        // If the particle is moving outward (same direction as n), reflect it
        if (vn > 0.0)
        {
            float restitution = 0.3;
            float friction = 0.2;

            vec3 v_reflected = -restitution * vn * n +   (1.0 - friction) * vt;   
            v_new = v_reflected;
        }

        return true;
    }

    return false;
}




// Returns true if (x,z) is over the table rectangle.
// Outputs the heightmap cell (i,j) in 'cell'.

bool projectToHeightmap(vec3 p, out ivec2 cell)
{
    float x = p.x;
    float z = p.z;

    if (x < tableMinX || x > tableMaxX ||
        z < tableMinZ || z > tableMaxZ)
    {
        return false;
    }

    float u = (x - tableMinX) / (tableMaxX - tableMinX);
    float v = (z - tableMinZ) / (tableMaxZ - tableMinZ);

    int i = int(floor(u * float(hmWidth)));
    int j = int(floor(v * float(hmHeight)));

    i = clamp(i, 0, hmWidth - 1);
    j = clamp(j, 0, hmHeight - 1);

    cell = ivec2(i, j);
    return true;
}

void respawnParticle(uint idx)
{
    float fi = float(idx);

    // --- circular spawn in XZ ---
    float u = rand(fi * 0.21);
    float v = rand(fi * 0.79);

    float radius = sqrt(u) * spawnRangeXZ;
    float angle = 6.2831853 * v;

    float dx = cos(angle) * radius;
    float dz = sin(angle) * radius;

    Position[idx].x = spawnCenter.x + dx;
    Position[idx].y = spawnCenter.y;
    Position[idx].z = spawnCenter.z + dz;

    // --- vertical velocity ---
    float rv = rand(fi * 0.5);
    float vy = -8.0 + rv * 8.0;

    // --- horizontal spray ---
    float u2 = rand(fi * 1.23);
    float v2 = rand(fi * 1.57);
    float hAngle = 6.2831853 * u2;       
    float hSpeed = 2.0 * v2;              

    float vx = cos(hAngle) * hSpeed;
    float vz = sin(hAngle) * hSpeed;

    Velocity[idx].xyz = vec3(vx, vy, vz);
}



void main()
{
    uint idx = gl_GlobalInvocationID.x;

    vec3 p = Position[idx].xyz;
    vec3 v = Velocity[idx].xyz;

    vec3 force = GRAVITY;
    vec3 a = force * PARTICLE_INV_MASS;

    vec3 v_new = v + a * DELTA_T;
    vec3 p_new = p + v_new * DELTA_T;

    if (p_new.y < KILL_Y)
    {
        respawnParticle(idx);
        return;
    }

    // Sifter bottom
    handleSifterBottomCollision(p, p_new, v_new);

    // Sifter side walls
    handleSifterSideCollision(p, p_new, v_new);

    // Table / flour heightmap
    ivec2 cell;

    if (projectToHeightmap(p_new, cell))
    {
        uint h = imageLoad(flourHeightmap, cell).r;
        float flourHeightWorld = tableY + float(h) * flourUnitHeight;

        if (p_new.y <= flourHeightWorld)
        {
            imageAtomicAdd(flourHeightmap, cell, 1u);
            respawnParticle(idx);
            return;
        }
    }

    Position[idx] = vec4(p_new, 1.0);
    Velocity[idx].xyz = v_new;

}
