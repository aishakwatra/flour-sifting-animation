#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// --- Physics Constants ---
const vec3 GRAVITY = vec3(0.0, -9.8, 0.0);
const float PARTICLE_INV_MASS = 1.0 / 0.1;
const float DELTA_T = 0.005;

// We will treat this as TABLE_Y (base of flour grid)
const float TABLE_Y = -2.3;

// --- Spawn variables from C++ ---
uniform vec3 spawnCenter;
uniform float spawnRangeXZ;

// --- Table / heightmap info from C++ ---
uniform float tableMinX;
uniform float tableMaxX;
uniform float tableMinZ;
uniform float tableMaxZ;
uniform float tableY;           // should match TABLE_Y
uniform int hmWidth;
uniform int hmHeight;
uniform float flourUnitHeight;  // height per "grain"

// --- Lifetime Constants ---
const float MIN_LIFETIME = 3.0;
const float MAX_LIFETIME = 5.0;

// --- Flour heightmap (integer image, for atomicAdd) ---
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


// Returns true if (x,z) is over the table rectangle.
// Outputs the heightmap cell (i,j) in 'cell'.

bool projectToHeightmap(vec3 p, out ivec2 cell)
{
    float x = p.x;
    float z = p.z;

    // Check if horizontally above the table
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
    Position[idx].x = spawnCenter.x + (rand(idx * 0.21) * 2.0 - 1.0) * spawnRangeXZ;
    Position[idx].y = spawnCenter.y; 
    Position[idx].z = spawnCenter.z + (rand(idx * 0.79) * 2.0 - 1.0) * spawnRangeXZ;

    Velocity[idx].xyz = vec3(0.0, rand(idx * 0.5) * -1.0, 0.0);
    Velocity[idx].w = MIN_LIFETIME + rand(idx) * (MAX_LIFETIME - MIN_LIFETIME);
}

void main()
{
    uint idx = gl_GlobalInvocationID.x;

    // Get current lifetime
    float life = Velocity[idx].w;
    life -= DELTA_T;

    if (life <= 0.0)
    {
        // Lifetime ended – respawn as fresh falling flour
        respawnParticle(idx);
        return;
    }

    // --- Particle is ALIVE, apply physics ---
    vec3 p = Position[idx].xyz;
    vec3 v = Velocity[idx].xyz;

    vec3 force = GRAVITY;
    vec3 a = force * PARTICLE_INV_MASS;
    vec3 v_new = v + a * DELTA_T;
    vec3 p_new = p + v * DELTA_T + 0.5 * a * DELTA_T * DELTA_T;

    // --- Flour / table collision & accumulation ---
    ivec2 cell;
    bool overTable = projectToHeightmap(p_new, cell);

    if (overTable)
    {
        // Current flour height at this cell (integer "grains")
        uint hInt = imageLoad(flourHeightmap, cell).r;
        float H = float(hInt) * flourUnitHeight;

        float surfaceY = tableY + H;

        if (p_new.y <= surfaceY)
        {
            // We've hit the flour/table surface:
            // 1) deposit one grain
            imageAtomicAdd(flourHeightmap, cell, 1u);

            // 2) respawn particle as new falling flour
            respawnParticle(idx);
            return;
        }
    }

    // If not collided with table/flour yet, keep falling
    Position[idx] = vec4(p_new, 1.0);
    Velocity[idx].xyz = v_new;
    Velocity[idx].w = life;
}