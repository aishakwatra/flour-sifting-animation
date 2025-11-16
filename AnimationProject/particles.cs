#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// --- Physics Constants ---
const vec3 GRAVITY = vec3(0.0, -9.8, 0.0);
const float PARTICLE_INV_MASS = 1.0 / 0.1;
const float DELTA_T = 0.005;
const float FLOOR_Y = -2.3;
const float DAMPING_Y = 0.6;   // Coefficient of restitution (how much Y-energy is kept on bounce)
const float DAMPING_XZ = 0.98; // Friction on the floor

// --- Spawn variables from C++ ---
uniform vec3 spawnCenter;
uniform float spawnRangeXZ;
uniform float spawnRangeY;

// --- Lifetime Constants ---
const float MIN_LIFETIME = 3.0;
const float MAX_LIFETIME = 5.0;

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

void main()
{
    uint idx = gl_GlobalInvocationID.x;

    // Get current lifetime from the .w component
    float life = Velocity[idx].w;
    life -= DELTA_T; // Decrease lifetime

    if (life <= 0.0)
    {
        // --- 1. Particle is DEAD, Respawn it ---

        // Reset position to a random spot at the top
        Position[idx].x = spawnCenter.x + (rand(idx * 0.21) * 2.0 - 1.0) * spawnRangeXZ;
        Position[idx].y = spawnCenter.y + rand(idx) * spawnRangeY;
        Position[idx].z = spawnCenter.z + (rand(idx * 0.79) * 2.0 - 1.0) * spawnRangeXZ;

        // Reset velocity
        Velocity[idx].xyz = vec3(0.0, rand(idx * 0.5) * -1.0, 0.0);
        // Reset lifetime
        Velocity[idx].w = MIN_LIFETIME + rand(idx) * (MAX_LIFETIME - MIN_LIFETIME);
    }
    else
    {
        // --- 2. Particle is ALIVE, apply physics ---
        vec3 p = Position[idx].xyz;
        vec3 v = Velocity[idx].xyz;

        // Apply constant gravity force
        vec3 force = GRAVITY;

        // Apply Euler integration
        vec3 a = force * PARTICLE_INV_MASS;
        vec3 v_new = v + a * DELTA_T;
        vec3 p_new = p + v * DELTA_T + 0.5 * a * DELTA_T * DELTA_T;

        // --- 3. Check for Bounce ---
        if (p_new.y < FLOOR_Y)
        {
            p_new.y = FLOOR_Y; // Clamp to floor

            v_new.y = -v_new.y * DAMPING_Y; // Reflect and dampen Y velocity
            v_new.x *= DAMPING_XZ; // Apply friction
            v_new.z *= DAMPING_XZ;
        }

        // Update buffers
        Position[idx] = vec4(p_new, 1.0);
        Velocity[idx].xyz = v_new;
        Velocity[idx].w = life; // Write back updated (reduced) lifetime
    }
}