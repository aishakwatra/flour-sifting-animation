#version 430

layout(location = 0) in vec2 inXZ;  // world-space x,z on table plane
layout(location = 1) in vec2 inUV;  // 0..1 across the table

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform float tableY;
uniform float flourUnitHeight;
uniform int hmWidth;
uniform int hmHeight;

// Integer heightmap from compute shader
layout(binding = 0) uniform usampler2D flourHeightmap;

out vec2 vUV;
out vec3 vWorldPos;

uint sampleGrains(vec2 uv)
{
    return texture(flourHeightmap, uv).r;
}

void main()
{
    float du = 1.0 / float(hmWidth);
    float dv = 1.0 / float(hmHeight);

    // 5x5 Gaussian-ish kernel weights (sum = 273)
    int w[5] = int[5](1, 4, 7, 4, 1);

    uint sum = 0u;
    uint wsum = 0u;

    for (int oy = -2; oy <= 2; ++oy)
    {
        for (int ox = -2; ox <= 2; ++ox)
        {
            uint grains = texture(flourHeightmap, inUV + vec2(ox*du, oy*dv)).r;
            uint weight = uint(w[ox+2] * w[oy+2]); // separable weights

            sum  += grains * weight;
            wsum += weight;
        }
    }

    float grainsSmoothed = float(sum) / float(wsum);
    float H = grainsSmoothed * flourUnitHeight;

    vec3 worldPos = vec3(inXZ.x, tableY + H, inXZ.y);

    vUV = inUV;
    vWorldPos = worldPos;

    gl_Position = projection * view * model * vec4(worldPos, 1.0);
}
