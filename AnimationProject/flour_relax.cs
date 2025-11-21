#version 430
layout(local_size_x = 16, local_size_y = 16) in;

uniform int hmWidth;
uniform int hmHeight;
uniform float talus;       // slope limit in grains
uniform float relaxRate;   // how strong relaxation is (0..1)

// read old heights
layout(r32ui, binding = 0) readonly uniform uimage2D heightIn;
// write new heights
layout(r32ui, binding = 1) writeonly uniform uimage2D heightOut;

uint H(ivec2 c)
{
    return imageLoad(heightIn, c).r;
}

void main()
{
    ivec2 c = ivec2(gl_GlobalInvocationID.xy);
    if (c.x >= hmWidth || c.y >= hmHeight) return;

    uint hC = H(c);
    float h = float(hC);

    // 4-neighborhood
    ivec2 n[4] = ivec2[4](
        ivec2(c.x + 1, c.y),
        ivec2(c.x - 1, c.y),
        ivec2(c.x, c.y + 1),
        ivec2(c.x, c.y - 1)
    );

    float hNew = h;

    for (int k = 0; k < 4; ++k)
    {
        ivec2 nb = n[k];
        if (nb.x < 0 || nb.x >= hmWidth || nb.y < 0 || nb.y >= hmHeight) continue;

        float hn = float(H(nb));
        float diff = hNew - hn;

        if (diff > talus)
        {
            // move only the excess slope
            float transfer = (diff - talus) * 0.5 * relaxRate;
            hNew -= transfer;
        }
        else if (diff < -talus)
        {
            // neighbor is too high; we gain a bit from it
            float transfer = (-diff - talus) * 0.5 * relaxRate;
            hNew += transfer;
        }
    }

    if (hNew < 0.0) hNew = 0.0;

    imageStore(heightOut, c, uvec4(uint(hNew), 0u, 0u, 0u));
}
