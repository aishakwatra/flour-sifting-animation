#version 430

in vec2 vUV;
in vec3 vWorldPos;

out vec4 FragColor;

layout(binding = 0) uniform usampler2D flourHeightmap;

uniform float flourUnitHeight;
uniform int hmWidth;
uniform int hmHeight;

uniform vec3 lightDir;     
uniform vec3 lightColor;   
uniform vec3 flourColor;   

float heightSmoothed(vec2 uv)
{
    float du = 1.0 / float(hmWidth);
    float dv = 1.0 / float(hmHeight);

    int w[5] = int[5](1, 4, 7, 4, 1);
    uint sum = 0u;
    uint wsum = 0u;

    for (int oy = -2; oy <= 2; ++oy)
    {
        for (int ox = -2; ox <= 2; ++ox)
        {
            uint g = texture(flourHeightmap, uv + vec2(ox*du, oy*dv)).r;
            uint weight = uint(w[ox+2] * w[oy+2]);
            sum += g * weight;
            wsum += weight;
        }
    }

    return (float(sum) / float(wsum)) * flourUnitHeight;
}

void main()
{
    uint grainsC = texture(flourHeightmap, vUV).r;
    if (grainsC == 0u) discard;

    float du = 1.0 / float(hmWidth);
    float dv = 1.0 / float(hmHeight);

    float hR = heightSmoothed(vUV + vec2(du, 0));
    float hL = heightSmoothed(vUV - vec2(du, 0));
    float hU = heightSmoothed(vUV + vec2(0, dv));
    float hD = heightSmoothed(vUV - vec2(0, dv));

    float dx = (hR - hL) / (2.0 * du);
    float dz = (hU - hD) / (2.0 * dv);

    vec3 N = normalize(vec3(-dx, 1.0, -dz));

    vec3 L = normalize(-lightDir);
    float diff = max(dot(N, L), 0.0);

    // --- BRIGHTER LIGHTING ---
    float ambient = 0.85;                 
    float diffuse = diff * 0.85;          

    vec3 color = flourColor * (ambient + diffuse) * lightColor;

    float slope = clamp(length(vec2(dx, dz)) * 0.08, 0.0, 1.0);
    color *= (1.0 - 0.10 * slope);       

    FragColor = vec4(color, 1.0);
}

