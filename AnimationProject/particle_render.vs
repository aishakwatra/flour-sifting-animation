#version 330 core
layout (location = 0) in vec4 aPos; // Particle position from SSBO (VAO)

uniform mat4 projection;
uniform mat4 view;

void main()
{
    gl_Position = projection * view * aPos;
}