#version 450

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec3 in_colour;

layout(location = 0) out vec3 f_normal;
layout(location = 1) out vec4 f_colour;

void main() {
    f_normal = in_normal;
    f_colour = vec4(in_colour, 1.0);
}
