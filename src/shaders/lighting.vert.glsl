#version 450

layout(location = 0) in vec3 position;

layout(set = 0, binding = 2) uniform MVP_Data {
    mat4 model;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    gl_Position = uniforms.projection * uniforms.view * uniforms.model* vec4(position, 1.0);
}