#version 450

layout(location = 0) in vec3 position;

layout(set = 0, binding = 2) uniform VpData {
    mat4 view;
    mat4 proj;
} vp;

layout(location = 0) out vec3 frag_pos;

void main() {
    frag_pos = position;
    gl_Position = vp.proj * vp.view * vec4(position, 1.0);
}