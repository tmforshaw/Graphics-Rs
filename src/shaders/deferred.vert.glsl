#version 450
        
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec3 out_colour;

layout(set = 0, binding = 0) uniform VpData {
    mat4 view;
    mat4 proj;
} vp;

void main() {
    out_colour = vec3(1.0); //vec3(0.75, 0.1, 0.2);
    out_normal = normal;

    gl_Position = vp.proj * vp.view * vec4(position, 1.0);
}
