#version 450
        
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 colour;

layout(location = 0) out vec3 out_colour;
layout(location = 1) out vec3 out_normal;

layout(set = 0, binding = 0) uniform MVP_Data {
        mat4 model;
        mat4 view;
        mat4 proj;
    } uniforms;
}

void main() {
        out_colour = colour;
        out_normal = mat3(uniforms.model) * normal;
        gl_Position = uniforms.projection * worldview * vec4(position, 1.0);
    }
}