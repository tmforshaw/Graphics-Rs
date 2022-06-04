#version 450

layout(location = 0) in vec3 frag_pos;

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_normals;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_colour;

layout(location = 0) out vec4 f_colour;

layout(set = 0, binding = 2) uniform MvpData {
    mat4 model;
    mat4 view;
    mat4 proj;
} mvp;

layout(set = 0, binding = 3) uniform LightData {
    vec3 position;
    vec3 colour;
    float intensity;
} light;


layout(set = 0, binding = 4) uniform CameraData {
    vec3 position;
    uint dt;
} camera;

void main() {
    vec3 colour = subpassLoad(u_colour).xyz;
    vec3 normals = subpassLoad(u_normals).xyz;
    
    vec3 lightDir = normalize(light.position - frag_pos);
    vec3 viewDir = normalize(camera.position - frag_pos);
    vec3 reflectDir = reflect(-lightDir, normals);
    
    vec3 ambient = vec3(0.1);
    vec3 diffuse = max(dot(normals, lightDir), 0.0) * light.colour * light.intensity; // * abs(vec3(sin(camera.dt * 0.05), cos(camera.dt * 0.1), sin(camera.dt * 0.2)));
    vec3 specular = pow(max(dot(viewDir, reflectDir), 0.0), 64) * 0.5 *  light.colour * light.intensity;
    
    f_colour = vec4((ambient + diffuse + specular) * colour, 1.0);
}
