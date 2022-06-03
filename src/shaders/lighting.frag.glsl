#version 450

layout(location = 0) in vec3 frag_pos;

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_normals;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_colour;

layout(location = 0) out vec4 f_colour;

void main() {
    vec3 colour = subpassLoad(u_colour).xyz;
    vec3 normals = subpassLoad(u_normals).xyz;
    
    vec3 lightDir = vec3(0.0, 0.0, -1.0);
    vec3 viewDir = normalize(vec3(0.0, 0.0, -1.0) - frag_pos);
    vec3 reflectDir = reflect(-lightDir, normals);
    
    vec3 ambient = vec3(0.1);
    vec3 diffuse = max(dot(normals, lightDir), 0.0) * vec3(1.0);
    vec3 specular = pow(max(dot(viewDir, reflectDir), 0.0), 64) * 0.5 * vec3(1.0);
    
    f_colour = vec4((ambient + diffuse + specular) * colour, 1.0);
}
