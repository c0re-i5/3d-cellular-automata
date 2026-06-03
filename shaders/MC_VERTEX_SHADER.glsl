
#version 430
layout(std430, binding=10) readonly buffer MCVerts { vec4 verts[]; } V;

uniform mat4 u_view_proj;
uniform vec3 u_camera_pos;
uniform vec3 u_world_per_cell; // box_dims/size — grid cell → world coord

out vec3 v_normal;
out vec3 v_world;

void main() {
    uint i = uint(gl_VertexID);
    vec3 p = V.verts[2u * i].xyz * u_world_per_cell;
    vec3 n = V.verts[2u * i + 1u].xyz;
    gl_Position = u_view_proj * vec4(p, 1.0);
    v_normal = n;
    v_world = p;
}
