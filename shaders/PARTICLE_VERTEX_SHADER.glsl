
#version 430

layout(std430, binding = 9) readonly buffer Particles { vec4 data[]; } P;

uniform mat4  u_view_proj;
uniform vec3  u_camera_pos;
uniform vec3  u_camera_right;
uniform vec3  u_camera_up;
uniform float u_size_world;     // particle radius in cell units
uniform vec3  u_world_per_cell; // box_dims/size — grid cell → world coord
uniform int   u_color_mode;     // 0=fixed, 1=speed, 2=lifetime, 3=position
uniform vec3  u_color_fixed;
uniform float u_speed_scale;

out vec2  v_uv;        // [-1..1]² for the quad
out vec3  v_color;
out float v_life;

void main() {
    uint pid = uint(gl_VertexID) / 6u;
    uint vid = uint(gl_VertexID) % 6u;
    vec4 a = P.data[2u * pid + 0u];
    vec4 b = P.data[2u * pid + 1u];
    vec3 pos = a.xyz;
    float life = a.w;
    vec3 vel = b.xyz;

    // 6-vertex quad: two tris covering [-1,1]²
    vec2 corners[6] = vec2[6](
        vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
        vec2( 1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
    );
    vec2 c = corners[vid];

    // Hide dead particles by collapsing them to a point off-screen.
    float vis = step(0.001, life);
    // Particle pos is in grid cell coords [0..size]; convert to world
    // coords [0..box_dims] before the view-projection transform.
    vec3 wp_center = pos * u_world_per_cell;
    float scl = (u_world_per_cell.x + u_world_per_cell.y + u_world_per_cell.z) / 3.0;
    vec3 world_pos = wp_center + (u_camera_right * c.x + u_camera_up * c.y) * u_size_world * scl;
    world_pos = mix(vec3(-1e6), world_pos, vis);

    gl_Position = u_view_proj * vec4(world_pos, 1.0);
    v_uv = c;
    v_life = life;

    vec3 col;
    if (u_color_mode == 1) {
        float s = length(vel) * u_speed_scale;
        col = mix(vec3(0.1, 0.3, 0.9), vec3(1.0, 0.7, 0.1), clamp(s, 0.0, 1.0));
    } else if (u_color_mode == 2) {
        float t = clamp(life, 0.0, 1.0);
        col = mix(vec3(1.0, 0.2, 0.1), vec3(0.6, 0.9, 1.0), t);
    } else if (u_color_mode == 3) {
        col = vec3(0.5) + 0.5 * sin(pos * 0.3 + vec3(0.0, 2.0, 4.0));
    } else {
        col = u_color_fixed;
    }
    v_color = col;
}
