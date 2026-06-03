
#version 430
layout(local_size_x=64, local_size_y=1, local_size_z=1) in;

// Voxel grid bound IN PLACE: same texture on both image units, agents
// read-modify-write the cells they visit.
layout(rgba32f, binding=0) uniform image3D u_grid_r;
layout(rgba32f, binding=1) uniform image3D u_grid_w;

struct Agent {
    ivec4 pos_dir;   // .xyz = pos, .w = dir id (0..5)
    ivec4 state;     // reserved per-rule
};
layout(std430, binding=8) buffer AgentBuf {
    Agent agents[];
};

uniform int u_size;
uniform float u_dt;
uniform int u_boundary;
uniform int u_frame;
uniform int u_pass;
uniform float u_param0;
uniform float u_param1;
uniform float u_param2;
uniform float u_param3;

// 6 axis-aligned direction vectors. Order MUST match AGENT_DIRS in host.
// Interleaved so consecutive entries are PERPENDICULAR (each +1 step is a
// 90° turn around some axis, never a 180° axis flip). Opposites are
// always 3 apart, so dir^3 (XOR with bit 0b11... actually +3 mod 6) is
// the reverse direction.
const ivec3 DIR[6] = ivec3[6](
    ivec3( 1, 0, 0), ivec3( 0, 1, 0), ivec3( 0, 0, 1),
    ivec3(-1, 0, 0), ivec3( 0,-1, 0), ivec3( 0, 0,-1)
);

ivec3 wrap_pos(ivec3 p) {
    return ((p % u_size) + u_size) % u_size;
}
