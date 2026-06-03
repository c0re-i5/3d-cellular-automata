
#version 430
layout(local_size_x=4, local_size_y=4, local_size_z=4) in;
uniform int u_occ_dim;
uniform int u_boundary;   // 0 = clamp, 1 = wrap (toroidal)
layout(r8ui, binding=0) uniform restrict readonly  uimage3D u_occ_in;
layout(r8ui, binding=1) uniform restrict writeonly uimage3D u_active_out;

uint sample_block(ivec3 b) {
    if (u_boundary == 1) {
        b = ((b % u_occ_dim) + u_occ_dim) % u_occ_dim;
    } else {
        if (any(lessThan(b, ivec3(0))) || any(greaterThanEqual(b, ivec3(u_occ_dim))))
            return 0u;
    }
    return imageLoad(u_occ_in, b).r;
}
void main() {
    ivec3 b = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(b, ivec3(u_occ_dim)))) return;
    uint a = sample_block(b);
    // 6-neighbour face dilation (axis-aligned, 1 block = 8 voxels)
    a |= sample_block(b + ivec3(-1, 0, 0));
    a |= sample_block(b + ivec3( 1, 0, 0));
    a |= sample_block(b + ivec3( 0,-1, 0));
    a |= sample_block(b + ivec3( 0, 1, 0));
    a |= sample_block(b + ivec3( 0, 0,-1));
    a |= sample_block(b + ivec3( 0, 0, 1));
    imageStore(u_active_out, b, uvec4(a, 0u, 0u, 0u));
}
