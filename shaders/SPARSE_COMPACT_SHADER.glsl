
#version 430
layout(local_size_x=4, local_size_y=4, local_size_z=4) in;
uniform int u_occ_dim;
layout(r8ui, binding=0) uniform restrict readonly uimage3D u_active_in;
// Slot 0..2 = (groupCountX, 1, 1); compact list lives in a separate SSBO.
layout(std430, binding=10) buffer SparseIndirect {
    uint sparse_groups_x;
    uint sparse_groups_y;
    uint sparse_groups_z;
};
layout(std430, binding=11) buffer SparseBlocks {
    uvec4 sparse_blocks[];   // .xyz = block coord, .w unused (16-byte aligned)
};
void main() {
    ivec3 b = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(b, ivec3(u_occ_dim)))) return;
    if (imageLoad(u_active_in, b).r == 0u) return;
    uint slot = atomicAdd(sparse_groups_x, 1u);
    sparse_blocks[slot] = uvec4(uint(b.x), uint(b.y), uint(b.z), 0u);
}
