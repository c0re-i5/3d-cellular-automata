
#version 430
layout(local_size_x=4, local_size_y=4, local_size_z=4) in;

uniform sampler3D u_view;    // R16F (channel/abs already applied)
uniform int u_size;          // full grid dimension
uniform int u_block_size;    // block edge length (e.g. 4 or 8)
uniform float u_threshold;   // alive threshold for occupancy

layout(r8ui, binding=0) writeonly uniform uimage3D u_occupancy;
layout(rg16f, binding=1) writeonly uniform image3D u_minmax;

void main() {
    ivec3 block = ivec3(gl_GlobalInvocationID);
    int occ_size = (u_size + u_block_size - 1) / u_block_size;
    if (any(greaterThanEqual(block, ivec3(occ_size)))) return;

    ivec3 base = block * u_block_size;
    bool found = false;
    float lo =  1e10;
    float hi = -1e10;

    for (int dz = 0; dz < u_block_size; dz++) {
        for (int dy = 0; dy < u_block_size; dy++) {
            for (int dx = 0; dx < u_block_size; dx++) {
                ivec3 pos = base + ivec3(dx, dy, dz);
                if (any(greaterThanEqual(pos, ivec3(u_size)))) continue;
                float v = texelFetch(u_view, pos, 0).r;
                if (v > u_threshold) found = true;
                lo = min(lo, v);
                hi = max(hi, v);
            }
        }
    }

    imageStore(u_occupancy, block, uvec4(found ? 1u : 0u, 0u, 0u, 0u));
    imageStore(u_minmax,    block, vec4(lo, hi, 0.0, 0.0));
}
