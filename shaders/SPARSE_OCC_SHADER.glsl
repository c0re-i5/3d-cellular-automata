
#version 430
// One workgroup per 8³ block; 512 threads each fetch ONE voxel and
// OR the result into shared memory. Replaces a previous serial
// 512-iter per-thread loop that ran ~100× slower at 384³ due to
// fully sequential texelFetch chains within each warp.
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;
uniform sampler3D u_volume;
uniform int u_size;       // voxel grid edge
layout(r8ui, binding=0) uniform restrict writeonly uimage3D u_occ_out;
shared uint s_occ;
void main() {
    if (gl_LocalInvocationIndex == 0u) s_occ = 0u;
    barrier();

    ivec3 b = ivec3(gl_WorkGroupID);
    ivec3 p = b * 8 + ivec3(gl_LocalInvocationID);
    if (all(lessThan(p, ivec3(u_size)))) {
        // ANY channel != 0 counts as occupied (covers multi-field rules).
        vec4 v = texelFetch(u_volume, p, 0);
        if (any(notEqual(v, vec4(0.0)))) {
            atomicOr(s_occ, 1u);
        }
    }
    barrier();
    if (gl_LocalInvocationIndex == 0u) {
        imageStore(u_occ_out, b, uvec4(s_occ, 0u, 0u, 0u));
    }
}
