
#version 430
layout(local_size_x=4, local_size_y=4, local_size_z=4) in;

uniform sampler3D u_minmax;   // RG16F: per-block (min, max) of view tex
uniform int u_dim;            // mipmap dimension along each axis

layout(std430, binding=0) buffer RangeBuf {
    uint range_min_bits;  // float bits of (min + 10.0)
    uint range_max_bits;  // float bits of (max + 10.0)
};

shared float s_lo[64];
shared float s_hi[64];

void main() {
    ivec3 p = ivec3(gl_GlobalInvocationID);
    uint lid = gl_LocalInvocationIndex;
    float lo =  1e10;
    float hi = -1e10;
    if (all(lessThan(p, ivec3(u_dim)))) {
        vec2 mm = texelFetch(u_minmax, p, 0).rg;
        lo = mm.r;
        hi = mm.g;
    }
    s_lo[lid] = lo;
    s_hi[lid] = hi;
    barrier();
    // Tree reduction in shared memory (64 -> 1)
    for (uint stride = 32u; stride > 0u; stride >>= 1) {
        if (lid < stride) {
            s_lo[lid] = min(s_lo[lid], s_lo[lid + stride]);
            s_hi[lid] = max(s_hi[lid], s_hi[lid + stride]);
        }
        barrier();
    }
    if (lid == 0u) {
        // Float-as-uint atomic min/max via +10 bias to keep IEEE bits
        // monotonic even for negative inputs (range is [-10, +Inf)).
        atomicMin(range_min_bits, floatBitsToUint(s_lo[0] + 10.0));
        atomicMax(range_max_bits, floatBitsToUint(s_hi[0] + 10.0));
    }
}
