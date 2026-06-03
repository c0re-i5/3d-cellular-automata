
#version 430
layout(local_size_x=1) in;
layout(std430, binding=4) buffer DrawIndirect {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint baseInstance;
};
layout(std430, binding=6) buffer TotalCounter {
    uint totalVisibleCount;
};
uniform int u_reset_total;  // 1 = also reset frame-level total (first chunk only)
uniform int u_vertex_count; // per-instance vertex count (36 cube / 72 rhombic-dodec)
void main() {
    vertexCount = uint(u_vertex_count);
    instanceCount = 0u;
    firstVertex = 0u;
    baseInstance = 0u;
    if (u_reset_total == 1)
        totalVisibleCount = 0u;
}
