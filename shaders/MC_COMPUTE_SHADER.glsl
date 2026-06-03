
#version 430
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

uniform sampler3D u_field_tex;
uniform int   u_size;
uniform int   u_channel;     // 0=R, 1=G, 2=B, 3=A
uniform int   u_use_abs;
uniform float u_iso;
uniform int   u_max_verts;

// Vertex output: vec4 pos, vec4 normal — 32 B per vertex.
layout(std430, binding=10) buffer MCVerts { vec4 verts[]; } V;
layout(std430, binding=11) buffer MCCounter { uint vert_count; uint pad[3]; } C;

// Triangulation LUT (256 cases × 16 entries, -1 = end).
layout(std430, binding=12) readonly buffer MCTriTable { int tri_table[]; } T;

float fs;

float sampleScalar(ivec3 p) {
    p = clamp(p, ivec3(0), ivec3(u_size - 1));
    vec4 v = texelFetch(u_field_tex, p, 0);
    float s;
    if (u_channel == 0) s = v.r;
    else if (u_channel == 1) s = v.g;
    else if (u_channel == 2) s = v.b;
    else s = v.a;
    if (u_use_abs != 0) s = abs(s);
    return s;
}

vec3 vertexInterp(vec3 p1, vec3 p2, float v1, float v2) {
    float denom = v2 - v1;
    if (abs(denom) < 1e-6) return 0.5 * (p1 + p2);
    float t = clamp((u_iso - v1) / denom, 0.0, 1.0);
    return mix(p1, p2, t);
}

void main() {
    fs = float(u_size);
    ivec3 cell = ivec3(gl_GlobalInvocationID);
    if (cell.x >= u_size - 1 || cell.y >= u_size - 1 || cell.z >= u_size - 1) return;

    // 8 corners (Lorensen ordering: 0..7 = (0,0,0),(1,0,0),(1,1,0),(0,1,0),
    //                                       (0,0,1),(1,0,1),(1,1,1),(0,1,1))
    ivec3 ofs[8] = ivec3[8](
        ivec3(0,0,0), ivec3(1,0,0), ivec3(1,1,0), ivec3(0,1,0),
        ivec3(0,0,1), ivec3(1,0,1), ivec3(1,1,1), ivec3(0,1,1)
    );
    float val[8];
    vec3  pos[8];
    int caseIndex = 0;
    for (int i = 0; i < 8; ++i) {
        ivec3 p = cell + ofs[i];
        val[i] = sampleScalar(p);
        pos[i] = vec3(p);
        if (val[i] >= u_iso) caseIndex |= (1 << i);
    }
    if (caseIndex == 0 || caseIndex == 255) return;

    // 12 edges, each connects two corners (Lorensen edge ordering).
    int e_a[12] = int[12](0,1,2,3,4,5,6,7,0,1,2,3);
    int e_b[12] = int[12](1,2,3,0,5,6,7,4,4,5,6,7);

    // Pre-interpolate the up to 12 active edge vertices.
    vec3 edge_v[12];
    for (int e = 0; e < 12; ++e) {
        int a = e_a[e]; int b = e_b[e];
        // Cheap to just always compute; tri table picks the ones we need.
        edge_v[e] = vertexInterp(pos[a], pos[b], val[a], val[b]);
    }

    int base = caseIndex * 16;
    for (int t = 0; t < 16; t += 3) {
        int e0 = T.tri_table[base + t];
        if (e0 < 0) break;
        int e1 = T.tri_table[base + t + 1];
        int e2 = T.tri_table[base + t + 2];
        vec3 a = edge_v[e0];
        vec3 b = edge_v[e1];
        vec3 c = edge_v[e2];
        vec3 n = cross(b - a, c - a);
        float nlen = length(n);
        if (nlen < 1e-8) continue;
        n /= nlen;

        uint vid = atomicAdd(C.vert_count, 3u);
        if (vid + 3u > uint(u_max_verts)) return;
        V.verts[2u*vid    ] = vec4(a, val[0]);
        V.verts[2u*vid + 1u] = vec4(n, 0.0);
        V.verts[2u*(vid+1u)    ] = vec4(b, val[0]);
        V.verts[2u*(vid+1u) + 1u] = vec4(n, 0.0);
        V.verts[2u*(vid+2u)    ] = vec4(c, val[0]);
        V.verts[2u*(vid+2u) + 1u] = vec4(n, 0.0);
    }
}
