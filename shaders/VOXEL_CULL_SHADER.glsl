
#version 430
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

#ifdef SPARSE_DISPATCH
// Brick-list cull: one workgroup per ACTIVE 8³ block, fetched from
// _sparse_blocks_ssbo. Skips traversal of empty / fully-buried regions
// entirely — dense fields with sparse surfaces win the most.
layout(std430, binding=11) readonly buffer ActiveBlocks {
    uvec4 sparse_blocks[];
};
#endif

// Read-only sampler for texture-cache-friendly access (vs imageLoad)
uniform sampler3D u_volume_tex;

layout(std430, binding=3) buffer VoxelBuffer {
    uint voxels[];   // 1 uint per voxel (position only, 4 bytes)
};

layout(std430, binding=4) buffer DrawIndirect {
    uint vertexCount;      // = 36 (set by CPU)
    uint instanceCount;    // written by this shader via atomicAdd
    uint firstVertex;      // = 0 (set by CPU)
    uint baseInstance;     // = 0 (set by CPU)
};

layout(std430, binding=6) buffer TotalCounter {
    uint totalVisibleCount;  // accumulates across all chunks (reset once per frame)
};

uniform int u_size;
uniform ivec3 u_dims;       // (W, H, D) — per-axis grid bounds
uniform float u_threshold;  // visibility threshold for non-element CAs
uniform int u_is_element_ca; // 1 = element mode, 0 = standard mode
uniform int u_max_voxels;
uniform int u_channel;      // which channel to read (0=R, 1=G, 2=B, 3=A)
uniform int u_use_abs;      // 1 = use abs(value) for alive test (wave mode)
// Multi-channel vis_mode (mirrors VIEW_TEX_BUILD_SHADER):
//   0 = DENSITY (legacy, use u_channel)
//   1 = RGB_CHANNELS  (alive = length(rgb)/sqrt(3))
//   4 = RGBA_BLEND    (alive = .a)
// Modes 2/3 are treated like DENSITY (single channel is fine for cull).
uniform int u_vis_mode;
// Chunk bounds for multi-pass rendering (default: full grid)
uniform ivec3 u_chunk_min;  // inclusive lower corner
uniform ivec3 u_chunk_max;  // exclusive upper corner

// Shared memory tile: 10x10x10 halo around the 8x8x8 workgroup.
// Only stores the scalar alive-test value (1 float per cell).
shared float s_tile[10][10][10];

float get_alive_value(vec4 c) {
    if (u_is_element_ca == 1) return c.r;
    if (u_vis_mode == 1) {
        // Team-coloured paint (rgb_channels): max of any channel proxies
        // visibility well and avoids the sqrt(3) suppression of
        // single-team voxels (red voxel has rgb=(1,0,0) -> length=1).
        return max(max(c.r, c.g), c.b);
    }
    if (u_vis_mode == 4) return c.a;
    float v = c[u_channel];
    if (u_use_abs == 1) v = abs(v);
    else if (u_use_abs == 2) v = abs(v);  // signed bipolar: visibility = magnitude
    return v;
}

// Fetch alive-test value, returning 0.0 (dead) for out-of-bounds cells.
// This prevents boundary cells from falsely seeing themselves as neighbors.
float fetch_alive(ivec3 gp) {
    if (any(lessThan(gp, ivec3(0))) || any(greaterThanEqual(gp, u_dims)))
        return 0.0;
    return get_alive_value(texelFetch(u_volume_tex, gp, 0));
}

bool is_alive(float v) {
    if (u_is_element_ca == 1) return int(round(v)) > 0;
    return v > u_threshold;
}

void main() {
    ivec3 chunk_size = u_chunk_max - u_chunk_min;
    ivec3 local = ivec3(gl_LocalInvocationID);
#ifdef SPARSE_DISPATCH
    // Each workgroup handles one active 8³ block from the compact list.
    uvec4 b = sparse_blocks[gl_WorkGroupID.x];
    ivec3 wg_origin = ivec3(b.xyz) * 8;
#else
    ivec3 wg_origin = ivec3(gl_WorkGroupID) * 8 + u_chunk_min;
#endif

    // ── Load 10x10x10 shared memory tile (8³ core + 1-cell halo) ──
    // Each thread in the 8³ workgroup loads its own cell at offset +1
    {
        ivec3 gp = wg_origin + local;
        s_tile[local.z + 1][local.y + 1][local.x + 1] = fetch_alive(gp);
    }

    // Halo: threads on the faces of the 8³ block load boundary cells.
    // Uses fetch_alive() which returns 0.0 for out-of-bounds positions.
    // X-axis halo (left/right)
    if (local.x == 0) {
        s_tile[local.z + 1][local.y + 1][0] =
            fetch_alive(wg_origin + ivec3(-1, local.y, local.z));
    }
    if (local.x == 7) {
        s_tile[local.z + 1][local.y + 1][9] =
            fetch_alive(wg_origin + ivec3(8, local.y, local.z));
    }
    // Y-axis halo (bottom/top)
    if (local.y == 0) {
        s_tile[local.z + 1][0][local.x + 1] =
            fetch_alive(wg_origin + ivec3(local.x, -1, local.z));
    }
    if (local.y == 7) {
        s_tile[local.z + 1][9][local.x + 1] =
            fetch_alive(wg_origin + ivec3(local.x, 8, local.z));
    }
    // Z-axis halo (front/back)
    if (local.z == 0) {
        s_tile[0][local.y + 1][local.x + 1] =
            fetch_alive(wg_origin + ivec3(local.x, local.y, -1));
    }
    if (local.z == 7) {
        s_tile[9][local.y + 1][local.x + 1] =
            fetch_alive(wg_origin + ivec3(local.x, local.y, 8));
    }

    barrier();
    memoryBarrierShared();

    // ── Surface cull using shared memory ──
    ivec3 pos = wg_origin + local;
#ifndef SPARSE_DISPATCH
    if (any(greaterThanEqual(pos, u_chunk_max))) return;
#endif
    if (any(greaterThanEqual(pos, u_dims))) return;

    // Read own cell from shared memory (offset +1 for halo)
    ivec3 s = local + 1;  // shared memory coords
    float myVal = s_tile[s.z][s.y][s.x];
    if (!is_alive(myVal)) return;

    // Check 6 neighbors from shared memory — no global reads
    int solid_neighbors = 0;
    if (is_alive(s_tile[s.z][s.y][s.x - 1])) solid_neighbors++;  // -X
    if (is_alive(s_tile[s.z][s.y][s.x + 1])) solid_neighbors++;  // +X
    if (is_alive(s_tile[s.z][s.y - 1][s.x])) solid_neighbors++;  // -Y
    if (is_alive(s_tile[s.z][s.y + 1][s.x])) solid_neighbors++;  // +Y
    if (is_alive(s_tile[s.z - 1][s.y][s.x])) solid_neighbors++;  // -Z
    if (is_alive(s_tile[s.z + 1][s.y][s.x])) solid_neighbors++;  // +Z
    if (solid_neighbors >= 6) return;  // fully buried, skip

    // Compute a richer surface-darkening hint: 0..6 raw face-neighbor count
    // mapped to 0..30 so it fits in a 5-bit field with smooth steps. We oversample
    // the corner contributions by also weighting the 12 edge neighbors (each
    // counted as 0.5) and 8 corner neighbors (each as 0.25). This produces
    // ambient-occlusion-like shading without any extra global memory traffic
    // (everything lives in s_tile which is already populated for the surface cull).
    int aoc = solid_neighbors * 4;  // face neighbors weighted x4 (range 0..24)
    // Edge (12) and corner (8) contributions for AO darkening
    if (is_alive(s_tile[s.z][s.y - 1][s.x - 1])) aoc++;
    if (is_alive(s_tile[s.z][s.y - 1][s.x + 1])) aoc++;
    if (is_alive(s_tile[s.z][s.y + 1][s.x - 1])) aoc++;
    if (is_alive(s_tile[s.z][s.y + 1][s.x + 1])) aoc++;
    if (is_alive(s_tile[s.z - 1][s.y][s.x - 1])) aoc++;
    if (is_alive(s_tile[s.z - 1][s.y][s.x + 1])) aoc++;
    if (is_alive(s_tile[s.z + 1][s.y][s.x - 1])) aoc++;
    if (is_alive(s_tile[s.z + 1][s.y][s.x + 1])) aoc++;
    // Cap the AO score in the shade-bit field range
    uint shade_hint = uint(min(aoc, int(__SHADE_MAX__)));

    // Append position only (1 uint = 4 bytes per voxel)
    atomicAdd(totalVisibleCount, 1u);  // frame-level total (not reset per chunk)
    uint idx = atomicAdd(instanceCount, 1u);
    if (idx >= uint(u_max_voxels)) return;

    // Pack: __PB__ bits per axis (xyz) + __SB__-bit shading.
    // Substituted at compile time based on max(W,H,D): 9/5 for size<=512,
    // 10/2 for size<=1024 (loses smooth AO levels but supports 1024^3).
    voxels[idx] = (uint(pos.x) & __PMASK__)
                | ((uint(pos.y) & __PMASK__) << __PB__)
                | ((uint(pos.z) & __PMASK__) << (2u * __PB__))
                | ((shade_hint  & __SMASK__) << (3u * __PB__));
}
