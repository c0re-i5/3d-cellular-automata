
#version 430
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

layout(rgba32f, binding=0) readonly uniform image3D u_grid;

layout(std430, binding=7) buffer DebugStats {
    uint finite_count[4];
    uint nan_count[4];
    uint inf_count[4];
    uint min_bits[4];
    uint max_bits[4];
    uint sum_fp[4];
    uint sumsq_fp[4];
    uint active_count;
    uint bbox_min[3];
    uint bbox_max[3];
    uint com_sum[3];
    uint rg_sum;
    uint boundary_count;
    uint hist[4 * 64];
};

uniform int   u_size;
uniform int   u_active_channel;   // which channel defines "active"
uniform float u_active_threshold; // active iff value > threshold (or |v|>thr in mode==2)
uniform int   u_active_mode;      // 0=>thr, 1=>thr, 2=>abs(thr), 3=>element-id
uniform int   u_boundary_shell;   // shell thickness (default 4)
uniform float u_hist_min[4];      // previous frame's min per channel
uniform float u_hist_max[4];      // previous frame's max per channel
uniform float u_norm_n;           // 1.0 / total_cells

// Encoded constants (must match Python decoder)
#define BIAS      1.0e6
#define SCALE_S   1.0e6
#define SCALE_Q   1.0e3

// Workgroup-shared accumulators -- one atomic per workgroup per metric.
// (Atomic-bombing the global SSBO from every thread would be ~5-50x
//  slower at 256³ on real hardware.)
shared uint s_finite[4];
shared uint s_nan[4];
shared uint s_inf[4];
shared uint s_min_bits[4];
shared uint s_max_bits[4];
shared int  s_sum_int[4];     // signed local sum, fixed-point
shared uint s_sumsq[4];       // unsigned local sum-of-squares, fixed-point
shared uint s_active;
shared uint s_bbox_min[3];
shared uint s_bbox_max[3];
shared uint s_com[3];
shared uint s_rg;
shared uint s_boundary;

bool is_active(vec4 cell) {
    float v = cell[u_active_channel];
    if (u_active_mode == 3) return abs(v) > 0.5 && abs(v - 119.0) > 0.5;
    if (u_active_mode == 2) return abs(v) > u_active_threshold;
    return v > u_active_threshold;
}

void main() {
    uint lid = gl_LocalInvocationIndex;

    // ── Init shared memory (1 thread per slot) ───────────────────────
    if (lid < 4u) {
        s_finite[lid] = 0u;
        s_nan[lid] = 0u;
        s_inf[lid] = 0u;
        // Min sentinel: a huge positive biased value (~floatBitsToUint(1e30),
        // a large positive uint). atomicMin against any real biased data
        // (BIAS ± v, a normal positive float bit-pattern ≪ 1e30 bits)
        // replaces the sentinel with the real value.
        s_min_bits[lid] = floatBitsToUint(BIAS + 1.0e30);
        // Max sentinel: 0u, the smallest possible uint. atomicMax against any
        // real biased data (positive uint, since BIAS keeps things > 0) wins.
        // CRITICAL: do NOT use floatBitsToUint(BIAS - 1e30); that float is
        // negative, which sets the sign bit and produces a HUGE uint that
        // beats every real datum in atomicMax (the bug that pinned max to
        // −1e30 in the first JSON dump).
        s_max_bits[lid] = 0u;
        s_sum_int[lid] = 0;
        s_sumsq[lid] = 0u;
    }
    if (lid < 3u) {
        s_bbox_min[lid] = uint(u_size);   // start large (= no active yet)
        s_bbox_max[lid] = 0u;             // start small
        s_com[lid] = 0u;
    }
    if (lid == 0u) {
        s_active = 0u; s_rg = 0u; s_boundary = 0u;
    }
    barrier();

    ivec3 pos = ivec3(gl_GlobalInvocationID);
    bool in_grid = all(lessThan(pos, ivec3(u_size)));

    if (in_grid) {
        vec4 cur = imageLoad(u_grid, pos);

        // ── Per-channel scalar stats ────────────────────────────────
        for (int c = 0; c < 4; c++) {
            float v = cur[c];
            if (isnan(v)) {
                atomicAdd(s_nan[c], 1u);
                continue;
            }
            if (isinf(v)) {
                atomicAdd(s_inf[c], 1u);
                continue;
            }
            atomicAdd(s_finite[c], 1u);
            // Keep per-cell accumulation UN-normalized so small values
            // don't truncate to zero (the bug that pinned ch0_mean=0 at
            // size>=128). Workgroup local sum is bounded by
            //   512 * |v|_max * SCALE_S  ~ 5.12e8  -> fits int32.
            // We normalize by u_norm_n only at flush-to-global time.
            int s_int = int(v * SCALE_S);
            // sum-of-squares is non-negative; 512 * v²_max * SCALE_Q
            // ~ 5e5..5e8 depending on rule -> fits uint32.
            uint sq_uint = uint(v * v * SCALE_Q);
            atomicAdd(s_sum_int[c], s_int);
            atomicAdd(s_sumsq[c], sq_uint);
            // Min/max via float-bits + bias for atomicMin/Max ordering.
            uint biased_bits = floatBitsToUint(v + BIAS);
            atomicMin(s_min_bits[c], biased_bits);
            atomicMax(s_max_bits[c], biased_bits);

            // Histogram. Use previous frame's range so we have stable bins.
            float lo = u_hist_min[c];
            float hi = u_hist_max[c];
            float span = max(hi - lo, 1.0e-12);
            int bin = int(clamp((v - lo) / span * 64.0, 0.0, 63.0));
            atomicAdd(hist[c * 64 + bin], 1u);
        }

        // ── Active-mask spatial stats ───────────────────────────────
        if (is_active(cur)) {
            atomicAdd(s_active, 1u);
            atomicMin(s_bbox_min[0], uint(pos.x));
            atomicMin(s_bbox_min[1], uint(pos.y));
            atomicMin(s_bbox_min[2], uint(pos.z));
            atomicMax(s_bbox_max[0], uint(pos.x));
            atomicMax(s_bbox_max[1], uint(pos.y));
            atomicMax(s_bbox_max[2], uint(pos.z));
            atomicAdd(s_com[0], uint(pos.x));
            atomicAdd(s_com[1], uint(pos.y));
            atomicAdd(s_com[2], uint(pos.z));
            // Radius-of-gyration accumulator: sum of |p - center|² where
            // center = size/2. Stored as fixed-point: divide by size² so
            // the per-cell contribution is in [0, 0.75] before scaling.
            vec3 d = vec3(pos) - vec3(u_size) * 0.5;
            float r2_norm = dot(d, d) / float(u_size * u_size);
            atomicAdd(s_rg, uint(r2_norm * 1024.0));
            // Boundary shell: count active cells within u_boundary_shell
            // of any face.
            int sh = u_boundary_shell;
            if (pos.x < sh || pos.x >= u_size - sh ||
                pos.y < sh || pos.y >= u_size - sh ||
                pos.z < sh || pos.z >= u_size - sh) {
                atomicAdd(s_boundary, 1u);
            }
        }
    }

    barrier();

    // ── Flush workgroup totals to global SSBO (one atomic each) ─────
    if (lid < 4u) {
        if (s_finite[lid] > 0u) atomicAdd(finite_count[lid], s_finite[lid]);
        if (s_nan[lid]    > 0u) atomicAdd(nan_count[lid],    s_nan[lid]);
        if (s_inf[lid]    > 0u) atomicAdd(inf_count[lid],    s_inf[lid]);
        // Min/max: ONLY flush when this workgroup actually saw finite data.
        // Otherwise the sentinels (huge-uint for min, 0 for max) would
        // corrupt the global running min/max via the atomic ops.
        if (s_finite[lid] > 0u) {
            atomicMin(min_bits[lid], s_min_bits[lid]);
            atomicMax(max_bits[lid], s_max_bits[lid]);
        }
        // Normalize workgroup totals by 1/N on flush. Each global-atomic
        // contribution is tiny per workgroup but sums correctly across
        // the whole grid. Signed->uint reinterpret preserves the bit
        // pattern so Python's int32 view decodes negatives properly.
        // CRITICAL: use round-to-nearest-even, not truncation. At huge
        // grids (512³) per-workgroup normalized values can be <1 and
        // `int(x)` would systematically truncate them to 0, losing the
        // entire signal.
        float sum_norm_f   = float(s_sum_int[lid]) * u_norm_n;
        float sumsq_norm_f = float(s_sumsq[lid])   * u_norm_n;
        int  ws_sum_norm   = int(floor(sum_norm_f + sign(sum_norm_f) * 0.5));
        uint ws_sumsq_norm = uint(sumsq_norm_f + 0.5);
        if (ws_sum_norm  != 0)  atomicAdd(sum_fp[lid],   uint(ws_sum_norm));
        if (ws_sumsq_norm > 0u) atomicAdd(sumsq_fp[lid], ws_sumsq_norm);
    }
    if (lid < 3u) {
        atomicMin(bbox_min[lid], s_bbox_min[lid]);
        atomicMax(bbox_max[lid], s_bbox_max[lid]);
        if (s_com[lid] > 0u) atomicAdd(com_sum[lid], s_com[lid]);
    }
    if (lid == 0u) {
        if (s_active > 0u)   atomicAdd(active_count,   s_active);
        if (s_rg > 0u)       atomicAdd(rg_sum,         s_rg);
        if (s_boundary > 0u) atomicAdd(boundary_count, s_boundary);
    }
}
