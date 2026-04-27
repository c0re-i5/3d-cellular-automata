#!/usr/bin/env python3
"""
3D CA Test Harness — headless GPU diagnostics, parameter sweeps, and discovery search.

Runs CA rules on the GPU without a visible window, measures behavioral metrics,
scores interestingness, and supports automated parameter space exploration.

Metrics measured per run:
  - alive_count: cells above threshold at each sample step
  - alive_ratio: fraction of grid that's active
  - mean/std/min/max: value statistics per channel
  - activity: fraction of cells that changed between consecutive steps
  - surface_ratio: surface cells / total alive (structural complexity)
  - has_nan/has_inf: numerical stability
  - stability: variance in alive_ratio over the run (low = stable)

Interestingness score (0-1):
  - Penalizes: dead (all 0), saturated (all 1), NaN/Inf, no change
  - Rewards: moderate alive ratio, spatial structure, sustained activity

Usage:
    python3 test_harness.py                        # audit all presets
    python3 test_harness.py --rule smoothlife_3d   # test one rule
    python3 test_harness.py --sweep game_of_life_3d --trials 100
    python3 test_harness.py --search lenia_3d --trials 200 --top 10
"""

import sys, os, json, time, argparse
import warnings
import numpy as np

try:
    from OpenGL import GL  # type: ignore
except Exception:  # pragma: no cover -- already optional in step()
    GL = None  # type: ignore

# Suppress numpy overflow/invalid warnings from degenerate CA states (all-NaN grids, etc.)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# ── Headless GPU context ──────────────────────────────────────────────

def create_headless_context():
    """Create an invisible GLFW window + moderngl context for compute."""
    import glfw
    import moderngl
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(64, 64, "test", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Window creation failed")
    glfw.make_context_current(window)
    ctx = moderngl.create_context()
    return window, ctx


def destroy_context(window):
    import glfw
    glfw.destroy_window(window)
    glfw.terminate()


# ── Lightweight simulation runner (no rendering, no UI) ───────────────

# Module-level shader cache: avoids recompiling the same compute shader
# every trial.  Keyed on (context id, shader source hash).
_shader_cache = {}

# Module-level texture pool: avoids alloc/dealloc of 3D textures every trial.
# Keyed on (context id, size, dtype).  Each entry is a list of (tex_a, tex_b) pairs.
_texture_pool = {}
# Shared element SSBO (same data every time)
_element_ssbo_cache = {}

# ── Per-phase timing (opt-in via CA_HARNESS_PROFILE=1) ───────────────
# Lightweight: O(few ns) per perf_counter call when enabled, O(0) when off.
# Phases tracked per trial: init, step, read, metrics, dynamics, release.
_PROFILE_ENABLED = os.environ.get('CA_HARNESS_PROFILE', '0') == '1'
_profile_totals = {'init': 0.0, 'step': 0.0, 'read': 0.0,
                   'metrics': 0.0, 'dynamics': 0.0, 'release': 0.0,
                   'wall': 0.0, 'trials': 0, 'steps': 0, 'reads': 0}

def _profile_print():
    """Print rollup of per-phase timings. Called at end of cmd_search."""
    if not _PROFILE_ENABLED or _profile_totals['trials'] == 0:
        return
    t = _profile_totals
    n = t['trials']
    print()
    print(f"=== PROFILE ({n} trials, {t['steps']} steps, {t['reads']} reads) ===")
    accounted = sum(t[k] for k in ('init', 'step', 'read', 'metrics',
                                    'dynamics', 'release'))
    other = max(0.0, t['wall'] - accounted)
    rows = [
        ('init',     t['init'],     t['init']     / n * 1000),
        ('step',     t['step'],     t['step']     / max(1, t['steps']) * 1000),
        ('read',     t['read'],     t['read']     / max(1, t['reads']) * 1000),
        ('metrics',  t['metrics'],  t['metrics']  / max(1, t['reads']) * 1000),
        ('dynamics', t['dynamics'], t['dynamics'] / n * 1000),
        ('release',  t['release'],  t['release']  / n * 1000),
        ('other',    other,         other         / n * 1000),
    ]
    print(f"  {'phase':<10} {'total(s)':>10} {'per-unit(ms)':>14} {'%':>6}")
    for name, total, per in rows:
        pct = 100.0 * total / max(1e-9, t['wall'])
        print(f"  {name:<10} {total:>10.3f} {per:>14.3f} {pct:>5.1f}%")
    print(f"  {'WALL':<10} {t['wall']:>10.3f} {t['wall']/n*1000:>14.3f}  100.0%")


# ── GPU-side metrics (opt-out via CA_HARNESS_GPU_METRICS=0) ──────────
# Replaces compute_metrics()'s 6+ whole-grid CPU passes with one (or two)
# compute dispatches plus a tiny ~96-byte SSBO readback. At size=256 the
# CPU read+metrics phase drops from ~360ms/sample to ~10-20ms/sample.
#
# Two-pass dispatch is required for `deviation` mode (alive iff |v - mean|
# > thr) because the mean is needed before thresholding. All other modes
# fit in a single dispatch.
_GPU_METRICS_ENABLED = os.environ.get('CA_HARNESS_GPU_METRICS', '1') == '1'

# Mode IDs (must match GLSL switch in HARNESS_METRICS_SHADER):
_MODE_ID = {
    'discrete':         0,   # alive iff v > thr
    'continuous':       0,   # same
    'wave':             1,   # alive iff |v| > thr
    'element':          2,   # non-vacuum & non-wall (uses ch0)
    'deviation':        3,   # alive iff |v - field_mean| > thr (needs 2 passes)
    'phase_coherence':  4,   # alive iff incoherence > thr (uses 6-nb sin)
}

# Layout of the metrics SSBO. All slots are uint32. Floats are bit-encoded
# via floatBitsToUint(v + BIAS) so atomicMin/Max work; sums are fixed-point
# pre-divided by a per-workgroup divisor so the global sum fits int32.
_METRICS_NUINTS = 16   # finite, nan, inf, min, max, sum, sumsq,
                       # alive, surface, activity, mean_sum_for_deviation,
                       # 5 spare
_METRICS_BIAS    = 1.0e6
_METRICS_SCALE_S = 1.0e6
_METRICS_SCALE_Q = 1.0e3

# Single-pass shader: computes per-channel scalar stats (on `measure_channel`
# only — the 4-channel version was unused by run_trial), the alive mask,
# surface count, and activity (vs prev_grid).
HARNESS_METRICS_SHADER = """
#version 430
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

layout(rgba32f, binding=0) readonly uniform image3D u_grid;
layout(rgba32f, binding=1) readonly uniform image3D u_prev;

layout(std430, binding=2) buffer Stats {
    uint finite_count;
    uint nan_count;
    uint inf_count;
    uint min_bits;        // floatBitsToUint(v + BIAS) of measured channel
    uint max_bits;
    uint sum_fp;          // signed (cast int->uint) fixed-point, pre-normalized
    uint sumsq_fp;        // unsigned fixed-point, pre-normalized
    uint alive_count;
    uint surface_count;
    uint activity_count;
    uint deviation_mean_fp;   // dispatch-1 output for deviation mode
    uint pad11;
    uint pad12;
    uint pad13;
    uint pad14;
    uint pad15;
};

uniform int   u_size;
uniform int   u_channel;
uniform int   u_mode;             // 0=thr, 1=abs(thr), 2=element, 3=deviation, 4=phase_coh
uniform float u_alive_thr;
uniform float u_change_thr;
uniform int   u_have_prev;        // 0/1 — skip activity when 0
uniform int   u_is_element;       // also count temp(ch1) changes for activity
uniform float u_field_mean;       // pre-computed mean for deviation mode (pass 2)
uniform float u_norm_n;           // 1.0 / per-wg flush divisor

#define BIAS    1.0e6
#define SCALE_S 1.0e6
#define SCALE_Q 1.0e3
#define TWO_PI  6.283185307179586

shared uint s_finite;
shared uint s_nan;
shared uint s_inf;
shared uint s_min_bits;
shared uint s_max_bits;
shared int  s_sum;
shared uint s_sumsq;
shared uint s_alive;
shared uint s_surface;
shared uint s_activity;

bool is_alive(vec4 cell, ivec3 pos) {
    float v = cell[u_channel];
    if (u_mode == 2) {
        // element: non-vacuum (|v|>0.5) AND non-wall (|v-119|>0.5)
        return abs(v) > 0.5 && abs(v - 119.0) > 0.5;
    }
    if (u_mode == 1) return abs(v) > u_alive_thr;
    if (u_mode == 3) return abs(v - u_field_mean) > u_alive_thr;
    if (u_mode == 4) {
        // phase_coherence: alive iff mean |sin(nb_phase - phase)| > thr.
        // Phase stored as v in [0,1] = [0, 2pi].
        float phase = v * TWO_PI;
        float incoh = 0.0;
        ivec3 nbs[6] = ivec3[6](
            ivec3( 1, 0, 0), ivec3(-1, 0, 0),
            ivec3( 0, 1, 0), ivec3( 0,-1, 0),
            ivec3( 0, 0, 1), ivec3( 0, 0,-1)
        );
        for (int i = 0; i < 6; i++) {
            ivec3 np = (pos + nbs[i] + ivec3(u_size)) % ivec3(u_size);
            float nv = imageLoad(u_grid, np)[u_channel] * TWO_PI;
            incoh += abs(sin(nv - phase));
        }
        incoh *= (1.0 / 6.0);
        return incoh > u_alive_thr;
    }
    return v > u_alive_thr;
}

void main() {
    uint lid = gl_LocalInvocationIndex;
    if (lid == 0u) {
        s_finite = 0u; s_nan = 0u; s_inf = 0u;
        s_min_bits = floatBitsToUint(BIAS + 1.0e30);
        s_max_bits = 0u;
        s_sum = 0; s_sumsq = 0u;
        s_alive = 0u; s_surface = 0u; s_activity = 0u;
    }
    barrier();

    ivec3 pos = ivec3(gl_GlobalInvocationID);
    bool in_grid = all(lessThan(pos, ivec3(u_size)));

    if (in_grid) {
        vec4 cur = imageLoad(u_grid, pos);
        float v = cur[u_channel];

        // ── Scalar stats on measured channel ────────────────────────
        if (isnan(v))      atomicAdd(s_nan, 1u);
        else if (isinf(v)) atomicAdd(s_inf, 1u);
        else {
            atomicAdd(s_finite, 1u);
            // For wave mode (mode==1), stats are on |v| not v -- matches CPU path
            float vstat = (u_mode == 1) ? abs(v) : v;
            int s_int = int(vstat * SCALE_S);
            uint sq_uint = uint(vstat * vstat * SCALE_Q);
            atomicAdd(s_sum, s_int);
            atomicAdd(s_sumsq, sq_uint);
            uint bits = floatBitsToUint(vstat + BIAS);
            atomicMin(s_min_bits, bits);
            atomicMax(s_max_bits, bits);
        }

        // ── Alive mask & surface count ──────────────────────────────
        bool alive = is_alive(cur, pos);
        if (alive) {
            atomicAdd(s_alive, 1u);
            // Surface = alive AND at least one of 6 toroidal neighbours dead.
            // Note: for phase_coherence mode this re-runs is_alive on each
            // neighbour (6 more 6-neighbour scans = 36 extra fetches per
            // alive cell). Acceptable: phase_coherence is one rule.
            bool has_dead_nb = false;
            ivec3 nbs[6] = ivec3[6](
                ivec3( 1, 0, 0), ivec3(-1, 0, 0),
                ivec3( 0, 1, 0), ivec3( 0,-1, 0),
                ivec3( 0, 0, 1), ivec3( 0, 0,-1)
            );
            for (int i = 0; i < 6; i++) {
                ivec3 np = (pos + nbs[i] + ivec3(u_size)) % ivec3(u_size);
                if (!is_alive(imageLoad(u_grid, np), np)) {
                    has_dead_nb = true;
                    break;
                }
            }
            if (has_dead_nb) atomicAdd(s_surface, 1u);
        }

        // ── Activity (vs prev_grid) ─────────────────────────────────
        if (u_have_prev != 0) {
            vec4 prev = imageLoad(u_prev, pos);
            bool changed = abs(v - prev[u_channel]) > u_change_thr;
            if (u_is_element != 0 && !changed) {
                changed = abs(cur.y - prev.y) > 1.0;
            }
            if (changed) atomicAdd(s_activity, 1u);
        }
    }

    barrier();

    // ── Flush workgroup totals to global SSBO ───────────────────────
    if (lid == 0u) {
        if (s_finite > 0u) atomicAdd(finite_count,    s_finite);
        if (s_nan    > 0u) atomicAdd(nan_count,       s_nan);
        if (s_inf    > 0u) atomicAdd(inf_count,       s_inf);
        if (s_finite > 0u) {
            atomicMin(min_bits, s_min_bits);
            atomicMax(max_bits, s_max_bits);
        }
        // Normalize per-wg sums by 1/divisor before flushing so global fits.
        float sum_f   = float(s_sum)   * u_norm_n;
        float sumsq_f = float(s_sumsq) * u_norm_n;
        int  sum_q    = int(floor(sum_f   + sign(sum_f) * 0.5));
        uint sumsq_q  = uint(sumsq_f + 0.5);
        if (sum_q   != 0)  atomicAdd(sum_fp,   uint(sum_q));
        if (sumsq_q > 0u)  atomicAdd(sumsq_fp, sumsq_q);
        if (s_alive    > 0u) atomicAdd(alive_count,    s_alive);
        if (s_surface  > 0u) atomicAdd(surface_count,  s_surface);
        if (s_activity > 0u) atomicAdd(activity_count, s_activity);
    }
}
"""

# Pre-pass for deviation mode: just sum the channel to compute the mean.
# Same flush-divisor pattern, but we only need finite_count + sum_fp, so we
# reuse the main shader's SSBO layout (other slots stay zeroed).


def _gpu_metrics_compile(ctx, glsl_fmt='rgba32f'):
    """One-time compile per (context, format), cached."""
    cache = _shader_cache
    key = (id(ctx), 'harness_metrics_v1', glsl_fmt)
    if key not in cache:
        src = HARNESS_METRICS_SHADER
        if glsl_fmt != 'rgba32f':
            src = src.replace('rgba32f', glsl_fmt)
        cache[key] = ctx.compute_shader(src)
    return cache[key]


def _pool_key(ctx, size, dtype):
    return (id(ctx), size, dtype)


def _acquire_textures(ctx, size, tex_dtype, tex_bpt, init_data_bytes):
    """Get a texture pair from the pool or create new ones."""
    key = _pool_key(ctx, size, tex_dtype)
    pool = _texture_pool.get(key, [])
    if pool:
        tex_a, tex_b = pool.pop()
        _texture_pool[key] = pool
        tex_a.write(init_data_bytes)
        tex_b.write(bytes(size ** 3 * tex_bpt))
        return tex_a, tex_b
    tex_a = ctx.texture3d((size, size, size), 4, init_data_bytes, dtype=tex_dtype)
    tex_b = ctx.texture3d((size, size, size), 4, bytes(size ** 3 * tex_bpt),
                          dtype=tex_dtype)
    return tex_a, tex_b


def _return_textures(ctx, size, tex_dtype, tex_a, tex_b):
    """Return a texture pair to the pool for reuse."""
    key = _pool_key(ctx, size, tex_dtype)
    pool = _texture_pool.setdefault(key, [])
    # Keep pool bounded to avoid unbounded VRAM growth
    if len(pool) < 4:
        pool.append((tex_a, tex_b))
    else:
        tex_a.release()
        tex_b.release()


def _acquire_element_ssbo(ctx):
    """Get or create the shared element SSBO."""
    key = id(ctx)
    if key not in _element_ssbo_cache:
        from element_data import ELEMENT_GPU_DATA
        _element_ssbo_cache[key] = ctx.buffer(data=ELEMENT_GPU_DATA.tobytes())
    return _element_ssbo_cache[key]


class HeadlessRunner:
    """Run a CA rule on the GPU and collect metrics. No window, no rendering."""

    def __init__(self, ctx, rule_name, size=32, seed=42, params=None, dt=None,
                 init_density=None, init_override=None):
        from simulator import (
            RULE_PRESETS, COMPUTE_HEADER, CA_RULES,
            ELEMENT_COMPUTE_HEADER, ELEMENT_CA_RULE, INIT_FUNCS,
            init_random_sparse, _tex_format_for_size
        )

        self.ctx = ctx
        self.size = size
        self.rule_name = rule_name
        self.preset = RULE_PRESETS[rule_name]
        self.is_element_ca = self.preset.get('is_element_ca', False)

        # Rule-type-aware measurement config
        shader = self.preset['shader']
        if self.is_element_ca:
            self.measure_channel = 0
            self.measure_mode = 'element'  # non-vacuum = alive
            self.alive_threshold = 0.5
            self.change_threshold = 0.5
        elif shader == 'reaction_diffusion_3d':
            self.measure_channel = 1  # V channel has the patterns
            self.measure_mode = 'continuous'
            self.alive_threshold = 0.01
            self.change_threshold = 0.001
        elif shader == 'wave_3d':
            self.measure_channel = 0
            self.measure_mode = 'wave'  # use abs(value)
            self.alive_threshold = 0.005
            self.change_threshold = 0.001
        elif shader in ('smoothlife_3d', 'lenia_3d'):
            self.measure_channel = 0
            self.measure_mode = 'continuous'
            self.alive_threshold = 0.01
            self.change_threshold = 0.001
        elif shader == 'predator_prey_3d':
            self.measure_channel = 0  # prey (u)
            self.measure_mode = 'continuous'
            self.alive_threshold = 0.02  # low: boom-bust cycles drop prey near 0
            self.change_threshold = 0.001
        elif shader == 'bz_3d':
            self.measure_channel = 0  # Re(A), range ~[-1.2, 1.2]
            self.measure_mode = 'wave'  # PDE: use abs(value), avoids blinker penalties
            self.alive_threshold = 0.3  # raised: abs(Re(A)) > 0.3 catches active wave regions
            self.change_threshold = 0.01
        elif shader == 'morphogen_3d':
            self.measure_channel = 0  # activator: Turing patterns show as spatial variation
            self.measure_mode = 'deviation'  # "alive" = cells deviating from spatial mean
            self.alive_threshold = 0.1  # captures Turing spots/stripes without over-counting
            self.change_threshold = 0.02  # PDE field: need meaningful structural change
        elif shader == 'kuramoto_3d':
            self.measure_channel = 0  # phase [0,1]
            self.measure_mode = 'phase_coherence'  # alive = domain boundary cells
            self.alive_threshold = 0.35  # boundary cells: high local phase incoherence
            self.change_threshold = 0.02  # continuous field
        elif shader == 'flocking_3d':
            self.measure_channel = 0  # density rho
            self.measure_mode = 'deviation'  # detect flock clusters vs voids
            self.alive_threshold = 0.15
            self.change_threshold = 0.005
        elif shader == 'cahn_hilliard':
            self.measure_channel = 0  # order parameter c ∈ [-1,1]
            self.measure_mode = 'deviation'  # interesting = deviation from mean (domain walls)
            self.alive_threshold = 0.3
            self.change_threshold = 0.01
        elif shader == 'erosion_3d':
            self.measure_channel = 0  # solid density
            self.measure_mode = 'deviation'  # detect carved vs uncarved terrain
            self.alive_threshold = 0.15
            self.change_threshold = 0.005
        elif shader == 'mycelium_3d':
            self.measure_channel = 0  # biomass
            self.measure_mode = 'continuous'
            self.alive_threshold = 0.1
            self.change_threshold = 0.01
        elif shader == 'em_wave_3d':
            self.measure_channel = 0  # Ez field
            self.measure_mode = 'wave'
            self.alive_threshold = 0.01
            self.change_threshold = 0.001
        elif shader == 'viscous_fingers_3d':
            self.measure_channel = 0  # saturation
            self.measure_mode = 'deviation'  # detect finger fronts vs uniformly saturated
            self.alive_threshold = 0.30  # raised: field fills grid, need larger deviation
            self.change_threshold = 0.005
        elif shader == 'fire_3d':
            self.measure_channel = 1  # temperature
            self.measure_mode = 'deviation'  # detect active fire front vs burned/unburned
            self.alive_threshold = 0.25  # raised: fire fills grid, need bigger deviation
            self.change_threshold = 0.01
        elif shader == 'physarum_3d':
            self.measure_channel = 0  # trail
            self.measure_mode = 'deviation'  # trail fills everywhere; detect network structure
            self.alive_threshold = 0.2
            self.change_threshold = 0.01
        elif shader == 'fracture_3d':
            self.measure_channel = 2  # integrity
            self.measure_mode = 'deviation'  # detect crack network (broken vs intact)
            self.alive_threshold = 0.25  # raised: integrity fills grid after fracture
            self.change_threshold = 0.01
        elif shader == 'galaxy_3d':
            self.measure_channel = 0  # density
            self.measure_mode = 'deviation'  # detect density contrast (cosmic web filaments vs voids)
            self.alive_threshold = 0.005  # very low: density field has tiny deviations (~0.01-0.05)
            self.change_threshold = 0.002
        elif shader == 'lichen_3d':
            self.measure_channel = 0  # species A
            self.measure_mode = 'deviation'  # detect territorial patches (A-dominant zones)
            self.alive_threshold = 0.30  # raised: A fills grid, need large deviation for territories
            self.change_threshold = 0.01
        elif shader == 'crystal_growth':
            self.measure_channel = 0  # solid concentration
            self.measure_mode = 'continuous'  # crystal grows from 0→1; 0.3 catches growth front
            self.alive_threshold = 0.3
            self.change_threshold = 0.01
        elif shader in ('schrodinger_3d', 'schrodinger_poisson_3d', 'schrodinger_molecule_3d'):
            self.measure_channel = 3  # |Ψ|² probability density in A channel
            self.measure_mode = 'wave'  # PDE: use value directly (already ≥0)
            self.alive_threshold = 0.0001  # very sensitive: probability is normalized, sparse
            self.change_threshold = 0.00005
        else:
            # Discrete CAs (game_of_life_3d)
            self.measure_channel = 0
            self.measure_mode = 'discrete'
            self.alive_threshold = 0.5
            self.change_threshold = 0.01

        # Allow param/dt override
        self.params = dict(self.preset['params'])
        if params:
            for k, v in params.items():
                if k in self.params:
                    self.params[k] = v
        self.dt = dt if dt is not None else self.preset['dt']

        # Compile compute shader (with caching)
        shader_key = self.preset['shader']
        self._tex_dtype, self._tex_np_dtype, self._tex_bpt, self._tex_glsl_fmt = \
            _tex_format_for_size(size)
        if shader_key == 'element_ca':
            source = ELEMENT_COMPUTE_HEADER + ELEMENT_CA_RULE
        else:
            source = COMPUTE_HEADER + CA_RULES[shader_key]
        if self._tex_glsl_fmt != 'rgba32f':
            source = source.replace('rgba32f', self._tex_glsl_fmt)

        cache_key = (id(ctx), hash(source))
        if cache_key in _shader_cache:
            self.compute_prog = _shader_cache[cache_key]
        else:
            self.compute_prog = ctx.compute_shader(source)
            _shader_cache[cache_key] = self.compute_prog

        # Cache uniform handles: `'u_foo' in prog` hashes the source every
        # step otherwise (tens of thousands of string hashes per search run).
        prog = self.compute_prog
        self._u_size     = prog.get('u_size', None)
        self._u_dt       = prog.get('u_dt', None)
        self._u_boundary = prog.get('u_boundary', None)
        self._u_frame    = prog.get('u_frame', None)
        self._u_params   = [prog.get(f'u_param{i}', None) for i in range(4)]

        # Init volume
        rng = np.random.RandomState(seed)
        init_name = init_override if init_override else self.preset['init']
        init_func = INIT_FUNCS.get(init_name, init_random_sparse)
        # If init_density is specified and this is a discrete CA, override init
        # with a random field at the requested density
        if init_density is not None and not self.is_element_ca and \
           self.preset['init'] in ('random_very_sparse', 'random_sparse', 'random_dense'):
            data = np.zeros((size, size, size, 4), dtype=np.float32)
            data[:, :, :, 0] = (rng.random((size, size, size)) < init_density).astype(np.float32)
        elif init_density is not None and not self.is_element_ca and \
             self.preset['init'] == 'random_smooth':
            # Smooth init with variable amplitude
            field = rng.random((size, size, size)).astype(np.float32) * init_density * 2.0
            for _ in range(3):
                padded = np.pad(field, 1, mode='wrap')
                field = (padded[:-2, 1:-1, 1:-1] + padded[2:, 1:-1, 1:-1] +
                         padded[1:-1, :-2, 1:-1] + padded[1:-1, 2:, 1:-1] +
                         padded[1:-1, 1:-1, :-2] + padded[1:-1, 1:-1, 2:] +
                         padded[1:-1, 1:-1, 1:-1]) / 7.0
            data = np.zeros((size, size, size, 4), dtype=np.float32)
            data[:, :, :, 0] = field.astype(np.float32)
        elif init_density is not None and self.preset['init'] == 'lenia_blobs':
            # Use density to scale blob amplitudes
            data = init_func(size, rng)
            data[:, :, :, 0] *= init_density * 2.0
            data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0.0, 1.0)
        elif init_density is not None and not self.is_element_ca:
            # Generic density scaling: generate standard init, then scale
            # relevant channels by the density factor
            data = init_func(size, rng)
            init_name = self.preset['init']
            if init_name == 'gray_scott':
                # Scale V coverage: threshold the V channel at different levels
                v = data[:, :, :, 1]
                v_thresh = np.percentile(v[v > 0.01], max(0, 100 - init_density * 300)) if (v > 0.01).any() else 0
                data[:, :, :, 1] = np.where(v > v_thresh, v, v * 0.01)
                data[:, :, :, 0] = np.where(v > v_thresh, 0.5, 1.0)
            elif init_name in ('flocking', 'physarum', 'fire', 'fracture'):
                # Scale primary field density
                data[:, :, :, 0] *= init_density
                data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0.0, 1.0)
                if init_name == 'flocking':
                    # Scale velocity with density
                    for ch in range(1, 4):
                        data[:, :, :, ch] *= init_density
                elif init_name == 'fracture':
                    # Scale stress intensity
                    data[:, :, :, 1] *= init_density
            elif init_name == 'phase_separation':
                # Scale perturbation amplitude
                data[:, :, :, 0] *= init_density / 0.05  # normalize to default amplitude
                data[:, :, :, 0] = np.clip(data[:, :, :, 0], -1.0, 1.0)
            elif init_name == 'viscous_fingers':
                # Scale injection zone radius (re-generate with different size)
                data[:, :, :, 0] *= init_density
                data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0.0, 1.0)
                data[:, :, :, 1] *= init_density
            elif init_name == 'bz_reaction':
                # Scale amplitude around limit cycle
                data[:, :, :, 0] *= init_density
                data[:, :, :, 1] *= init_density
            elif init_name == 'morphogen':
                # Scale activator perturbation
                mean_a = 0.2  # baseline
                data[:, :, :, 0] = mean_a + (data[:, :, :, 0] - mean_a) * init_density
                data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0.01, 5.0)
            elif init_name == 'galaxy':
                # Scale density field (default base ~0.1)
                data[:, :, :, 0] *= init_density / 0.1
                data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0.001, None)
            elif init_name == 'lichen':
                # Scale seed density and resource level
                data[:, :, :, 0] *= init_density  # species A seeds
                data[:, :, :, 1] *= init_density  # species B seeds
                data[:, :, :, 3] *= init_density  # species C seeds
                data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0.0, 1.0)
                data[:, :, :, 1] = np.clip(data[:, :, :, 1], 0.0, 1.0)
                data[:, :, :, 3] = np.clip(data[:, :, :, 3], 0.0, 1.0)
        else:
            data = init_func(size, rng)

        init_bytes = data.astype(self._tex_np_dtype).tobytes()
        self.tex_a, self.tex_b = _acquire_textures(
            ctx, size, self._tex_dtype, self._tex_bpt, init_bytes)
        self.ping = 0

        # Element SSBO (shared across trials)
        self.element_ssbo = _acquire_element_ssbo(ctx)

    def step(self):
        src = self.tex_a if self.ping == 0 else self.tex_b
        dst = self.tex_b if self.ping == 0 else self.tex_a

        src.bind_to_image(0, read=True, write=False)
        dst.bind_to_image(1, read=False, write=True)

        if self.is_element_ca:
            self.element_ssbo.bind_to_storage_buffer(2)

        # Use cached uniform handles: avoids the per-step hash lookup against
        # the shader's uniform table (significant in tight search loops).
        if self._u_size is not None:
            self._u_size.value = self.size
        if self._u_dt is not None:
            self._u_dt.value = self.dt
        if self._u_boundary is not None:
            # Match the simulator's mapping: toroidal=0, clamped=1, mirror=2.
            b = self.preset.get('boundary', 'toroidal')
            self._u_boundary.value = {'clamped': 1, 'mirror': 2,
                                       'neumann': 2, 'reflect': 2,
                                       'zero_flux': 2}.get(b, 0)
        if self._u_frame is not None:
            if not hasattr(self, '_frame'):
                self._frame = 0
            self._u_frame.value = self._frame
            self._frame += 1

        param_values = list(self.params.values())
        for i, cu in enumerate(self._u_params):
            if cu is not None:
                cu.value = float(param_values[i]) if i < len(param_values) else 0.0

        groups = (self.size + 7) // 8
        self.compute_prog.run(groups, groups, groups)
        try:
            GL.glMemoryBarrier(GL.GL_ALL_BARRIER_BITS)
        except Exception:
            self.ctx.memory_barrier()

        self.ping = 1 - self.ping

    def read_grid(self):
        src = self.tex_a if self.ping == 0 else self.tex_b
        raw = np.frombuffer(src.read(), dtype=self._tex_np_dtype).reshape(
            self.size, self.size, self.size, 4)
        # Only convert if not already float32 (size > ~400 uses float16)
        if self._tex_np_dtype == np.float32:
            return raw.copy()
        return raw.astype(np.float32)

    # ── GPU-side metrics ─────────────────────────────────────────────
    # Lazy-allocates prev-grid texture + stats SSBO + program on first call.
    # All resources released in release().

    def _ensure_metrics_resources(self):
        if getattr(self, '_metrics_prog', None) is not None:
            return
        self._metrics_prog = _gpu_metrics_compile(self.ctx, self._tex_glsl_fmt)
        # Cache uniform handles
        p = self._metrics_prog
        self._mu = {
            'size':       p.get('u_size', None),
            'channel':    p.get('u_channel', None),
            'mode':       p.get('u_mode', None),
            'alive_thr':  p.get('u_alive_thr', None),
            'change_thr': p.get('u_change_thr', None),
            'have_prev':  p.get('u_have_prev', None),
            'is_element': p.get('u_is_element', None),
            'field_mean': p.get('u_field_mean', None),
            'norm_n':     p.get('u_norm_n', None),
        }
        # 16 uint32s
        self._metrics_ssbo = self.ctx.buffer(reserve=_METRICS_NUINTS * 4)
        # Prev-grid texture (for activity); created lazily on first sample
        self._prev_tex = None
        self._has_prev = False
        # Per-wg flush divisor; set per dispatch.
        self._metrics_divisor = 1.0

    def _alloc_prev_tex(self):
        if self._prev_tex is not None:
            return
        # Match main grid format
        sz = self.size
        self._prev_tex = self.ctx.texture3d((sz, sz, sz), 4, dtype=self._tex_dtype)

    def _metrics_dispatch(self, src_tex, mode_id, field_mean=0.0,
                           with_prev=True):
        """Dispatch the metrics shader against src_tex. Returns nothing;
        result lives in _metrics_ssbo."""
        prog = self._metrics_prog
        ssbo = self._metrics_ssbo

        # Zero SSBO and pre-seed min sentinel (bits of BIAS+1e30).
        zeros = bytearray(_METRICS_NUINTS * 4)
        sentinel = np.float32(_METRICS_BIAS + 1.0e30).tobytes()
        zeros[12:16] = sentinel  # min_bits is uint index 3
        ssbo.write(bytes(zeros))

        src_tex.bind_to_image(0, read=True, write=False)
        if with_prev and self._has_prev:
            self._prev_tex.bind_to_image(1, read=True, write=False)
            have_prev = 1
        else:
            # Bind something to slot 1 anyway (driver requirement)
            src_tex.bind_to_image(1, read=True, write=False)
            have_prev = 0
        ssbo.bind_to_storage_buffer(2)

        N = float(self.size ** 3)
        divisor = max(1.0, N / 2048.0)
        self._metrics_divisor = divisor

        mu = self._mu
        if mu['size']       is not None: mu['size'].value       = self.size
        if mu['channel']    is not None: mu['channel'].value    = self.measure_channel
        if mu['mode']       is not None: mu['mode'].value       = mode_id
        if mu['alive_thr']  is not None: mu['alive_thr'].value  = float(self.alive_threshold)
        if mu['change_thr'] is not None: mu['change_thr'].value = float(self.change_threshold)
        if mu['have_prev']  is not None: mu['have_prev'].value  = have_prev
        if mu['is_element'] is not None: mu['is_element'].value = 1 if self.is_element_ca else 0
        if mu['field_mean'] is not None: mu['field_mean'].value = float(field_mean)
        if mu['norm_n']     is not None: mu['norm_n'].value     = 1.0 / divisor

        gx = (self.size + 7) // 8
        prog.run(gx, gx, gx)
        if GL is not None:
            try:
                GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT |
                                   GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
            except Exception:
                self.ctx.memory_barrier()
        else:
            self.ctx.memory_barrier()

    def _metrics_decode(self):
        """Read the SSBO and decode into a metrics dict matching compute_metrics()."""
        raw = self._metrics_ssbo.read()
        u = np.frombuffer(raw, dtype=np.uint32)
        finite = int(u[0])
        nans   = int(u[1])
        infs   = int(u[2])
        min_bits = u[3]
        max_bits = u[4]
        sum_signed = np.int32(u[5])  # bit reinterpret to signed
        sumsq_u    = u[6]
        alive_count    = int(u[7])
        surface_count  = int(u[8])
        activity_count = int(u[9])

        N = float(self.size ** 3)
        divisor = self._metrics_divisor
        if finite > 0:
            v_mean = float(sum_signed) * divisor / (N * _METRICS_SCALE_S)
            e_x2   = float(sumsq_u)    * divisor / (N * _METRICS_SCALE_Q)
            v_std  = float(np.sqrt(max(e_x2 - v_mean * v_mean, 0.0)))
            mins_f = np.frombuffer(np.uint32(min_bits).tobytes(), dtype=np.float32)[0]
            maxs_f = np.frombuffer(np.uint32(max_bits).tobytes(), dtype=np.float32)[0]
            v_min = float(mins_f - _METRICS_BIAS) if min_bits != 0 and not (mins_f - _METRICS_BIAS) > 1e29 else 0.0
            v_max = float(maxs_f - _METRICS_BIAS) if max_bits != 0 else 0.0
        else:
            v_mean = v_std = v_min = v_max = 0.0

        return {
            'finite': finite, 'nans': nans, 'infs': infs,
            'mean': v_mean, 'std': v_std, 'min': v_min, 'max': v_max,
            'sum_raw': float(sum_signed) * divisor / _METRICS_SCALE_S,
            'alive_count': alive_count,
            'surface_count': surface_count,
            'activity_count': activity_count,
        }

    def compute_metrics_gpu(self, prev_is_valid):
        """Replacement for CPU compute_metrics(). Returns the same dict shape.

        prev_is_valid: True iff a prev-grid texture exists with last sample's
        data (i.e. not the first sample of a trial).
        """
        self._ensure_metrics_resources()
        src = self.tex_a if self.ping == 0 else self.tex_b
        size = self.size
        total = size ** 3
        mode = self.measure_mode
        mode_id = _MODE_ID.get(mode, 0)

        # Mark whether the prev texture is meaningful for this dispatch.
        self._has_prev = prev_is_valid and self._prev_tex is not None

        if mode == 'deviation':
            # Pass 1: compute mean by running shape with mode_id=0 but we
            # only consume sum_fp. The thresholding is wrong for this pass
            # (alive/surface/activity will be garbage) — but we ignore them
            # and re-dispatch.
            # To avoid the wasted alive/surface work we set mode to a cheap
            # 'continuous' (mode 0) with an impossibly high threshold so
            # alive_count comes out 0 and surface_count is 0 — only the
            # scalar stats are useful.
            saved_thr = self.alive_threshold
            self.alive_threshold = 1.0e30  # nothing alive
            self._metrics_dispatch(src, mode_id=0, with_prev=False)
            self.alive_threshold = saved_thr
            partial = self._metrics_decode()
            field_mean = partial['mean']
            # Pass 2: real deviation thresholding + activity
            self._metrics_dispatch(src, mode_id=3, field_mean=field_mean,
                                    with_prev=True)
            stats = self._metrics_decode()
            # Use mean/std/min/max from the pass-1 stats (the second pass's
            # are identical -- same channel data -- but pass-1 already paid
            # for them and pass 2 sees same inputs, so either works).
            stats['mean'] = field_mean
            stats['std']  = partial['std']
            stats['min']  = partial['min']
            stats['max']  = partial['max']
            stats['finite'] = partial['finite']
            stats['nans']   = partial['nans']
            stats['infs']   = partial['infs']
        else:
            self._metrics_dispatch(src, mode_id=mode_id, with_prev=True)
            stats = self._metrics_decode()

        # Copy current src into prev_tex for next sample's activity calc.
        # glCopyImageSubData is the right primitive (server-side, no readback).
        self._alloc_prev_tex()
        copied = False
        if GL is not None:
            try:
                GL.glCopyImageSubData(
                    src.glo, GL.GL_TEXTURE_3D, 0, 0, 0, 0,
                    self._prev_tex.glo, GL.GL_TEXTURE_3D, 0, 0, 0, 0,
                    size, size, size,
                )
                copied = True
            except Exception:
                copied = False
        if not copied:
            # Fallback: read & write via host. Slow but safe.
            self._prev_tex.write(src.read())
        self._has_prev = True

        # Build output dict matching compute_metrics() shape.
        alive_count = stats['alive_count']
        alive_ratio = alive_count / total
        activity = stats['activity_count'] / total if prev_is_valid else 0.0
        surface_ratio = stats['surface_count'] / max(alive_count, 1)
        has_nan = stats['nans'] > 0
        has_inf = stats['infs'] > 0

        return {
            'alive_count': alive_count,
            'alive_ratio': alive_ratio,
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max'],
            'activity': activity,
            'surface_ratio': surface_ratio,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'measure_mode': mode,
        }

    def release(self):
        # Release prev tex + SSBO to the OpenGL driver. (No pool for these:
        # prev-tex is per-runner only and ssbo is tiny.)
        pt = getattr(self, '_prev_tex', None)
        if pt is not None:
            try: pt.release()
            except Exception: pass
            self._prev_tex = None
        ssbo = getattr(self, '_metrics_ssbo', None)
        if ssbo is not None:
            try: ssbo.release()
            except Exception: pass
            self._metrics_ssbo = None
        _return_textures(self.ctx, self.size, self._tex_dtype,
                         self.tex_a, self.tex_b)


# ── Metrics ───────────────────────────────────────────────────────────

def compute_metrics(grid, prev_grid, runner):
    """Compute behavioral metrics from a grid snapshot using rule-aware config."""
    size = grid.shape[0]
    total = size ** 3

    ch = grid[:, :, :, runner.measure_channel]
    mode = runner.measure_mode

    if mode == 'element':
        alive_mask = np.abs(grid[:, :, :, 0]) > 0.5  # non-vacuum
        alive_mask &= np.abs(grid[:, :, :, 0] - 119.0) > 0.5  # exclude wall
    elif mode == 'wave':
        alive_mask = np.abs(ch) > runner.alive_threshold
    elif mode == 'deviation':
        # For field CAs: "alive" = deviating from spatial mean
        field_mean = ch.mean()
        alive_mask = np.abs(ch - field_mean) > runner.alive_threshold
    elif mode == 'phase_coherence':
        # Kuramoto: measure local phase incoherence (domain boundaries)
        # Phase wraps [0,1] = [0, 2π], so use sin of phase difference
        phase_2pi = ch * (2 * np.pi)
        padded = np.pad(phase_2pi, 1, mode='wrap')
        # Mean |sin(neighbor_phase - my_phase)| over 6 von Neumann neighbors
        incoherence = np.zeros_like(ch)
        shifts = [
            (slice(2, None), slice(1, -1), slice(1, -1)),   # +x
            (slice(None, -2), slice(1, -1), slice(1, -1)),  # -x
            (slice(1, -1), slice(2, None), slice(1, -1)),   # +y
            (slice(1, -1), slice(None, -2), slice(1, -1)),  # -y
            (slice(1, -1), slice(1, -1), slice(2, None)),   # +z
            (slice(1, -1), slice(1, -1), slice(None, -2)),  # -z
        ]
        for s in shifts:
            nb_phase = padded[s]
            incoherence += np.abs(np.sin(nb_phase - phase_2pi))
        incoherence /= 6.0  # range [0, 1]: 0=synchronized, 1=anti-phase
        # "alive" = cells at domain boundaries (high incoherence)
        alive_mask = incoherence > runner.alive_threshold
    else:
        alive_mask = ch > runner.alive_threshold

    alive_count = int(alive_mask.sum())
    alive_ratio = alive_count / total

    # Value stats on measured channel (guard against all-NaN / empty)
    has_nan = bool(np.isnan(ch).any())
    has_inf = bool(np.isinf(ch).any())

    val_for_stats = np.abs(ch) if mode == 'wave' else ch
    finite = val_for_stats[np.isfinite(val_for_stats)]
    if finite.size > 0:
        v_mean = float(np.mean(finite))
        v_std = float(np.std(finite))
        v_min = float(np.min(finite))
        v_max = float(np.max(finite))
    else:
        v_mean = v_std = v_min = v_max = 0.0

    # Activity: cells that changed since last step
    activity = 0.0
    if prev_grid is not None:
        prev_ch = prev_grid[:, :, :, runner.measure_channel]
        changed = np.abs(ch - prev_ch) > runner.change_threshold
        if mode == 'element':
            # Also count temperature changes as activity
            temp_changed = np.abs(grid[:, :, :, 1] - prev_grid[:, :, :, 1]) > 1.0
            changed = changed | temp_changed
        activity = changed.sum() / total

    # Surface ratio: surface cells / alive cells (0 = no structure, high = complex)
    surface_count = 0
    if alive_count > 0:
        # Pad to handle boundaries (toroidal)
        padded = np.pad(alive_mask.astype(np.float32), 1, mode='wrap')
        neighbor_sum = (
            padded[:-2, 1:-1, 1:-1] + padded[2:, 1:-1, 1:-1] +
            padded[1:-1, :-2, 1:-1] + padded[1:-1, 2:, 1:-1] +
            padded[1:-1, 1:-1, :-2] + padded[1:-1, 1:-1, 2:]
        )
        # Surface = alive AND at least one dead neighbor
        surface = alive_mask & (neighbor_sum < 6)
        surface_count = int(surface.sum())

    surface_ratio = surface_count / max(alive_count, 1)

    return {
        'alive_count': alive_count,
        'alive_ratio': alive_ratio,
        'mean': v_mean,
        'std': v_std,
        'min': v_min,
        'max': v_max,
        'activity': activity,
        'surface_ratio': surface_ratio,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'measure_mode': mode,
    }


def score_interestingness(metric_history):
    """Score 0-1 from a list of metric dicts over a run. Higher = more interesting.

    Uses smooth/continuous scoring to avoid creating hard attractor basins where
    all discoveries cluster into the same few modes.

    Activity is measured as the fraction of cells that changed between consecutive
    steps (1-step comparison via ping-pong textures).  This means:
      - Truly frozen pattern: activity ≈ 0
      - Slowly evolving PDE:  activity ≈ 0.001-0.05
      - Period-2 blinker:     activity ≈ alive_ratio (50%+ of alive cells toggle)

    Scoring uses median alive (not just final) so transient processes like fire/fracture
    that have interesting dynamics mid-run but boring final states still score well.
    """
    if not metric_history:
        return 0.0

    # Check for NaN/Inf at any point
    if any(m['has_nan'] or m['has_inf'] for m in metric_history):
        return 0.0

    alive_ratios = [m['alive_ratio'] for m in metric_history]
    activities = [m['activity'] for m in metric_history]
    surface_ratios = [m['surface_ratio'] for m in metric_history]

    final_alive = alive_ratios[-1]
    median_alive = float(np.median(alive_ratios))
    # Use the better of median vs final — transient rules benefit from median,
    # steady-state rules have median ≈ final so it doesn't matter
    repr_alive = max(median_alive, final_alive)
    mean_activity = np.mean(activities[1:]) if len(activities) > 1 else 0.0
    peak_activity = float(np.max(activities[1:])) if len(activities) > 1 else 0.0
    mean_surface = np.mean(surface_ratios)
    peak_surface = float(np.max(surface_ratios))
    alive_std = np.std(alive_ratios)

    mode = metric_history[0].get('measure_mode', 'discrete')
    continuous_field = mode in ('deviation', 'phase_coherence', 'continuous', 'wave')

    # Dead penalty — soft: if both final AND median are dead, it's truly dead
    if repr_alive < 0.001:
        return 0.0

    # Saturated penalty — smooth ramp starting at 0.70 instead of a cliff at
    # 0.95.  Filling 80% of the cube is an undesirable equilibrium even with
    # activity (it means the rule expands without bound), so we taper credit
    # quickly through the 0.70-0.95 band and floor it after that.
    if repr_alive > 0.95:
        if mean_activity < 0.001:
            return 0.02
        return max(0.05, 0.10 * min(mean_activity * 10, 1.0))
    saturation_factor = 1.0
    if repr_alive > 0.70:
        # Linear ramp: 1.0 at 0.70, 0.10 at 0.95
        saturation_factor = max(0.10, 1.0 - (repr_alive - 0.70) / 0.25 * 0.90)

    # Frozen penalty (no activity after initial transient) — softer for transient events
    late_activity = np.mean(activities[-5:]) if len(activities) >= 5 else mean_activity
    if late_activity < 0.0001 and mean_activity < 0.001:
        # If there was a significant event at any point, give partial credit
        if peak_activity > 0.05:
            return max(0.15, 0.35 * min(peak_activity * 3, 1.0))
        return 0.1
    # If only late activity is dead but mid-run had significant activity, partial credit
    if late_activity < 0.0001:
        return max(0.15, 0.35 * min(peak_activity * 3, 1.0))

    score = 0.0

    # Alive ratio: smooth bell curve using repr_alive (best of median/final).
    # Centers/widths tightened so an 80%-filled cube no longer scores ~0.85;
    # the prior values let cube-fillers dominate the search.
    alive_center = 0.20 if continuous_field else 0.10
    alive_width = 1.1 if continuous_field else 0.9
    alive_score = np.exp(-0.5 * ((np.log(max(repr_alive, 0.005)) - np.log(alive_center)) / alive_width)**2)
    score += 0.25 * alive_score

    # Activity: log-scale bell curve
    # PDE peak at -0.7 (~0.2) — high activity is normal for continuous fields
    # Discrete peak at -2.0 (0.01) — high activity means blinker
    best_activity = max(late_activity, mean_activity)
    if best_activity > 0.0001:
        act_log = np.log10(best_activity)
        act_peak = -0.7 if continuous_field else -2.0
        act_width = 1.8 if continuous_field else 1.5
        act_score = np.exp(-0.5 * ((act_log - act_peak) / act_width)**2)
        score += 0.25 * act_score
    else:
        score += 0.02

    # Blinker penalty: high activity relative to alive count = mass oscillation
    # Only for discrete CAs — PDEs naturally have activity ≈ alive
    if not continuous_field and repr_alive > 0.10 and late_activity > 0.02:
        act_alive_ratio = late_activity / max(repr_alive, 0.01)
        if act_alive_ratio > 1.0:
            score -= 0.25        # global blinker — most cells toggle every step
        elif act_alive_ratio > 0.6:
            score -= 0.15        # substantial blinker
        elif act_alive_ratio > 0.4:
            score -= 0.05        # mild oscillation

    # Alive count wildly oscillating — softer penalty for PDEs (natural oscillation)
    if alive_std > 0.1:
        penalty = 0.10 if not continuous_field else 0.05
        score -= penalty

    # Stability: smooth — less oscillation = better, no hard steps
    stability_decay = 10.0 if not continuous_field else 5.0
    stability_score = np.exp(-stability_decay * alive_std)
    score += 0.15 * stability_score

    # Surface complexity: use best of mean and peak — transient rules often have
    # high surface complexity during the event but low at steady state
    best_surface = max(mean_surface, peak_surface * 0.7)  # peak discounted slightly
    score += 0.20 * min(best_surface, 1.0)

    # Bonus: gradual change over time (evolving, not static)
    if len(alive_ratios) >= 10:
        early = np.mean(alive_ratios[:5])
        late = np.mean(alive_ratios[-5:])
        drift = abs(early - late)
        if 0.01 < drift < 0.3:
            score += 0.15        # evolving (widened from 0.2 to 0.3 for transient rules)
        elif drift < 0.01 and best_activity > 0.0001:
            score += 0.1         # stable but active

    # Apply saturation taper from the 0.70-0.95 band (computed above).
    score *= saturation_factor

    return max(min(score, 1.0), 0.0)


# ── Structural analysis (slice / projection metrics) ──────────────────

def _binary_slice(grid_3d, axis, index, threshold=0.5):
    """Extract a 2D binary slice from a 3D grid along the given axis."""
    if axis == 0:
        s = grid_3d[index, :, :]
    elif axis == 1:
        s = grid_3d[:, index, :]
    else:
        s = grid_3d[:, :, index]
    return (s > threshold).astype(np.float32)


def _gol_2d_step(grid_2d):
    """One step of 2D Game of Life on a binary grid. Toroidal boundary."""
    padded = np.pad(grid_2d, 1, mode='wrap')
    neighbors = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2] +                     padded[1:-1, 2:] +
        padded[2:, :-2]  + padded[2:, 1:-1]  + padded[2:, 2:]
    )
    birth = (grid_2d == 0) & (neighbors == 3)
    survive = (grid_2d == 1) & ((neighbors == 2) | (neighbors == 3))
    return (birth | survive).astype(np.float32)


def slice_gol_coherence(grid, channel=0, axis=2, threshold=0.5):
    """Measure how well consecutive Z-slices follow 2D GoL rules.

    Returns a value in [0, 1] where 1.0 means each slice is the exact
    GoL successor of the previous slice. This detects whether the 3D
    structure embeds a 2D GoL spacetime.
    """
    vol = grid[:, :, :, channel]
    size = vol.shape[axis]
    if size < 2:
        return 0.0

    # Binarize once and move the stacking axis to front so b[i] is the
    # i-th slice. Vectorized GoL step across ALL slices in a single pass
    # (replaces the 31-iteration Python loop + scipy.pad-per-slice that
    # used to dominate analyze_structure).
    b = (vol > threshold).astype(np.int8)
    b = np.moveaxis(b, axis, 0)  # (size, H, W)
    # 8-neighbor sum with toroidal wrap via np.roll (no pad allocation).
    rp1 = np.roll(b, 1, axis=1); rm1 = np.roll(b, -1, axis=1)
    cp1 = np.roll(b, 1, axis=2); cm1 = np.roll(b, -1, axis=2)
    n = (rp1 + rm1 + cp1 + cm1 +
         np.roll(rp1, 1, axis=2) + np.roll(rp1, -1, axis=2) +
         np.roll(rm1, 1, axis=2) + np.roll(rm1, -1, axis=2))
    predicted = ((b == 0) & (n == 3)) | ((b == 1) & ((n == 2) | (n == 3)))
    # Compare predicted[i] to actual slice[i+1].
    pred_cur = predicted[:-1]
    actual_next = b[1:].astype(bool)
    # Skip trivial pairs (both dead or both saturated).
    alive_cur  = b[:-1].mean(axis=(1, 2))
    alive_next = b[1:].mean(axis=(1, 2))
    trivial = ((alive_cur < 0.01) & (alive_next < 0.01)) | \
              ((alive_cur > 0.99) & (alive_next > 0.99))
    agreement = (pred_cur == actual_next).mean(axis=(1, 2))
    valid = ~trivial
    return float(agreement[valid].mean()) if valid.any() else 0.0


def projection_entropy(grid, channel=0, threshold=0.01):
    """Compute Shannon entropy of max-projection along each axis.

    Returns dict with 'entropy_x', 'entropy_y', 'entropy_z' (each 0-1 normalized)
    and 'projection_complexity' (mean entropy).
    """
    vol = grid[:, :, :, channel]
    vol = np.abs(vol)  # handle wave-type CAs

    entropies = {}
    for ax, name in enumerate(['x', 'y', 'z']):
        proj = np.max(vol, axis=ax)
        # Normalize to [0, 1]
        pmax = proj.max()
        if pmax > threshold:
            proj = proj / pmax
        else:
            entropies[f'entropy_{name}'] = 0.0
            continue
        # Discretize into bins for entropy
        bins = 16
        hist, _ = np.histogram(proj.ravel(), bins=bins, range=(0, 1))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        ent = -np.sum(hist * np.log2(hist))
        # Normalize by max possible entropy
        entropies[f'entropy_{name}'] = float(ent / np.log2(bins))

    entropies['projection_complexity'] = float(np.mean([
        entropies['entropy_x'], entropies['entropy_y'], entropies['entropy_z']
    ]))
    return entropies


def projection_structure(grid, channel=0):
    """Measure spatial structure in projections (edge density, not just entropy).

    Returns dict with 'structure_x/y/z' and 'projection_structure' mean.
    Higher values = more internal spatial pattern (edges, boundaries).
    """
    vol = np.abs(grid[:, :, :, channel])

    structures = {}
    for ax, name in enumerate(['x', 'y', 'z']):
        proj = np.max(vol, axis=ax)
        pmax = proj.max()
        if pmax < 0.01:
            structures[f'structure_{name}'] = 0.0
            continue
        proj = proj / pmax
        # Sobel-like edge measure
        dx = np.abs(np.diff(proj, axis=0))
        dy = np.abs(np.diff(proj, axis=1))
        edge_density = (dx.mean() + dy.mean()) / 2.0
        structures[f'structure_{name}'] = float(edge_density)

    structures['projection_structure'] = float(np.mean([
        structures['structure_x'], structures['structure_y'], structures['structure_z']
    ]))
    return structures


def slice_mutual_info(grid, channel=0, axis=2, n_samples=8, threshold=0.5):
    """Measure mutual information between evenly-spaced slices along an axis.

    High MI = slices are related (the 3D structure has depth coherence).
    Low MI = slices are independent (random 3D noise).
    Returns value in [0, 1].
    """
    vol = grid[:, :, :, channel]
    size = vol.shape[axis]
    if size < n_samples:
        n_samples = size

    indices = np.linspace(0, size - 1, n_samples, dtype=int)
    slices = [_binary_slice(vol, axis, i, threshold).ravel() for i in indices]

    # Pairwise normalized MI
    mis = []
    bins = 2  # binary slices
    for i in range(len(slices)):
        for j in range(i + 1, len(slices)):
            # Joint histogram
            joint = slices[i] * bins + slices[j]
            hist = np.bincount(joint.astype(int), minlength=bins * bins).astype(float)
            hist /= hist.sum()
            hist = hist.reshape(bins, bins)

            # Marginals
            px = hist.sum(axis=1)
            py = hist.sum(axis=0)

            # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
            mi = 0.0
            for xi in range(bins):
                for yi in range(bins):
                    if hist[xi, yi] > 0 and px[xi] > 0 and py[yi] > 0:
                        mi += hist[xi, yi] * np.log2(hist[xi, yi] / (px[xi] * py[yi]))
            mis.append(mi)

    return float(np.mean(mis)) if mis else 0.0


def spatial_variation(grid, channel=0, n_blocks=8):
    """Measure spatial heterogeneity by dividing grid into blocks and comparing density.

    Returns a value in [0, 1] where 0 = spatially uniform (global oscillator)
    and 1 = highly heterogeneous (distinct spatial patterns).

    Divides the grid into n_blocks^3 sub-volumes, computes mean absolute value
    per block, then returns the coefficient of variation (std/mean) clamped to [0,1].
    """
    vol = np.abs(grid[:, :, :, channel])
    sz = vol.shape[0]
    bsz = max(1, sz // n_blocks)
    # When the grid divides evenly, do a single vectorized block-reduce
    # instead of the Python triple loop (common case; 8× faster).
    if sz == bsz * n_blocks:
        block_means = vol.reshape(n_blocks, bsz, n_blocks, bsz, n_blocks, bsz)\
                         .mean(axis=(1, 3, 5)).ravel()
    else:
        # Fallback for non-divisible sizes.
        block_means = []
        for ix in range(0, sz, bsz):
            for iy in range(0, sz, bsz):
                for iz in range(0, sz, bsz):
                    block = vol[ix:ix+bsz, iy:iy+bsz, iz:iz+bsz]
                    block_means.append(block.mean())
        block_means = np.array(block_means)
    mean_val = block_means.mean()
    if mean_val < 1e-6:
        return 0.0
    cv = float(block_means.std() / mean_val)
    return min(cv, 1.0)


def analyze_structure(grid, channel=0):
    """Run all structural analysis on a grid snapshot.

    Returns a dict with all slice/projection metrics.
    """
    result = {}
    result['gol_coherence_z'] = slice_gol_coherence(grid, channel, axis=2)
    result['gol_coherence_y'] = slice_gol_coherence(grid, channel, axis=1)
    result['gol_coherence_x'] = slice_gol_coherence(grid, channel, axis=0)
    result['gol_coherence_max'] = max(
        result['gol_coherence_z'], result['gol_coherence_y'], result['gol_coherence_x']
    )
    result.update(projection_entropy(grid, channel))
    result.update(projection_structure(grid, channel))
    result['slice_mi_z'] = slice_mutual_info(grid, channel, axis=2)
    result['slice_mi_y'] = slice_mutual_info(grid, channel, axis=1)
    result['slice_mi_x'] = slice_mutual_info(grid, channel, axis=0)
    result['slice_mi_max'] = max(
        result['slice_mi_z'], result['slice_mi_y'], result['slice_mi_x']
    )
    result['spatial_variation'] = spatial_variation(grid, channel)
    return result


# ── Advanced dynamics metrics (period, gliders, growth, symmetry) ─────

def _grid_hash(binary_grid):
    """Fast hash of a binary grid for period detection."""
    return hash(binary_grid.tobytes())


def detect_period(grid_snapshots, channel=0, threshold=0.5):
    """Detect exact periodicity in a sequence of grid snapshots.

    Returns dict with:
      period: cycle length (0 = no period found)
      period_start: step index where cycle starts
      period_score: 0-1, how clean the period is (1 = perfect cycle)
    """
    # Binarize snapshots and hash them
    hashes = []
    for g in grid_snapshots:
        binary = (g[:, :, :, channel] > threshold).astype(np.uint8)
        hashes.append(_grid_hash(binary))

    # Look for repeated hashes (period detection)
    n = len(hashes)
    best_period = 0
    best_start = 0
    best_confirmations = 0

    # Check periods from 1 to n//3 (need at least 3 repetitions to confirm)
    max_period = min(n // 3, 200)
    for p in range(1, max_period + 1):
        # Check from the end backwards for this period
        for start in range(n - 2 * p, max(0, n - 4 * p) - 1, -1):
            confirmations = 0
            valid = True
            for k in range(start, n - p, p):
                if hashes[k] == hashes[k + p]:
                    confirmations += 1
                else:
                    valid = False
                    break
            if valid and confirmations >= 2 and confirmations > best_confirmations:
                best_period = p
                best_start = start
                best_confirmations = confirmations

    period_score = 0.0
    if best_period > 0:
        # Score: short periods are more remarkable, more confirmations = more confident
        period_score = min(1.0, best_confirmations / 5.0) * min(1.0, 20.0 / best_period)

        # Devalue trivial period-2 global oscillation:
        # If most of the grid is alive and toggles every step, period=2 is boring
        if best_period <= 2 and len(grid_snapshots) >= 2:
            last = grid_snapshots[-1]
            alive_frac = (last[:, :, :, channel] > threshold).mean()
            if alive_frac > 0.2:
                # Check if activity is global (compare last two snapshots)
                prev = grid_snapshots[-2]
                changed = ((last[:, :, :, channel] > threshold) !=
                          (prev[:, :, :, channel] > threshold)).mean()
                if changed > 0.3:  # >30% of grid changes = global oscillation
                    period_score *= 0.1  # nearly zero for boring global blink

    return {
        'period': best_period,
        'period_start': best_start,
        'period_score': float(period_score),
    }


def detect_translation(grid_snapshots, channel=0, threshold=0.5):
    """Detect translating structures (gliders/spaceships) via FFT cross-correlation.

    Uses FFT-based phase correlation to find the dominant shift between
    time-separated snapshots, then scores consistency across frame pairs.

    Returns dict with:
      translation_score: 0-1 (1 = perfect glider-like translation)
      translation_speed: cells/step of detected translation
      translation_dir: (dx, dy, dz) unit direction
    """
    if len(grid_snapshots) < 10:
        return {'translation_score': 0.0, 'translation_speed': 0.0, 'translation_dir': (0, 0, 0)}

    # Use snapshots from the second half (after transient)
    half = len(grid_snapshots) // 2
    snaps = grid_snapshots[half:]
    size = snaps[0].shape[0]

    # Binarize
    bins = [(g[:, :, :, channel] > threshold).astype(np.float32) for g in snaps]

    # Don't bother if grid is mostly empty or mostly full
    alive = bins[-1].mean()
    if alive < 0.005 or alive > 0.5:
        return {'translation_score': 0.0, 'translation_speed': 0.0, 'translation_dir': (0, 0, 0)}

    # FFT-based phase correlation to find dominant shift
    best_score = 0.0
    best_shift = (0, 0, 0)
    best_dt = 1

    for dt_steps in [2, 4, 8]:
        if dt_steps >= len(bins):
            continue

        # Accumulate cross-correlation across frame pairs for this dt
        pairs = list(range(0, len(bins) - dt_steps, max(1, dt_steps)))
        if not pairs:
            continue

        # Average the cross-power spectrum across pairs for robustness
        accum = None
        for i in pairs:
            fa = np.fft.rfftn(bins[i])
            fb = np.fft.rfftn(bins[i + dt_steps])
            cross = fa * np.conj(fb)
            # Normalize to get phase correlation (avoid div by zero)
            mag = np.abs(cross)
            mag[mag < 1e-12] = 1e-12
            if accum is None:
                accum = cross / mag
            else:
                accum += cross / mag

        corr = np.fft.irfftn(accum, s=(size, size, size))

        # Zero out the DC (no-shift) peak — we want actual translation
        corr[0, 0, 0] = 0.0

        # Find peak within ±3 cells (matching original shift_range)
        peak_val = 0.0
        peak_shift = (0, 0, 0)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                for dz in range(-3, 4):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    val = corr[dx % size, dy % size, dz % size]
                    if val > peak_val:
                        peak_val = val
                        peak_shift = (dx, dy, dz)

        # Normalize: max possible is len(pairs) (perfect correlation at every pair)
        norm_score = peak_val / max(len(pairs), 1)

        # Verify with IoU on a few pairs to convert to a true overlap score
        if norm_score > 0.1:
            overlaps = []
            check_pairs = pairs[:min(4, len(pairs))]
            dx, dy, dz = peak_shift
            for i in check_pairs:
                a = bins[i]
                b = bins[i + dt_steps]
                b_shifted = np.roll(np.roll(np.roll(b, -dx, axis=0), -dy, axis=1), -dz, axis=2)
                union_sum = np.maximum(a, b_shifted).sum()
                if union_sum > 10:
                    overlaps.append(float((a * b_shifted).sum() / union_sum))
            if overlaps:
                iou_score = np.mean(overlaps)
                if iou_score > best_score:
                    best_score = iou_score
                    best_shift = peak_shift
                    best_dt = dt_steps

    speed = np.sqrt(best_shift[0]**2 + best_shift[1]**2 + best_shift[2]**2) / best_dt if best_dt > 0 else 0

    return {
        'translation_score': float(best_score),
        'translation_speed': float(speed),
        'translation_dir': best_shift,
    }


def detect_growth(metric_history):
    """Detect monotonic growth patterns (guns, replicators).

    Returns dict with:
      growth_score: 0-1 (1 = steady monotonic growth from sparse start)
      growth_rate: alive cells gained per step
      growth_type: 'none', 'linear', 'accelerating', 'decelerating'
    """
    if len(metric_history) < 10:
        return {'growth_score': 0.0, 'growth_rate': 0.0, 'growth_type': 'none'}

    alive = np.array([m['alive_ratio'] for m in metric_history])

    # Need to start sparse
    if alive[0] > 0.3:
        return {'growth_score': 0.0, 'growth_rate': 0.0, 'growth_type': 'none'}

    # Check for sustained growth: alive count should increase over time
    # Split into quarters and check each is higher than the previous
    n = len(alive)
    quarters = [alive[i*n//4:(i+1)*n//4].mean() for i in range(4)]

    monotonic_quarters = sum(1 for i in range(3) if quarters[i+1] > quarters[i] * 1.02)
    if monotonic_quarters < 2:
        return {'growth_score': 0.0, 'growth_rate': 0.0, 'growth_type': 'none'}

    # Growth rate
    growth = alive[-1] - alive[0]
    if growth < 0.01:
        return {'growth_score': 0.0, 'growth_rate': 0.0, 'growth_type': 'none'}

    # Classify growth type
    mid = alive[n//2]
    expected_linear_mid = (alive[0] + alive[-1]) / 2
    if mid > expected_linear_mid * 1.1:
        growth_type = 'accelerating'
    elif mid < expected_linear_mid * 0.9:
        growth_type = 'decelerating'
    else:
        growth_type = 'linear'

    # Score: steady growth is more interesting than explosive growth
    # Penalize if it just fills everything
    if alive[-1] > 0.9:
        score = 0.2  # filled up — not that interesting
    elif alive[-1] > 0.5:
        score = 0.5
    else:
        score = 0.8  # grew but didn't saturate — possible gun/replicator

    # Bonus for linearity (suggests structured replication)
    diffs = np.diff(alive)
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) > 5:
        cv = np.std(positive_diffs) / (np.mean(positive_diffs) + 1e-10)
        if cv < 0.3:
            score = min(1.0, score + 0.2)  # very regular growth

    rate = growth / len(alive)

    return {
        'growth_score': float(score),
        'growth_rate': float(rate),
        'growth_type': growth_type,
    }


def analyze_clusters(grid, channel=0, threshold=0.5):
    """Analyze connected components in the grid (cluster analysis).

    Finds discrete structures, measures their sizes and isolation.

    Returns dict with:
      n_clusters: number of connected components
      cluster_score: 0-1 (high = multiple well-separated interesting clusters)
      largest_cluster_frac: fraction of alive cells in largest cluster
      mean_cluster_size: average cluster size in cells
    """
    from scipy import ndimage

    vol = (grid[:, :, :, channel] > threshold).astype(np.int32)
    alive = vol.sum()
    if alive < 5:
        return {'n_clusters': 0, 'cluster_score': 0.0,
                'largest_cluster_frac': 0.0, 'mean_cluster_size': 0}

    # Label connected components (6-connectivity)
    structure = np.zeros((3, 3, 3), dtype=np.int32)
    structure[1, 1, :] = 1
    structure[1, :, 1] = 1
    structure[:, 1, 1] = 1
    labels, n_clusters = ndimage.label(vol, structure=structure)

    if n_clusters == 0:
        return {'n_clusters': 0, 'cluster_score': 0.0,
                'largest_cluster_frac': 0.0, 'mean_cluster_size': 0}

    # Cluster sizes
    sizes = ndimage.sum(vol, labels, range(1, n_clusters + 1))
    sizes = np.array(sizes, dtype=float)
    largest = sizes.max()
    mean_size = sizes.mean()

    # Score: multiple medium-sized clusters is most interesting
    # (single blob = boring, dust = boring, multiple structures = gliders!)
    alive_frac = alive / max(1, grid.shape[0] * grid.shape[1] * grid.shape[2])
    if n_clusters == 1:
        score = 0.1  # single blob
    elif n_clusters > 1000:
        score = 0.05  # dust/noise
    elif n_clusters > 50 and alive_frac > 0.2:
        score = 0.05  # many clusters + high alive = noise, not discrete structures
    else:
        # Ideal: 2-50 clusters, not dominated by one huge one
        size_variety = 1.0 - (largest / alive)  # 0 = one cluster has everything
        count_score = min(1.0, n_clusters / 10.0) * min(1.0, 50.0 / max(n_clusters, 1))
        score = 0.3 * count_score + 0.4 * size_variety + 0.3 * min(1.0, mean_size / 100.0)

    return {
        'n_clusters': int(n_clusters),
        'cluster_score': float(min(1.0, score)),
        'largest_cluster_frac': float(largest / alive),
        'mean_cluster_size': float(mean_size),
    }


def measure_symmetry(grid, channel=0, threshold=0.5):
    """Measure rotational and reflective symmetry of the grid.

    Returns dict with:
      symmetry_score: 0-1 (1 = perfectly symmetric under all transforms)
      reflection_score: avg reflective symmetry (x, y, z mirrors)
      rotation_score: avg rotational symmetry (90° rotations)
    """
    vol = (grid[:, :, :, channel] > threshold).astype(np.float32)
    alive = vol.sum()
    if alive < 5:
        return {'symmetry_score': 0.0, 'reflection_score': 0.0, 'rotation_score': 0.0}

    total = vol.size

    # Reflective symmetry along each axis
    ref_scores = []
    for ax in range(3):
        flipped = np.flip(vol, axis=ax)
        agreement = np.sum(vol == flipped) / total
        ref_scores.append(agreement)

    # 90° rotational symmetry (rotate around each axis)
    rot_scores = []
    # Around Z axis
    rotated = np.rot90(vol, k=1, axes=(0, 1))
    rot_scores.append(np.sum(vol == rotated) / total)
    # Around Y axis
    rotated = np.rot90(vol, k=1, axes=(0, 2))
    rot_scores.append(np.sum(vol == rotated) / total)
    # Around X axis
    rotated = np.rot90(vol, k=1, axes=(1, 2))
    rot_scores.append(np.sum(vol == rotated) / total)

    ref_mean = float(np.mean(ref_scores))
    rot_mean = float(np.mean(rot_scores))

    # Combined score — subtract the baseline agreement for a random grid
    # (for sparse grids, random agreement is ~(1-alive_frac)^2 + alive_frac^2)
    alive_frac = alive / total
    baseline = (1 - alive_frac)**2 + alive_frac**2
    sym_score = max(0.0, (ref_mean + rot_mean) / 2.0 - baseline) / max(0.01, 1.0 - baseline)

    return {
        'symmetry_score': float(min(1.0, sym_score)),
        'reflection_score': float(max(0, ref_mean - baseline) / max(0.01, 1.0 - baseline)),
        'rotation_score': float(max(0, rot_mean - baseline) / max(0.01, 1.0 - baseline)),
    }


def analyze_dynamics(grid_snapshots, metric_history, channel=0, threshold=0.5):
    """Run all advanced dynamics analysis on a time series of grid snapshots.

    Returns a dict with period, translation, growth, cluster, and symmetry metrics.
    """
    result = {}
    result.update(detect_period(grid_snapshots, channel, threshold))
    result.update(detect_translation(grid_snapshots, channel, threshold))
    result.update(detect_growth(metric_history))
    result.update(analyze_clusters(grid_snapshots[-1], channel, threshold))
    result.update(measure_symmetry(grid_snapshots[-1], channel, threshold))
    return result


# ── Run a single trial ────────────────────────────────────────────────

def run_trial(ctx, rule_name, size=32, seed=42, steps=100, sample_interval=15,
              params=None, dt=None, verbose=False, capture_dynamics=False,
              init_density=None, init_override=None):
    """Run a CA for N steps, sample metrics every interval, return summary.

    If capture_dynamics=True, stores grid snapshots every sample_interval
    and runs advanced dynamics analysis (period, translation, growth, clusters,
    symmetry).  This uses more memory but enables detection of gliders,
    oscillators, guns, etc.
    """
    _t_wall0 = time.perf_counter() if _PROFILE_ENABLED else 0.0
    _t0 = _t_wall0
    runner = HeadlessRunner(ctx, rule_name, size=size, seed=seed, params=params,
                           dt=dt, init_density=init_density,
                           init_override=init_override)
    if _PROFILE_ENABLED:
        _profile_totals['init'] += time.perf_counter() - _t0

    metric_history = []
    grid_snapshots = [] if capture_dynamics else None
    prev_grid = None
    # Use GPU metrics path when (a) globally enabled, (b) we don't need full
    # grid readback for capture_dynamics. The readback is required when
    # snapshotting for analyze_dynamics(), so GPU metrics buy nothing then.
    use_gpu_metrics = _GPU_METRICS_ENABLED and not capture_dynamics
    have_prev_gpu = False  # tracks runner's prev-tex state

    # Early termination state
    _abort = False
    _abort_reason = None

    for step in range(steps + 1):
        if step % sample_interval == 0:
            if use_gpu_metrics:
                if _PROFILE_ENABLED:
                    _t0 = time.perf_counter()
                m = runner.compute_metrics_gpu(prev_is_valid=have_prev_gpu)
                m['step'] = step
                if _PROFILE_ENABLED:
                    _profile_totals['metrics'] += time.perf_counter() - _t0
                    _profile_totals['reads'] += 1  # count for averaging
                have_prev_gpu = True
                grid = None  # not read back; callers below must handle this
            else:
                if _PROFILE_ENABLED:
                    _t0 = time.perf_counter()
                grid = runner.read_grid()
                if _PROFILE_ENABLED:
                    _profile_totals['read'] += time.perf_counter() - _t0
                    _profile_totals['reads'] += 1
                    _t0 = time.perf_counter()
                m = compute_metrics(grid, prev_grid, runner)
                m['step'] = step
                if _PROFILE_ENABLED:
                    _profile_totals['metrics'] += time.perf_counter() - _t0

            # Auto-calibrate alive threshold on first sample if it gives extreme results
            # This catches cases where parameter exploration shifts the field range
            if len(metric_history) == 0 and runner.measure_mode not in ('element', 'discrete'):
                ar = m['alive_ratio']
                if ar < 0.005 or ar > 0.995:
                    # Need full grid for percentile calibration. If we're on
                    # the GPU path we have to read back this once.
                    if grid is None:
                        if _PROFILE_ENABLED:
                            _t0 = time.perf_counter()
                        grid = runner.read_grid()
                        if _PROFILE_ENABLED:
                            _profile_totals['read'] += time.perf_counter() - _t0
                    ch = grid[:, :, :, runner.measure_channel]
                    if runner.measure_mode == 'wave':
                        ch = np.abs(ch)
                    elif runner.measure_mode == 'deviation':
                        ch = np.abs(ch - ch.mean())
                    # Set threshold at 75th percentile — targets ~25% alive
                    new_thresh = float(np.percentile(ch[np.isfinite(ch)], 75)) if np.isfinite(ch).any() else runner.alive_threshold
                    if new_thresh > 1e-6 and abs(new_thresh - runner.alive_threshold) > 1e-6:
                        runner.alive_threshold = new_thresh
                        if use_gpu_metrics:
                            # Re-dispatch with the new threshold. prev tex was
                            # set to current src by previous call; not valid
                            # for activity yet (no prior frame), so pass
                            # have_prev=False to match original semantics.
                            m = runner.compute_metrics_gpu(prev_is_valid=False)
                            have_prev_gpu = True
                        else:
                            m = compute_metrics(grid, prev_grid, runner)
                        m['step'] = step

            metric_history.append(m)

            if capture_dynamics:
                grid_snapshots.append(grid.copy())

            if verbose:
                print(f"  step {step:4d}: alive={m['alive_count']:6d} "
                      f"({m['alive_ratio']:.3f}) activity={m['activity']:.4f} "
                      f"surface={m['surface_ratio']:.3f} "
                      f"range=[{m['min']:.3f}, {m['max']:.3f}]"
                      f"{'  NaN!' if m['has_nan'] else ''}"
                      f"{'  Inf!' if m['has_inf'] else ''}")

            # On GPU path the grid is None; prev_grid stays None (CPU
            # compute_metrics is no longer called for activity, the GPU
            # shader handles it via prev_tex).
            if grid is not None:
                prev_grid = grid

            # Early termination checks (after at least 3 samples)
            if len(metric_history) >= 3 and step >= 30:
                # NaN/Inf — unsalvageable
                if m['has_nan'] or m['has_inf']:
                    _abort = True
                    _abort_reason = 'nan_inf'
                # Completely dead
                elif m['alive_ratio'] < 1e-6 and m['activity'] < 1e-6:
                    _abort = True
                    _abort_reason = 'dead'
                # Fully saturated with no activity
                elif m['alive_ratio'] > 0.99 and m['activity'] < 1e-4:
                    _abort = True
                    _abort_reason = 'saturated'

            if _abort:
                break

        if step < steps:
            if _PROFILE_ENABLED:
                _t0 = time.perf_counter()
            runner.step()
            if _PROFILE_ENABLED:
                _profile_totals['step'] += time.perf_counter() - _t0
                _profile_totals['steps'] += 1

    channel = runner.measure_channel
    measure_mode = runner.measure_mode
    params = dict(runner.params)
    dt = runner.dt
    # Always need final grid for analyze_structure (CPU-side slice metrics).
    # On the GPU-metrics path we never read it back during the loop, so do
    # it here. Counted under 'read' for honest profiling.
    if grid is None:
        if _PROFILE_ENABLED:
            _t0 = time.perf_counter()
        grid = runner.read_grid()
        if _PROFILE_ENABLED:
            _profile_totals['read'] += time.perf_counter() - _t0
    if _PROFILE_ENABLED:
        _t0 = time.perf_counter()
    runner.release()
    if _PROFILE_ENABLED:
        _profile_totals['release'] += time.perf_counter() - _t0

    score = score_interestingness(metric_history)
    final = metric_history[-1]

    # Structural analysis on the final grid
    structure = analyze_structure(grid, channel=channel)

    result = {
        'rule': rule_name,
        'params': params,
        'dt': dt,
        'size': size,
        'seed': seed,
        'steps': steps,
        'score': score,
        'final_alive': final['alive_ratio'],
        'median_alive': float(np.median([m['alive_ratio'] for m in metric_history])),
        'final_activity': final['activity'],
        'mean_activity': float(np.mean([m['activity'] for m in metric_history[1:]])) if len(metric_history) > 1 else 0.0,
        'final_surface': final['surface_ratio'],
        'has_nan': any(m['has_nan'] for m in metric_history),
        'has_inf': any(m['has_inf'] for m in metric_history),
        'history': metric_history,
        'measure_mode': measure_mode,
    }
    if init_density is not None:
        result['init_density'] = init_density
    result.update(structure)

    # Advanced dynamics analysis
    if capture_dynamics and grid_snapshots:
        if _PROFILE_ENABLED:
            _t0 = time.perf_counter()
        dynamics = analyze_dynamics(grid_snapshots, metric_history, channel=channel)
        result.update(dynamics)
        if _PROFILE_ENABLED:
            _profile_totals['dynamics'] += time.perf_counter() - _t0

    if _PROFILE_ENABLED:
        _profile_totals['wall'] += time.perf_counter() - _t_wall0
        _profile_totals['trials'] += 1

    return result


# ── Commands ──────────────────────────────────────────────────────────

def cmd_audit(ctx, args):
    """Audit all presets — verify they produce interesting behavior."""
    from simulator import RULE_PRESETS
    print(f"{'Rule':<25} {'Score':>5} {'Alive':>7} {'Act':>6} {'Surf':>5} {'GoL':>5} {'ProjC':>5} {'MI':>5} {'Status'}")
    print("─" * 90)

    results = []
    for name in RULE_PRESETS:
        result = run_trial(ctx, name, size=args.size, steps=args.steps,
                          seed=args.seed, verbose=args.verbose)
        results.append(result)

        status = "OK"
        if result['has_nan']:
            status = "NaN!"
        elif result['has_inf']:
            status = "Inf!"
        elif result['score'] < 0.1:
            status = "DEAD/SAT"
        elif result['score'] < 0.3:
            status = "WEAK"

        print(f"{name:<25} {result['score']:5.2f} "
              f"{result['final_alive']:7.3f} "
              f"{result['final_activity']:6.3f} "
              f"{result['final_surface']:5.3f} "
              f"{result.get('gol_coherence_max', 0):5.3f} "
              f"{result.get('projection_complexity', 0):5.3f} "
              f"{result.get('slice_mi_max', 0):5.3f} {status}")

    print()
    good = sum(1 for r in results if r['score'] >= 0.3)
    weak = sum(1 for r in results if 0.1 <= r['score'] < 0.3)
    bad = sum(1 for r in results if r['score'] < 0.1)
    print(f"Summary: {good} good, {weak} weak, {bad} dead/broken out of {len(results)}")


def cmd_test(ctx, args):
    """Test a single rule with verbose output."""
    print(f"Testing: {args.rule} (size={args.size}, steps={args.steps}, seed={args.seed})")
    print()
    result = run_trial(ctx, args.rule, size=args.size, steps=args.steps,
                      seed=args.seed, verbose=True)
    print()
    print(f"Score: {result['score']:.3f}")
    print(f"Params: {result['params']}")
    print(f"dt: {result['dt']}")


def randomize_params(preset, rng):
    """Generate random parameters within the preset's defined ranges.

    Uses log-uniform sampling for params with >10x range spread to properly
    explore both small and large regimes (e.g., diffusion 0.001-1.0).
    """
    params = {}
    ranges = preset['param_ranges']
    for name, (lo, hi) in ranges.items():
        if isinstance(lo, int) and isinstance(hi, int):
            params[name] = int(rng.randint(lo, hi + 1))
        else:
            # Log-uniform for wide ranges where both ends are positive
            if lo > 0 and hi / lo > 10.0:
                params[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                params[name] = float(rng.uniform(lo, hi))
    # Ensure min <= max for paired parameters (Birth, Survive, etc.)
    for prefix in ('Birth', 'Survive'):
        k_min, k_max = f'{prefix} min', f'{prefix} max'
        if k_min in params and k_max in params:
            a, b = params[k_min], params[k_max]
            params[k_min], params[k_max] = min(a, b), max(a, b)
    return params


def _randomize_params_elegance(preset, rng):
    """Generate params biased toward elegant patterns: moderate diffusion, low dt.

    Elegance favors structured sparse patterns — push toward lower reaction rates,
    moderate diffusion (enough to form structures but not wash out), and params
    that create slow, deliberate pattern formation.
    """
    params = randomize_params(preset, rng)
    ranges = preset['param_ranges']

    # Bias diffusion-like params toward mid-range (structures need moderate diffusion)
    for name in params:
        lo, hi = ranges[name]
        if isinstance(lo, int):
            continue
        name_l = name.lower()
        if 'diffusion' in name_l or name_l.startswith('d') and ('inhibitor' in name_l or 'activator' in name_l):
            # Bias toward geometric mean of range (moderate values)
            if lo > 0:
                geo_mean = np.sqrt(lo * hi)
                # 60% chance: sample around geometric mean ±50%
                if rng.random() < 0.6:
                    narrow_lo = max(lo, geo_mean * 0.5)
                    narrow_hi = min(hi, geo_mean * 2.0)
                    params[name] = float(rng.uniform(narrow_lo, narrow_hi))
        elif 'rate' in name_l or 'growth' in name_l or 'reaction' in name_l:
            # Bias reaction rates toward lower half — slower reactions = more structured
            if rng.random() < 0.5:
                mid = (lo + hi) * 0.5
                params[name] = float(rng.uniform(lo, mid))

    return params


    """Generate params biased toward GoL-like patterns: narrow birth, rare survival."""
    ranges = preset['param_ranges']
    params = randomize_params(preset, rng)  # fallback for non-GoL params

    # Only apply bias if this preset has Birth/Survive params
    if 'Birth min' not in ranges:
        return params

    b_lo, b_hi = ranges['Birth min']

    # Birth: favor narrow ranges (50% exact, 30% ±1-2, 20% random)
    r = rng.random()
    if r < 0.50:
        # Exact birth count — most GoL-like
        k = int(rng.randint(max(b_lo, 1), min(b_hi, 8) + 1))
        params['Birth min'] = k
        params['Birth max'] = k
    elif r < 0.80:
        # Narrow birth range (width 1-2)
        k = int(rng.randint(max(b_lo, 1), min(b_hi, 8) + 1))
        w = int(rng.randint(1, 3))
        params['Birth min'] = max(b_lo, k - w // 2)
        params['Birth max'] = min(b_hi, k + (w - w // 2))
    # else: keep fully random birth from base randomize_params

    # Survival: favor impossible or very high threshold
    if 'Survive min' in ranges:
        s_lo, s_hi = ranges['Survive min']
        r = rng.random()
        if r < 0.40:
            # Impossible survival (min > max)
            params['Survive min'] = int(rng.randint(max(s_lo, 10), s_hi + 1))
            params['Survive max'] = int(rng.randint(s_lo, max(s_lo + 1, params['Survive min'])))
        elif r < 0.70:
            # Very high threshold (survive needs many neighbors)
            params['Survive min'] = int(rng.randint(max(s_lo, 10), s_hi + 1))
            params['Survive max'] = int(rng.randint(params['Survive min'], s_hi + 1))
        # else: keep random survival

    return params


def _get_metric(result, metric_name):
    """Extract a metric value from a trial result. Supports compound metrics."""
    if metric_name == 'score':
        return result['score']
    elif metric_name == 'gol_coherence':
        return result.get('gol_coherence_max', 0)
    elif metric_name == 'projection':
        return result.get('projection_complexity', 0)
    elif metric_name == 'structure':
        return result.get('projection_structure', 0)
    elif metric_name == 'slice_mi':
        return result.get('slice_mi_max', 0)
    elif metric_name == 'combined':
        # Weighted combination: interestingness + structural richness
        s = result['score']
        pc = result.get('projection_complexity', 0)
        ps = result.get('projection_structure', 0)
        mi = result.get('slice_mi_max', 0)
        sv = result.get('spatial_variation', 0)
        # Spatial variation gates slice_mi: identical uniform slices shouldn't score well
        mi_adj = mi * max(sv, 0.1)  # dampen MI when spatially uniform
        raw = s * 0.30 + ps * 0.25 + sv * 0.20 + mi_adj * 0.15 + pc * 0.10
        # Cluster-mass penalty: if one connected component owns >50% of
        # alive cells, the rule has merged into a single mass — the opposite
        # of an interesting Life-like ensemble of distinct structures.
        lcf = result.get('largest_cluster_frac', 0.0)
        alive = result.get('final_alive', 0.0)
        if alive > 0.05 and lcf > 0.5:
            raw -= min(0.30, (lcf - 0.5) * 0.6)  # up to -0.30 at lcf=1.0
        return max(0.0, raw)
    elif metric_name == 'period':
        return result.get('period_score', 0)
    elif metric_name == 'glider':
        return result.get('translation_score', 0)
    elif metric_name == 'growth':
        return result.get('growth_score', 0)
    elif metric_name == 'clusters':
        return result.get('cluster_score', 0)
    elif metric_name == 'symmetry':
        return result.get('symmetry_score', 0)
    elif metric_name == 'elegance':
        # Compound metric: rewards periodicity, distinct structures, symmetry
        # Penalizes global oscillators (full-cube blinkers)
        s = result['score']
        period = result.get('period_score', 0)
        glider = result.get('translation_score', 0)
        clust = result.get('cluster_score', 0)
        sym = result.get('symmetry_score', 0)
        grow = result.get('growth_score', 0)
        pc = result.get('projection_complexity', 0)
        sv = result.get('spatial_variation', 0)
        alive = result.get('final_alive', 0)
        activity = result.get('final_activity', 0)

        # Penalize (not reject) dense/flashing patterns — let scoring handle it
        # Skip blinker penalty for continuous-field CAs
        density_penalty = 0.0
        mmode = result.get('measure_mode', 'discrete')
        is_pde = mmode in ('continuous', 'wave', 'deviation', 'phase_coherence')
        if not is_pde:
            if alive > 0.40 and activity > 0.6:
                act_ratio = activity / max(alive, 0.01)
                if act_ratio > 1.5:
                    density_penalty = 0.25  # global blinker
            elif alive > 0.65:
                density_penalty = 0.10  # quite dense

        # Penalize spatially uniform patterns (global oscillators)
        if sv < 0.05:
            density_penalty += 0.20  # very uniform = boring
        elif sv < 0.15:
            density_penalty += 0.10  # somewhat uniform

        # Sparsity/density bonus depends on rule type
        density_bonus = 0.0
        if is_pde:
            # PDE rules: reward structured density in [0.15, 0.70] range
            # Dense patterns are GOOD for PDEs — reward activity + structure
            surface = result.get('final_surface', 0)
            if 0.15 <= alive <= 0.70 and activity > 0.0005 and surface > 0.3:
                density_bonus = 0.10  # structured PDE pattern
            elif alive > 0.70 and activity > 0.005 and sv > 0.15:
                density_bonus = 0.05  # very dense but still evolving and structured
        else:
            # Discrete CAs: GoL-like patterns use <25% of grid
            if alive < 0.03:
                density_bonus = 0.15
            elif alive < 0.10:
                density_bonus = 0.10
            elif alive < 0.25:
                density_bonus = 0.05
            elif alive < 0.40:
                density_bonus = 0.02

        raw = (s * 0.10 + period * 0.15 + glider * 0.20
               + clust * 0.15 + sym * 0.10 + grow * 0.05 + pc * 0.05
               + sv * 0.10 + density_bonus)
        # Cluster-mass penalty: one giant blob is the opposite of elegant.
        lcf = result.get('largest_cluster_frac', 0.0)
        if alive > 0.05 and lcf > 0.5:
            density_penalty += min(0.30, (lcf - 0.5) * 0.6)
        return max(0.0, raw - density_penalty)
    elif metric_name == 'gol_like':
        # Metric targeting classic 2D-GoL-like patterns in 3D:
        # thin membrane structures with flickering steady-state, narrow birth
        # range, effectively no survival.
        alive = result.get('final_alive', 0)
        activity = result.get('final_activity', 0)
        surface = result.get('final_surface', 0)
        coherence = result.get('gol_coherence_max', 0)
        pc = result.get('projection_complexity', 0)
        ps = result.get('projection_structure', 0)
        params = result.get('params', {})

        # Hard reject
        if alive < 0.02 or alive > 0.45:
            return 0.0
        if activity < 0.005:
            return 0.0

        score = 0.0

        # Surface ratio: want 1.0 (all cells exposed, no solid interior)
        if surface >= 0.95:
            score += 0.20
        elif surface >= 0.80:
            score += 0.10

        # Activity/alive ratio: classic GoL oscillators have ratio ~1.5-2.0
        # (most cells die and are reborn each step)
        ratio = activity / max(alive, 0.001)
        if 1.3 <= ratio <= 2.2:
            score += 0.25
        elif 1.0 <= ratio <= 2.5:
            score += 0.10

        # Alive in sweet spot (0.05-0.20)
        if 0.05 <= alive <= 0.20:
            score += 0.15
        elif alive <= 0.25:
            score += 0.05

        # Moderate GoL coherence (not too high = dead, not too low = chaos)
        if 0.65 <= coherence <= 0.90:
            score += 0.15
        elif 0.55 <= coherence <= 0.95:
            score += 0.05

        # Narrow birth range bonus (exact k or small window)
        bmin = params.get('Birth min', 0)
        bmax = params.get('Birth max', 26)
        birth_width = bmax - bmin
        if birth_width == 0:
            score += 0.15  # exact birth count (most GoL-like)
        elif birth_width <= 2:
            score += 0.08
        elif birth_width <= 4:
            score += 0.03

        # Projection structure (visible spatial patterns)
        score += min(pc * 0.5, 0.10)

        return min(score, 1.0)
    else:
        return result.get(metric_name, 0)


_DYNAMICS_METRICS = {'period', 'glider', 'growth', 'clusters', 'symmetry', 'elegance'}


def _needs_dynamics(metric_name):
    """Return True if this metric requires dynamics capture (grid snapshots)."""
    return metric_name in _DYNAMICS_METRICS


# Behavioural fingerprint dimensions used by novelty / dedup. Keep this in
# ONE place -- previously the same 10-12 features were inlined in three
# different spots, which made it impossible to keep the dedup logic and
# the novelty-bonus logic consistent.
_FINGERPRINT_KEYS = (
    'final_alive',
    'final_activity',
    'final_surface',
    'spatial_variation',
    'projection_complexity',
    'projection_structure',
    'gol_coherence_max',
    'period_score',
    'translation_score',
    'cluster_score',
    'symmetry_score',
    'growth_score',
)


def _fingerprint(r):
    """Extract the 12-D behavioural fingerprint as a numpy vector."""
    return np.array([float(r.get(k, 0.0)) for k in _FINGERPRINT_KEYS])


def _normalize_fingerprints(vectors):
    """Z-score every dimension of a stack of fingerprints across the run.
    Without this, raw L2 over the 12 features is dominated by whichever
    one or two features happen to have the largest absolute spread (often
    `projection_complexity` or `gol_coherence_max`), so the other 10
    dimensions silently contribute almost nothing to novelty distance and
    the search collapses onto the same one or two axes of variation.
    Returns (normalized_vectors, mu, sigma) so single new results can be
    normalized into the same space later via (v - mu) / sigma.
    """
    if len(vectors) == 0:
        return vectors, np.zeros(len(_FINGERPRINT_KEYS)), np.ones(len(_FINGERPRINT_KEYS))
    arr = np.asarray(vectors, dtype=np.float64)
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)  # avoid div-zero on flat dims
    return (arr - mu) / sigma, mu, sigma


def _make_discovery(r):
    """Build a discovery dict from a trial result, including any dynamics data."""
    d = {
        'rule': r['rule'],
        'params': r['params'],
        'dt': r['dt'],
        'score': r['score'],
        'seed': r['seed'],
        'final_alive': r['final_alive'],
        'median_alive': r.get('median_alive', r['final_alive']),
        'final_activity': r['final_activity'],
        'mean_activity': r.get('mean_activity', r['final_activity']),
        'final_surface': r['final_surface'],
        'gol_coherence': r.get('gol_coherence_max', 0),
        'projection_complexity': r.get('projection_complexity', 0),
        'projection_structure': r.get('projection_structure', 0),
        'slice_mi': r.get('slice_mi_max', 0),
        'spatial_variation': r.get('spatial_variation', 0),
    }
    if 'init_density' in r:
        d['init_density'] = r['init_density']
    # Include dynamics metrics if present
    for key in ('period', 'period_score', 'translation_score', 'translation_speed',
                'growth_score', 'growth_rate', 'growth_type',
                'n_clusters', 'cluster_score', 'symmetry_score'):
        if key in r:
            d[key] = r[key]
    # Convert tuple to list for JSON serialization
    if 'translation_dir' in r:
        d['translation_dir'] = list(r['translation_dir'])
    return d


def _is_quality(result, min_score=0.15):
    """Check if a result is genuinely interesting (not fill-all/die-off/frozen/noise)."""
    alive = result.get('final_alive', 0)
    median_alive = result.get('median_alive', alive)
    repr_alive = max(alive, median_alive)  # best of final/median
    activity = result.get('final_activity', 0)
    mean_act = result.get('mean_activity', activity)
    best_activity = max(activity, mean_act)
    score = result.get('score', 0)
    # Reject dead — both final AND median must be dead
    if repr_alive < 0.003:
        return False
    # Reject fill-all (>95% alive with low structure)
    if repr_alive > 0.95 and result.get('projection_complexity', 0) < 0.2 and best_activity < 0.001:
        return False
    # For phase_coherence (kuramoto): alive>0.7 means disordered (noise), not domains
    if alive > 0.70 and result.get('measure_mode', '') == 'phase_coherence':
        return False
    # Reject frozen (no activity through the whole run)
    if best_activity < 0.0005 and score < 0.3:
        return False
    # Reject global blinker: high alive + every cell toggles every step
    # Skip for continuous-field CAs where activity is PDE evolution, not toggling
    # Use act_ratio > 1.8 (near-complete toggling) to avoid rejecting evolving CAs
    mode = result.get('measure_mode', 'discrete')
    if mode not in ('deviation', 'phase_coherence', 'continuous', 'wave'):
        if repr_alive > 0.15 and activity > 0.02:
            act_ratio = activity / max(repr_alive, 0.01)
            if act_ratio > 1.8:
                return False  # global blinker — nearly every cell toggles
            if act_ratio > 0.8 and repr_alive > 0.40:
                return False  # dense partial blinker
    # Reject spatially uniform patterns (global oscillators, uniform noise)
    # Skip for PDE rules where sv may be low but field has real structure (surface_ratio)
    sv = result.get('spatial_variation', -1)
    mode = result.get('measure_mode', 'discrete')
    is_pde = mode in ('deviation', 'phase_coherence', 'continuous', 'wave')
    if sv >= 0 and sv < 0.02 and repr_alive > 0.05 and best_activity > 0.01:
        if not is_pde or result.get('final_surface', 0) < 0.15:
            return False
    # Reject low score (scoring already penalizes density/flashing)
    if score < min_score:
        return False
    return True


def cmd_sweep(ctx, args):
    """Random parameter sweep — find interesting parameter combinations."""
    from simulator import RULE_PRESETS
    preset = RULE_PRESETS[args.rule]

    metric = getattr(args, 'metric', 'score')
    dynamics = _needs_dynamics(metric)
    print(f"Sweeping: {args.rule} ({args.trials} trials, size={args.size}, metric={metric})")
    if dynamics:
        print(f"  (dynamics capture enabled for {metric})")
    print(f"Param ranges: {preset['param_ranges']}")
    print()

    results = []
    rng = np.random.RandomState(args.seed)

    for trial in range(args.trials):
        params = randomize_params(preset, rng)
        trial_seed = int(rng.randint(0, 10_000_000))

        result = run_trial(ctx, args.rule, size=args.size, steps=args.steps,
                          seed=trial_seed, params=params, verbose=False,
                          capture_dynamics=dynamics)
        results.append(result)

        if result['score'] >= 0.3 or (trial + 1) % 20 == 0:
            tag = "***" if result['score'] >= 0.5 else "  *" if result['score'] >= 0.3 else "   "
            print(f"{tag} trial {trial+1:4d}: score={result['score']:.3f} "
                  f"alive={result['final_alive']:.3f} "
                  f"activity={result['final_activity']:.4f} "
                  f"params={_fmt_params(params)}")

    # Sort by chosen metric and show top results
    metric = getattr(args, 'metric', 'score')
    results.sort(key=lambda r: _get_metric(r, metric), reverse=True)
    top_n = min(args.top, len(results))

    print()
    print(f"Top {top_n} results (by {metric}):")
    print(f"{'#':>3} {'Score':>5} {'Alive':>7} {'Act':>6} {'GoL':>5} {'ProjC':>5} {'Struct':>5} {'MI':>5}  Params")
    print("─" * 100)
    for i, r in enumerate(results[:top_n]):
        print(f"{i+1:3d} {r['score']:5.3f} "
              f"{r['final_alive']:7.3f} "
              f"{r['final_activity']:6.3f} "
              f"{r.get('gol_coherence_max', 0):5.3f} "
              f"{r.get('projection_complexity', 0):5.3f} "
              f"{r.get('projection_structure', 0):5.3f} "
              f"{r.get('slice_mi_max', 0):5.3f}  {_fmt_params(r['params'])}")

    # Save discoveries
    if args.save:
        discoveries = [_make_discovery(r)
                       for r in results[:top_n]
                       if r['score'] >= 0.3 or _get_metric(r, metric) >= 0.3]

        if discoveries:
            save_path = os.path.join(os.path.dirname(__file__), args.save)
            # Append to existing file
            existing = []
            if os.path.exists(save_path):
                with open(save_path) as f:
                    existing = json.load(f)
            existing.extend(discoveries)
            with open(save_path, 'w') as f:
                json.dump(existing, f, indent=2)
            print(f"\nSaved {len(discoveries)} discoveries to {save_path}")
        else:
            print("\nNo discoveries above threshold to save.")

    # Stats
    scores = [r['score'] for r in results]
    print(f"\nSweep stats: mean={np.mean(scores):.3f} max={np.max(scores):.3f} "
          f">{0.3:.0%}: {sum(1 for s in scores if s >= 0.3)}/{len(scores)}")


def _param_distance(params_a, params_b, ranges):
    """Normalized Euclidean distance between two parameter sets in [0, 1]."""
    dists = []
    for name in params_a:
        if name.startswith("unused"):
            continue
        lo, hi = ranges[name]
        span = max(hi - lo, 1e-10)
        d = (float(params_a[name]) - float(params_b[name])) / span
        dists.append(d * d)
    return np.sqrt(sum(dists) / max(len(dists), 1))


def _novelty_bonus(result, archive, ranges):
    """Compute novelty bonus: average distance to k-nearest neighbours in archive,
    measured in PER-DIMENSION-NORMALIZED behavioural space + parameter space.

    Without normalization, raw L2 over the 12 fingerprint features collapses
    onto whichever 1-2 features have the widest absolute spread -- in our
    runs that's usually projection_complexity (range 0..3) drowning out
    period_score / translation_score (range 0..1). Result: the search
    keeps finding patterns "novel" along the same one axis and clustering
    on the others.

    Returns a bonus where ~0 means an exact behavioural match and >1 means
    "noticeably different in multiple independent dimensions" (since each
    z-scored dim contributes ~1 stdev when fully novel).
    """
    if not archive:
        return 1.0  # first result is maximally novel
    # Stack the archive fingerprints + this one, z-score every column once
    archive_fps = [_fingerprint(o) for o in archive]
    all_fps = archive_fps + [_fingerprint(result)]
    normed, _, _ = _normalize_fingerprints(all_fps)
    this_v = normed[-1]
    archive_v = normed[:-1]
    # Per-archive distance: blend behavioural (dominant) + param (regularizer)
    distances = []
    for v, other in zip(archive_v, archive):
        b_dist = float(np.linalg.norm(this_v - v) / np.sqrt(len(_FINGERPRINT_KEYS)))
        p_dist = _param_distance(result['params'], other['params'], ranges)
        # Behavioural carries 70% (it's what the user actually sees);
        # 30% on params keeps two visually-similar regimes in different
        # corners of param-space distinguishable.
        distances.append(0.7 * b_dist + 0.3 * p_dist)
    k = min(5, len(distances))
    distances.sort()
    return float(np.mean(distances[:k]))


# Density ranges per init type for randomization
_DENSITY_RANGES = {
    'random_very_sparse': (0.01, 0.15),
    'random_sparse': (0.02, 0.35),
    'random_dense': (0.10, 0.70),
    'random_smooth': (0.1, 0.8),
    'lenia_blobs': (0.05, 0.5),
    'gray_scott': (0.05, 0.30),       # V cluster coverage fraction
    'flocking': (0.3, 1.0),           # streak density multiplier
    'physarum': (0.3, 1.0),           # agent density multiplier
    'phase_separation': (0.02, 0.15), # perturbation amplitude
    'viscous_fingers': (0.5, 2.0),    # injection radius multiplier
    'fire': (0.3, 1.0),               # fuel density multiplier
    'fracture': (0.3, 1.0),           # stress intensity multiplier
    'bz_reaction': (0.5, 1.5),        # amplitude around limit cycle
    'morphogen': (0.5, 2.0),          # activator perturbation scale
    'galaxy': (0.05, 0.5),            # base density (affects gravity vs pressure balance)
    'lichen': (0.3, 1.5),             # seed density / resource multiplier
}


def _pick_init_for_params(rule_name, params, init_variants, rng):
    """Param-aware init selection: pick init that complements the parameter regime.

    Returns a variant name, or falls back to random if no heuristic applies.
    Heuristics are soft — 70% use the suggested init, 30% random for exploration.
    """
    suggested = None

    if rule_name == 'predator_prey_3d':
        # High predation + low prey growth → separated init gives spatial refuge
        if params.get('Predation rate', 5) > 8 or params.get('Prey growth', 2) < 1.0:
            suggested = 'predator_prey_separated'

    elif rule_name.startswith('crystal_'):
        # High anisotropy or low undercooling → multi-seed shows competing growth fronts
        if params.get('Anisotropy strength', 0.5) > 1.5 or params.get('Undercooling', 0.5) < 0.15:
            suggested = 'crystal_multi_seed'

    elif rule_name in ('bz_spiral_waves', 'bz_turbulence', 'bz_excitable'):
        # For BZ rules: spiral seed creates organized phase singularities
        # Especially useful when diffusion is moderate (spiral waves can propagate)
        if params.get('Diffusion', 0.3) > 0.15:
            suggested = 'bz_spiral_seed'

    elif rule_name == 'erosion':
        # High gravity + high erosion → ridges init provides pre-carved channels
        if params.get('Gravity', 2) > 4 or params.get('Erosion', 1) > 3:
            suggested = 'erosion_ridges'

    elif rule_name == 'flocking_3d':
        # High alignment → vortex init seeds organized collective motion
        if params.get('Alignment', 2) > 2.5:
            suggested = 'flocking_vortex'

    elif rule_name == 'morphogen_spots':
        # High reaction rate → hotspots seed ensures nucleation sites exist
        if params.get('Reaction', 0.06) > 0.15:
            suggested = 'morphogen_hotspots'

    elif rule_name == 'game_of_life_3d':
        # Lower survive → centered init gives spatial structure room to grow
        if params.get('Survive min', 5) < 4:
            suggested = 'game_of_life_centered'

    elif rule_name == 'smoothlife_3d':
        # Narrow birth range → sparse blobs avoid uniform saturation
        if params.get('Birth range', 0.05) < 0.03:
            suggested = 'smoothlife_sparse'

    elif rule_name == 'gray_scott_worms':
        # High feed rate → dense V baseline helps worms nucleate faster
        if params.get('Feed rate', 0.046) > 0.05:
            suggested = 'gray_scott_worms_dense'

    elif rule_name == 'kuramoto_3d':
        # Low coupling → pre-formed clusters survive longer before sync
        if params.get('Coupling K', 0.5) < 0.3:
            suggested = 'kuramoto_clusters'

    elif rule_name == 'galaxy':
        # High gravity → filament init provides structure at Jeans scale
        if params.get('Gravity', 5) > 8:
            suggested = 'galaxy_filaments'

    elif rule_name == 'lichen':
        # High competition → dense init means territories form immediately
        if params.get('Competition', 1) > 3:
            suggested = 'lichen_dense'

    elif rule_name == 'mycelium':
        # High branching → foraging init gives tips + distant food targets
        if params.get('Branching', 0.8) > 1.2:
            suggested = 'mycelium_foraging'

    elif rule_name == 'fire':
        # Low diffusion → sparse fuel islands create interesting jump dynamics
        if params.get('Diffusion', 0.4) < 0.2:
            suggested = 'fire_sparse'

    elif rule_name == 'phase_separation':
        # High mobility → quench init gives pre-separated domains for fast coarsening
        if params.get('Mobility', 0.5) > 1.0:
            suggested = 'phase_separation_quench'

    elif rule_name == 'element_ca':
        # Layered init for high gravity → stratification is physical
        if params.get('Gravity', 2) > 3:
            suggested = 'element_layered'

    # Apply suggestion with 70% probability, 30% random for diversity
    if suggested and suggested in init_variants and rng.random() < 0.7:
        return suggested
    return init_variants[rng.randint(len(init_variants))]


def cmd_search(ctx, args):
    """Diversity-aware search: explores parameter space broadly, varies init density,
    and uses novelty bonus to favor discoveries in unexplored regions.
    All results above quality threshold are kept — rare finds are never discarded."""
    from simulator import RULE_PRESETS
    preset = RULE_PRESETS[args.rule]

    metric = getattr(args, 'metric', 'score')
    dynamics = _needs_dynamics(metric)
    print(f"Searching: {args.rule} ({args.trials} trials, size={args.size}, metric={metric})")
    if dynamics:
        print(f"  (dynamics capture enabled — longer trials recommended, use --steps 300+)")

    # Check if this rule supports density variation
    init_type = preset.get('init', '')
    can_vary_density = init_type in _DENSITY_RANGES
    if can_vary_density:
        d_lo, d_hi = _DENSITY_RANGES[init_type]
        print(f"  (varying init density: {d_lo:.2f} - {d_hi:.2f})")

    # Check if this rule supports dt variation
    dt_range = preset.get('dt_range')
    if dt_range:
        print(f"  (varying dt: {dt_range[0]:.3f} - {dt_range[1]:.3f})")

    # Check if this rule supports init variants
    init_variants = preset.get('init_variants')
    if init_variants and len(init_variants) > 1:
        print(f"  (init variants: {', '.join(init_variants)})")
    print()

    rng = np.random.RandomState(args.seed)

    # Use biased param generator for specific metrics
    if metric == 'gol_like':
        _gen_params = _randomize_params_gol_like
    elif metric == 'elegance':
        _gen_params = _randomize_params_elegance
    else:
        _gen_params = randomize_params

    # === MAP-Elites search ====================================================
    # Maintain a grid of best-scoring "elites" partitioned across a 3-D
    # behavioural-descriptor (BD) space:
    #   axis 0: log10(final_alive)              (density regime)
    #   axis 1: log10(final_activity)           (dynamism regime)
    #   axis 2: largest_cluster_frac            (1 blob vs many structures)
    #
    # Each cell holds at most one result -- the highest-scoring one ever
    # placed there. New trials pick a random filled cell as parent, mutate
    # its (params, dt, density, init) tuple jointly, and try to claim a
    # cell with the result. This:
    #   - automatically enforces diversity (each cell has one occupant)
    #   - preserves elites (only better-scoring trials displace incumbents)
    #   - keeps mutation in the parent's regime (joint dt+density mutation)
    #   - removes the O(N^2) novelty-archive recompute
    #
    # Replaces the prior random-explore + novelty-bonus + late-refinement
    # scheme. Output discoveries = all elites above quality threshold.
    GRID = 6  # 6**3 = 216 cells; with TRIALS=500 most filled cells get >1 hit
    elites = {}  # cell_key -> result

    def _bin(value, lo, hi, log=True):
        """Map value into [0, GRID-1] bin index."""
        if log:
            v = np.log10(max(value, lo))
            t = (v - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        else:
            t = (value - lo) / (hi - lo)
        return int(np.clip(t * GRID, 0, GRID - 1))

    def cell_key(r):
        a  = _bin(r.get('final_alive', 0),     1e-3, 1.0,  log=True)
        ac = _bin(r.get('final_activity', 0),  1e-4, 1.0,  log=True)
        lc = _bin(r.get('largest_cluster_frac', 0), 0.0, 1.0, log=False)
        return (a, ac, lc)

    def try_insert(r, base_val):
        # Quality floor: don't waste a cell on dead/NaN trials
        if base_val < 0.05:
            return False
        k = cell_key(r)
        cur = elites.get(k)
        if cur is None or _get_metric(cur, metric) < base_val:
            elites[k] = r
            return True
        return False

    def sample_init_density():
        return float(rng.uniform(d_lo, d_hi)) if can_vary_density else None

    def sample_dt(parent_dt=None):
        if not dt_range:
            return None
        if parent_dt is not None:
            # Joint mutation: stay in parent's regime, +-25%
            lo = max(dt_range[0], parent_dt * 0.75)
            hi = min(dt_range[1] * 1.5, parent_dt * 1.25)
            return float(rng.uniform(lo, hi))
        if rng.random() < 0.15:
            # Edge-of-stability probe
            return float(rng.uniform(dt_range[1] * 1.2, dt_range[1] * 1.5))
        return float(rng.uniform(dt_range[0], dt_range[1]))

    def sample_init(params=None):
        if not init_variants or len(init_variants) <= 1:
            return None
        if params is not None and rng.random() < 0.5:
            return _pick_init_for_params(args.rule, params, init_variants, rng)
        return init_variants[rng.randint(len(init_variants))]

    def evaluate(params, init_density, trial_dt, trial_init):
        trial_seed = int(rng.randint(0, 10_000_000))
        r = run_trial(ctx, args.rule, size=args.size, steps=args.steps,
                      seed=trial_seed, params=params, dt=trial_dt, verbose=False,
                      capture_dynamics=dynamics, init_density=init_density,
                      init_override=trial_init)
        if trial_init:
            r['init_variant'] = trial_init
        if init_density is not None:
            r['init_density'] = init_density
        bv = _get_metric(r, metric)
        # Backward-compat fields used by the display/save code below.
        r['_diverse_score'] = bv
        r['_novelty'] = 0.0
        return r, bv

    # Phase 1: bootstrap with uniform-random params to seed the grid.
    # Larger bootstrap fraction (40%) for small trial budgets so the grid
    # doesn't fall back into "no elites yet -> uniform" too often.
    n_bootstrap = max(40, args.trials * 2 // 5)
    n_bootstrap = min(n_bootstrap, args.trials)
    print(f"MAP-Elites: bootstrap {n_bootstrap} trials, then {args.trials - n_bootstrap} from elites")
    print(f"  BD grid: {GRID}x{GRID}x{GRID} = {GRID**3} cells "
          f"(alive log, activity log, largest_cluster_frac linear)")

    for trial in range(args.trials):
        # Decide: bootstrap (random) or mutate-from-elite?
        use_random = (trial < n_bootstrap) or (not elites)
        if use_random:
            params = _gen_params(preset, rng)
            init_density = sample_init_density()
            trial_dt = sample_dt()
            trial_init = sample_init(params)
            parent_key = None
        else:
            # Pick parent uniformly at random from filled cells -- this is
            # the curiosity-driven "fill the grid" objective. Cells already
            # filled are still re-sampled so successful regimes get more
            # local exploration; new cells get filled by mutations of
            # neighbouring elites.
            keys = list(elites.keys())
            parent_key = keys[rng.randint(len(keys))]
            parent = elites[parent_key]
            # Tighter mutation than before (5-15% vs 10-40%): we want to
            # stay in or adjacent to the parent's regime. The grid takes
            # care of macro-exploration; mutation only does local refine.
            scale = 0.05 + rng.random() * 0.10
            params = _mutate_params(parent['params'], preset['param_ranges'], rng, scale=scale)
            # Joint mutation of dt + density: inherit from parent then jitter.
            parent_dt = parent.get('dt', preset.get('dt'))
            trial_dt = sample_dt(parent_dt) if dt_range else None
            parent_dens = parent.get('init_density')
            if can_vary_density and parent_dens is not None:
                # +-30% multiplicative drift, clipped to range
                d = parent_dens * float(np.exp(rng.normal(0, 0.25)))
                init_density = float(np.clip(d, d_lo, d_hi))
            else:
                init_density = sample_init_density()
            # Init variant: usually inherit, sometimes resample.
            if init_variants and len(init_variants) > 1 and rng.random() < 0.2:
                trial_init = sample_init(params)
            else:
                trial_init = parent.get('init_variant')

        result, base_val = evaluate(params, init_density, trial_dt, trial_init)
        placed = try_insert(result, base_val)

        if base_val >= 0.3 or (trial + 1) % 50 == 0:
            tag = "***" if base_val >= 0.5 else "  *" if base_val >= 0.3 else "   "
            density_str = f" d={init_density:.2f}" if init_density is not None else ""
            dt_str = f" dt={trial_dt:.3f}" if trial_dt is not None else ""
            mark = "+" if placed else " "
            phase = "boot" if use_random else "mut "
            print(f"{tag} {phase} {trial+1:4d}: {metric}={base_val:.3f} "
                  f"cells={len(elites):3d}/{GRID**3} {mark}{density_str}{dt_str} "
                  f"{_fmt_params(params)}")

    print(f"\nMAP-Elites complete: {len(elites)}/{GRID**3} cells filled")
    # Output: all elites sorted by metric value.
    best_results = sorted(elites.values(),
                          key=lambda r: _get_metric(r, metric), reverse=True)
    top_n = min(args.top, len(best_results))
    print()
    print(f"Top {top_n} discoveries (by {metric} + novelty):")
    if dynamics:
        print(f"{'#':>3} {'Score':>5} {'Nov':>5} {'Per':>5} {'Glid':>5} {'Clust':>5} {'Sym':>5} {'Grow':>5} {'Eleg':>5}  Params")
        print("─" * 110)
        for i, r in enumerate(best_results[:top_n]):
            print(f"{i+1:3d} {r['score']:5.3f} "
                  f"{r.get('_novelty', 0):5.2f} "
                  f"{r.get('period_score', 0):5.3f} "
                  f"{r.get('translation_score', 0):5.3f} "
                  f"{r.get('cluster_score', 0):5.3f} "
                  f"{r.get('symmetry_score', 0):5.3f} "
                  f"{r.get('growth_score', 0):5.3f} "
                  f"{_get_metric(r, 'elegance'):5.3f}  {_fmt_params(r['params'])}")
    else:
        print(f"{'#':>3} {'Score':>5} {'Nov':>5} {'Alive':>7} {'Act':>6} {'GoL':>5} {'ProjC':>5} {'Struct':>5} {'MI':>5}  Params")
        print("─" * 110)
        for i, r in enumerate(best_results[:top_n]):
            print(f"{i+1:3d} {r['score']:5.3f} "
                  f"{r.get('_novelty', 0):5.2f} "
                  f"{r['final_alive']:7.3f} "
                  f"{r['final_activity']:6.3f} "
                  f"{r.get('gol_coherence_max', 0):5.3f} "
                  f"{r.get('projection_complexity', 0):5.3f} "
                  f"{r.get('projection_structure', 0):5.3f} "
                  f"{r.get('slice_mi_max', 0):5.3f}  {_fmt_params(r['params'])}")

    # Save -- deduplicate behaviorally similar discoveries before saving
    if args.save:
        min_q = getattr(args, 'min_quality', 0.25)
        candidates = [_make_discovery(r)
                      for r in best_results[:top_n]
                      if _is_quality(r, min_score=min_q)]

        # Behavioural deduplication in z-scored fingerprint space.
        # Threshold ~0.4 means "different by ~0.4 stdev average across the
        # 12 dimensions" -- empirically separates visually-distinct
        # patterns while collapsing ones that only differ by a few percent
        # in one or two dimensions. The OLD raw-L2 threshold of 0.03 over
        # un-normalized features admitted near-identical results because
        # one feature differing by 0.03 alone passed; conversely it could
        # also reject genuinely different results when one large-scale
        # feature happened to match closely.
        if candidates:
            cand_fps = [_fingerprint(c) for c in candidates]
            normed, _, _ = _normalize_fingerprints(cand_fps)
            kept_indices = []
            for i, v in enumerate(normed):
                too_close = False
                for j in kept_indices:
                    avg_z_dist = np.linalg.norm(v - normed[j]) / np.sqrt(len(_FINGERPRINT_KEYS))
                    if avg_z_dist < 0.4:
                        too_close = True
                        break
                if not too_close:
                    kept_indices.append(i)
            discoveries = [candidates[i] for i in kept_indices]
        else:
            discoveries = []

        if discoveries:
            save_path = os.path.join(os.path.dirname(__file__), args.save)
            existing = []
            if os.path.exists(save_path):
                with open(save_path) as f:
                    existing = json.load(f)
            existing.extend(discoveries)
            with open(save_path, 'w') as f:
                json.dump(existing, f, indent=2)
            n_deduped = len(candidates) - len(discoveries)
            dedup_msg = f" ({n_deduped} duplicates removed)" if n_deduped > 0 else ""
            print(f"\nSaved {len(discoveries)} discoveries to {save_path}{dedup_msg}")

    _profile_print()


def _mutate_params(params, ranges, rng, scale=0.15):
    """Mutate parameters by small random amounts within valid ranges."""
    mutated = {}
    for name, val in params.items():
        lo, hi = ranges[name]
        if isinstance(lo, int) and isinstance(hi, int):
            delta = max(1, int((hi - lo) * scale))
            new_val = int(val + rng.randint(-delta, delta + 1))
            mutated[name] = max(lo, min(hi, new_val))
        else:
            span = hi - lo
            delta = rng.normal(0, span * scale)
            mutated[name] = max(lo, min(hi, val + delta))
    return mutated


def _fmt_params(params):
    """Format params dict compactly."""
    parts = []
    for k, v in params.items():
        if k.startswith("unused"):
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def cmd_explore(ctx, args):
    """Search for interesting CAs then launch the simulator on the best find."""
    import subprocess
    from simulator import RULE_PRESETS
    preset = RULE_PRESETS[args.rule]
    metric = getattr(args, 'metric', 'combined')
    dynamics = _needs_dynamics(metric)

    print(f"Exploring: {args.rule} ({args.trials} trials, metric={metric})")
    if dynamics:
        print(f"  (dynamics capture enabled for {metric})")
    print(f"Will launch simulator on best result at size {args.view_size}")
    print()

    rng = np.random.RandomState(args.seed)
    best_results = []

    for trial in range(args.trials):
        if best_results and rng.random() < 0.4:
            parent = best_results[rng.randint(len(best_results))]
            params = _mutate_params(parent['params'], preset['param_ranges'], rng, scale=0.15)
        else:
            params = randomize_params(preset, rng)

        trial_seed = int(rng.randint(0, 10_000_000))
        result = run_trial(ctx, args.rule, size=args.size, steps=args.steps,
                          seed=trial_seed, params=params, verbose=False,
                          capture_dynamics=dynamics)

        best_results.append(result)
        best_results.sort(key=lambda r: _get_metric(r, metric), reverse=True)
        best_results = best_results[:max(args.top * 2, 20)]

        m_val = _get_metric(result, metric)
        if m_val >= 0.3 or (trial + 1) % 10 == 0:
            best_val = _get_metric(best_results[0], metric)
            bar = "█" * int(m_val * 20)
            print(f"  {trial+1:3d}/{args.trials}  {metric}={m_val:.3f} {bar}  (best={best_val:.3f})")

    # Save top results
    top_n = min(args.top, len(best_results))
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save)
    existing = []
    if os.path.exists(save_path):
        with open(save_path) as f:
            existing = json.load(f)

    new_start_idx = len(existing)
    discoveries = [_make_discovery(r) for r in best_results[:top_n]]

    existing.extend(discoveries)
    with open(save_path, 'w') as f:
        json.dump(existing, f, indent=2)

    best_idx = new_start_idx  # index of the best result in the file

    print()
    print(f"Saved {len(discoveries)} discoveries (indices {new_start_idx}-{new_start_idx + len(discoveries) - 1})")
    print()
    print(f"{'#':>3} {'Score':>5} {'GoL':>5} {'ProjC':>5} {'Struct':>5} {'MI':>5}  {metric}")
    print("─" * 60)
    for i, r in enumerate(best_results[:top_n]):
        print(f"  {new_start_idx + i:3d} {r['score']:5.3f} "
              f"{r.get('gol_coherence_max', 0):5.3f} "
              f"{r.get('projection_complexity', 0):5.3f} "
              f"{r.get('projection_structure', 0):5.3f} "
              f"{r.get('slice_mi_max', 0):5.3f}  "
              f"{_get_metric(r, metric):.3f}")

    print(f"\nLaunching simulator on discovery #{best_idx}...")
    sim_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulator.py')

    # Note: the headless context will be cleaned up by main()'s finally block
    # after this subprocess returns.
    subprocess.run([
        sys.executable, sim_script,
        '--discovery', save_path,
        '--discovery-index', str(best_idx),
        '--size', str(args.view_size),
    ])


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3D CA Test Harness")
    parser.add_argument('--size', type=int, default=32, help='Grid size (default: 32)')
    parser.add_argument('--steps', type=int, default=100, help='Simulation steps (default: 100)')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument('--verbose', '-v', action='store_true')

    sub = parser.add_subparsers(dest='command')

    # audit: test all presets
    sub.add_parser('audit', help='Audit all rule presets')

    # test: test one rule
    p_test = sub.add_parser('test', help='Test a single rule')
    p_test.add_argument('rule', type=str, help='Rule name')

    # sweep: random parameter sweep
    p_sweep = sub.add_parser('sweep', help='Random parameter sweep')
    p_sweep.add_argument('rule', type=str, help='Rule name to sweep')
    p_sweep.add_argument('--trials', type=int, default=50, help='Number of random trials')
    p_sweep.add_argument('--top', type=int, default=10, help='Show top N results')
    p_sweep.add_argument('--save', type=str, default='discoveries.json',
                         help='Save discoveries to file')
    p_sweep.add_argument('--metric', type=str, default='score',
                         choices=['score', 'gol_coherence', 'projection', 'structure',
                                  'slice_mi', 'combined', 'period', 'glider',
                                  'growth', 'clusters', 'symmetry', 'elegance',
                                  'gol_like'],
                         help='Metric to optimize (default: score)')

    # search: evolutionary search
    p_search = sub.add_parser('search', help='Evolutionary parameter search')
    p_search.add_argument('rule', type=str, help='Rule name to search')
    p_search.add_argument('--trials', type=int, default=100, help='Number of trials')
    p_search.add_argument('--top', type=int, default=10, help='Top N pool size')
    p_search.add_argument('--save', type=str, default='discoveries.json',
                          help='Save discoveries to file')
    p_search.add_argument('--metric', type=str, default='score',
                          choices=['score', 'gol_coherence', 'projection', 'structure',
                                   'slice_mi', 'combined', 'period', 'glider',
                                   'growth', 'clusters', 'symmetry', 'elegance',
                                   'gol_like'],
                          help='Metric to optimize (default: score)')
    p_search.add_argument('--min_quality', type=float, default=0.25,
                          help='Minimum score to save a discovery (default: 0.25)')

    # explore: search + auto-launch simulator on best result
    p_explore = sub.add_parser('explore', help='Search then launch simulator on best result')
    p_explore.add_argument('rule', type=str, help='Rule name to explore')
    p_explore.add_argument('--trials', type=int, default=50, help='Number of search trials')
    p_explore.add_argument('--top', type=int, default=5, help='Top N to save')
    p_explore.add_argument('--save', type=str, default='discoveries.json',
                           help='Save discoveries to file')
    p_explore.add_argument('--metric', type=str, default='combined',
                           choices=['score', 'gol_coherence', 'projection', 'structure',
                                    'slice_mi', 'combined', 'period', 'glider',
                                    'growth', 'clusters', 'symmetry', 'elegance',
                                    'gol_like'],
                           help='Metric to optimize (default: combined)')
    p_explore.add_argument('--view-size', type=int, default=64,
                           help='Grid size for viewing (default: 64)')

    args = parser.parse_args()

    if not args.command:
        args.command = 'audit'

    window, ctx = create_headless_context()

    try:
        if args.command == 'audit':
            cmd_audit(ctx, args)
        elif args.command == 'test':
            cmd_test(ctx, args)
        elif args.command == 'sweep':
            cmd_sweep(ctx, args)
        elif args.command == 'search':
            cmd_search(ctx, args)
        elif args.command == 'explore':
            cmd_explore(ctx, args)
    finally:
        destroy_context(window)


if __name__ == '__main__':
    main()
