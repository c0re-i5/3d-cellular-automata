#!/usr/bin/env python3
"""
3D Cellular Automata Simulator — GPU-accelerated.

Real-time 3D CA simulation using OpenGL compute shaders,
volumetric ray marching for rendering, and imgui for controls.

Uses rgba32f textures (4 floats per cell) for proper multi-field
simulations: wave equation (displacement + velocity), Gray-Scott
reaction-diffusion (U + V species), and single-field CAs.
Falls back to rgba16f for grids whose textures exceed the driver's
per-allocation limit (~1 GiB on Mesa/Nouveau).

Requires: moderngl, glfw, numpy, imgui-bundle, PyOpenGL

Usage:
    python3 simulator.py
    python3 simulator.py --size 128 --rule game_of_life_3d
    python3 simulator.py --size 64 --rule reaction_diffusion_3d
"""

import sys, math, time, argparse, json, os, re, random, subprocess, shutil, threading
import queue as _queue
import numpy as np
import glfw
import moderngl
from OpenGL import GL
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
from element_data import ELEMENT_GPU_DATA, SYMBOLS, NAMES, NUM_ELEMENTS, FLOATS_PER_ELEMENT, WALL_ID

# Per-allocation size limit for a single rgba32f 3D texture, above which
# the grid falls back to rgba16f. The default is conservative (1 GiB) so
# the simulator runs on older drivers (Mesa/Nouveau historically capped
# single allocations at ~1 GiB); `_probe_tex_alloc_limit()` below raises
# it at GL-init time on drivers that support queryable VRAM (NVIDIA NVX),
# since we ping-pong two simulation textures plus render targets and want
# roughly a third of VRAM per texture.
_TEX_ALLOC_LIMIT = 1_000_000_000  # bytes, default — see probe below


def _probe_tex_alloc_limit(ctx):
    """Return a per-texture byte budget for this GL context.

    On NVIDIA we query GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX
    (extension GL_NVX_gpu_memory_info). We reserve ~1 GB for render
    targets / metrics buffers / driver overhead, divide the rest by 4
    (two sim textures + headroom for resize operations + staging
    buffers), and allow the remaining budget per allocation. On other
    drivers we keep the conservative 1 GiB default.
    """
    try:
        from OpenGL import GL as _GL
        renderer = (ctx.info.get('GL_RENDERER', '') or '').lower()
        vendor   = (ctx.info.get('GL_VENDOR',   '') or '').lower()
        is_nvidia = 'nvidia' in vendor or 'nvidia' in renderer
        if is_nvidia:
            total_kb = _GL.glGetIntegerv(0x9048)          # NVX_gpu_memory_info
            if total_kb and total_kb > 0:
                total_bytes = int(total_kb) * 1024
                # Reserve 1 GB for non-sim GL objects, then divide remainder by 4
                budget = max(1_000_000_000, (total_bytes - 1_000_000_000) // 4)
                return min(budget, 8_000_000_000)          # sanity cap at 8 GB
    except Exception:
        pass
    return 1_000_000_000

# Boundary modes — must match the integer values used in shader fetch():
#   0 = toroidal  (periodic wrap)
#   1 = clamped   (Dirichlet — out-of-bounds reads as zero)
#   2 = mirror    (Neumann zero-flux — reflect at the wall; preserves mass for
#                  reaction/diffusion PDEs and avoids spurious boundary layers)
_BOUNDARY_NAME_TO_MODE = {
    'toroidal': 0, 'periodic': 0, 'wrap': 0,
    'clamped': 1, 'dirichlet': 1, 'zero': 1,
    'mirror': 2, 'neumann': 2, 'reflect': 2, 'zero_flux': 2,
}


def _tex_format_for_size(size):
    """Return (moderngl_dtype, numpy_dtype, bytes_per_texel, glsl_format) for a grid of `size`."""
    if size ** 3 * 16 > _TEX_ALLOC_LIMIT:
        return 'f2', np.float16, 8, 'rgba16f'
    return 'f4', np.float32, 16, 'rgba32f'

# ── Shader sources ────────────────────────────────────────────────────

# Compute shader: 3D CA step
# Uses rgba32f textures (4 floats per cell) for multi-field support
COMPUTE_HEADER = """
#version 430
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

layout(rgba32f, binding=0) uniform image3D u_src;
layout(rgba32f, binding=1) uniform image3D u_dst;

uniform int u_size;
uniform float u_dt;
uniform float u_param0;
uniform float u_param1;
uniform float u_param2;
uniform float u_param3;
uniform int u_boundary;  // 0 = toroidal (wrap), 1 = clamped (Dirichlet, zero outside), 2 = mirror (Neumann, zero-flux)
uniform int u_frame;     // step counter for temporal noise

// ── Grid-spacing scale factors ──────────────────────────────────────
// PDE rules use discrete Laplacians with h=1 (voxel spacing).
// The actual Laplacian is (sum - 6*center) / h², where h = REF/size.
// h_sq normalizes so parameters tuned at REF_SIZE work at any resolution.
// h_inv = size/REF for scaling radii and gradients.
const float REF_SIZE = 128.0;
float h_sq  = (REF_SIZE / float(u_size)) * (REF_SIZE / float(u_size));  // multiply Laplacians by this
float h_inv = float(u_size) / REF_SIZE;  // multiply radii by this

vec4 fetch(ivec3 p) {
    if (u_boundary == 1) {
        // Clamped (Dirichlet): out-of-bounds returns zero
        if (any(lessThan(p, ivec3(0))) || any(greaterThanEqual(p, ivec3(u_size))))
            return vec4(0.0);
        return imageLoad(u_src, p);
    }
    if (u_boundary == 2) {
        // Mirror (Neumann zero-flux): reflect index across boundary.
        // Preserves conservation laws for diffusion/reaction-diffusion PDEs.
        // Equivalent to the boundary cell having a phantom neighbor equal to itself.
        p = clamp(p, ivec3(0), ivec3(u_size - 1));
        return imageLoad(u_src, p);
    }
    // Toroidal: wrap around
    p = (p + u_size) % u_size;
    return imageLoad(u_src, p);
}

// Temporal hash: changes every frame (unlike position-only fract(sin(...)))
// Uses Wang hash on (pos + frame) for high-quality pseudo-random numbers
float hash_temporal(ivec3 p, int channel) {
    uint seed = uint(p.x * 73856093) ^ uint(p.y * 19349663) ^ uint(p.z * 83492791)
              ^ uint(u_frame * 2654435761u) ^ uint(channel * 668265263u);
    seed = (seed ^ (seed >> 16u)) * 0x45d9f3bu;
    seed = (seed ^ (seed >> 16u)) * 0x45d9f3bu;
    seed = seed ^ (seed >> 16u);
    return float(seed) / 4294967295.0;
}

// Trilinear interpolation: fetch at fractional position (for semi-Lagrangian advection)
vec4 fetch_interp(vec3 p) {
    vec3 pf = floor(p);
    vec3 frac_p = p - pf;
    ivec3 p0 = ivec3(pf);

    // Eight corners of the cube
    vec4 c000 = fetch(p0);
    vec4 c100 = fetch(p0 + ivec3(1,0,0));
    vec4 c010 = fetch(p0 + ivec3(0,1,0));
    vec4 c110 = fetch(p0 + ivec3(1,1,0));
    vec4 c001 = fetch(p0 + ivec3(0,0,1));
    vec4 c101 = fetch(p0 + ivec3(1,0,1));
    vec4 c011 = fetch(p0 + ivec3(0,1,1));
    vec4 c111 = fetch(p0 + ivec3(1,1,1));

    // Trilinear blend
    float fx = frac_p.x, fy = frac_p.y, fz = frac_p.z;
    vec4 i0 = mix(mix(c000, c100, fx), mix(c010, c110, fx), fy);
    vec4 i1 = mix(mix(c001, c101, fx), mix(c011, c111, fx), fy);
    return mix(i0, i1, fz);
}
"""

# Different CA rule kernels
CA_RULES = {
    "game_of_life_3d": """
// 3D Game of Life (Moore neighborhood, 26 neighbors)
// Birth: b1-b2 neighbors alive, Survival: s1-s2 neighbors alive
//
// USE_SHARED_MEM=1 loads a 10^3 float tile (just the .r channel — the
// other three are never read here) so the 26 stencil reads come from
// on-chip shared memory. Biggest potential win among the Moore rules
// because the kernel is almost pure counting.

#if USE_SHARED_MEM
#define GLTILE 10
#define GLTILE3 (GLTILE * GLTILE * GLTILE)
shared float s_gl[GLTILE3];
int gl_idx(int x, int y, int z) {
    return z * GLTILE * GLTILE + y * GLTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < GLTILE3; i += 512) {
        int tz = i / (GLTILE * GLTILE);
        int ty = (i / GLTILE) % GLTILE;
        int tx = i % GLTILE;
        s_gl[i] = fetch(tile_origin + ivec3(tx, ty, tz)).r;
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    float self = s_gl[gl_idx(tp.x, tp.y, tp.z)];
    int alive = 0;

    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        if (s_gl[gl_idx(tp.x + dx, tp.y + dy, tp.z + dz)] > 0.5) alive++;
    }
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    float self = fetch(pos).r;
    int alive = 0;

    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        if (fetch(pos + ivec3(dx, dy, dz)).r > 0.5) alive++;
    }
#endif

    float result;
    int b1 = int(u_param0), b2 = int(u_param1);
    int s1 = int(u_param2), s2 = int(u_param3);

    if (self < 0.5) {
        result = (alive >= b1 && alive <= b2) ? 1.0 : 0.0;
    } else {
        result = (alive >= s1 && alive <= s2) ? 1.0 : 0.0;
    }

    imageStore(u_dst, pos, vec4(result, 0.0, 0.0, 0.0));
}
""",

    "smoothlife_3d": """
// 3D SmoothLife — continuous states, smooth transitions
// Uses Rafler's SmoothLife transition with birth/survival intervals.
//
// Two code paths: USE_SHARED_MEM=1 uses cooperative shared memory tiling
// for ~100x fewer global memory ops; USE_SHARED_MEM=0 uses direct
// imageLoad (compatible with nouveau and other limited drivers).
//
// Kernel radii are in *reference voxels* (1.5/2.5 at REF_SIZE=128) so the
// physical feature size stays consistent across resolutions. This scales
// with h_inv = size/REF_SIZE. At size 384+ the scan would exceed the
// shared-mem tile; MAX_SCAN caps the scan AND shrinks outer_r/inner_r
// proportionally in *both* code paths so dynamics remain identical
// between paths (even though the effective kernel becomes smaller than
// theoretical above size ~256). See commit log for details.

#define MAX_SCAN 5

#if USE_SHARED_MEM
#define TILE (8 + 2 * MAX_SCAN)
#define TILE3 (TILE * TILE * TILE)
shared float s_tile[TILE3];
int tile_idx(int x, int y, int z) {
    return z * TILE * TILE + y * TILE + x;
}
#endif

float smooth_sigmoid(float x, float center, float width) {
    return 1.0 / (1.0 + exp(-(x - center) / max(width, 0.001)));
}

float smooth_interval(float x, float lo, float hi, float width) {
    return smooth_sigmoid(x, lo, width) * (1.0 - smooth_sigmoid(x, hi, width));
}

float lerp_thresh(float alive, float birth_val, float survive_val) {
    return birth_val * (1.0 - alive) + survive_val * alive;
}

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

    float inner_r = 1.5 * h_inv;
    float outer_r = 2.5 * h_inv;
    int scan = int(ceil(outer_r));
    // Shared cap — applied identically in tiled and direct paths.
    if (scan > MAX_SCAN) {
        float shrink = float(MAX_SCAN) / outer_r;
        outer_r *= shrink;
        inner_r *= shrink;
        scan = MAX_SCAN;
    }

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 group_origin = ivec3(gl_WorkGroupID) * 8;

    ivec3 tile_origin = group_origin - ivec3(scan);
    for (int i = local_flat; i < TILE3; i += 512) {
        int tz = i / (TILE * TILE);
        int ty = (i / TILE) % TILE;
        int tx = i % TILE;
        s_tile[i] = fetch(tile_origin + ivec3(tx, ty, tz)).r;
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tile_pos = local + ivec3(scan);
    float self = s_tile[tile_idx(tile_pos.x, tile_pos.y, tile_pos.z)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    float self = fetch(pos).r;
#endif

    float inner_sum = 0.0, outer_sum = 0.0;
    int inner_count = 0, outer_count = 0;
    float inner_r2 = inner_r * inner_r;
    float outer_r2 = outer_r * outer_r;

    for (int dz = -scan; dz <= scan; dz++)
    for (int dy = -scan; dy <= scan; dy++)
    for (int dx = -scan; dx <= scan; dx++) {
        float dist2 = float(dx*dx + dy*dy + dz*dz);
        if (dist2 < 0.25) continue;
        if (dist2 > outer_r2) continue;
#if USE_SHARED_MEM
        float v = s_tile[tile_idx(tile_pos.x + dx, tile_pos.y + dy, tile_pos.z + dz)];
#else
        float v = fetch(pos + ivec3(dx, dy, dz)).r;
#endif
        if (dist2 <= inner_r2) {
            inner_sum += v; inner_count++;
        } else {
            outer_sum += v; outer_count++;
        }
    }

    float m = inner_count > 0 ? inner_sum / float(inner_count) : 0.0;
    float n = outer_count > 0 ? outer_sum / float(outer_count) : 0.0;

    float b_center = u_param0;
    float b_range = u_param1;
    float s_center = u_param2;
    float s_range = u_param3;

    float b_lo = b_center - b_range;
    float b_hi = b_center + b_range;
    float s_lo = s_center - s_range;
    float s_hi = s_center + s_range;

    float sigma_width = 0.03;
    float lo = lerp_thresh(m, b_lo, s_lo);
    float hi = lerp_thresh(m, b_hi, s_hi);
    float growth = smooth_interval(n, lo, hi, sigma_width);

    float result = self + u_dt * (2.0 * growth - 1.0);
    result = clamp(result, 0.0, 1.0);

    imageStore(u_dst, pos, vec4(result, 0.0, 0.0, 0.0));
}
""",

    "reaction_diffusion_3d": """
// 3D Gray-Scott Reaction-Diffusion — two coupled species
// Channel R = concentration U (substrate), Channel G = concentration V (catalyst)
//   dU/dt = Du * lap(U) - U*V^2 + F*(1-U)
//   dV/dt = Dv * lap(V) + U*V^2 - (F+k)*V
//
// USE_SHARED_MEM=1 cooperatively loads a 10^3 vec2 tile (RG only — the
// A,B channels aren't used here) into shared memory so the 6 stencil
// reads come from on-chip storage instead of 6 imageLoad ops per cell.
// Tile cost: 10*10*10 * 8 bytes = 8000 bytes shared per workgroup.

#if USE_SHARED_MEM
#define RDTILE 10
#define RDTILE3 (RDTILE * RDTILE * RDTILE)
shared vec2 s_uv[RDTILE3];
int rd_idx(int x, int y, int z) {
    return z * RDTILE * RDTILE + y * RDTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    // Cooperative tile load — every thread participates BEFORE any
    // early exit so the barrier doesn't deadlock and halo entries are
    // populated even when the workgroup straddles the grid boundary.
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < RDTILE3; i += 512) {
        int tz = i / (RDTILE * RDTILE);
        int ty = (i / RDTILE) % RDTILE;
        int tx = i % RDTILE;
        s_uv[i] = fetch(tile_origin + ivec3(tx, ty, tz)).rg;
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec2 c = s_uv[rd_idx(tp.x, tp.y, tp.z)];
    float U = c.r;
    float V = c.g;

    vec2 sum = vec2(0.0);
    sum += s_uv[rd_idx(tp.x + 1, tp.y,     tp.z    )];
    sum += s_uv[rd_idx(tp.x - 1, tp.y,     tp.z    )];
    sum += s_uv[rd_idx(tp.x,     tp.y + 1, tp.z    )];
    sum += s_uv[rd_idx(tp.x,     tp.y - 1, tp.z    )];
    sum += s_uv[rd_idx(tp.x,     tp.y,     tp.z + 1)];
    sum += s_uv[rd_idx(tp.x,     tp.y,     tp.z - 1)];
    float sum_U = sum.x;
    float sum_V = sum.y;
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_val = fetch(pos);
    float U = self_val.r;
    float V = self_val.g;

    // Sum 6 von Neumann neighbors for both species simultaneously
    float sum_U = 0.0, sum_V = 0.0;
    vec4 nb;
    nb = fetch(pos + ivec3( 1, 0, 0)); sum_U += nb.r; sum_V += nb.g;
    nb = fetch(pos + ivec3(-1, 0, 0)); sum_U += nb.r; sum_V += nb.g;
    nb = fetch(pos + ivec3( 0, 1, 0)); sum_U += nb.r; sum_V += nb.g;
    nb = fetch(pos + ivec3( 0,-1, 0)); sum_U += nb.r; sum_V += nb.g;
    nb = fetch(pos + ivec3( 0, 0, 1)); sum_U += nb.r; sum_V += nb.g;
    nb = fetch(pos + ivec3( 0, 0,-1)); sum_U += nb.r; sum_V += nb.g;
#endif

    // Standard 3D discrete Laplacian, scaled by h^2 for resolution-independence
    float lap_U = (sum_U - 6.0 * U) * h_sq;
    float lap_V = (sum_V - 6.0 * V) * h_sq;

    float F  = u_param0;  // feed rate
    float k  = u_param1;  // kill rate
    float Du = u_param2;  // U diffusion rate
    float Dv = u_param3;  // V diffusion rate

    float uvv = U * V * V;
    float dU = Du * lap_U - uvv + F * (1.0 - U);
    float dV = Dv * lap_V + uvv - (F + k) * V;

    float new_U = clamp(U + dU * u_dt, 0.0, 1.0);
    float new_V = clamp(V + dV * u_dt, 0.0, 1.0);

    imageStore(u_dst, pos, vec4(new_U, new_V, 0.0, 0.0));
}
""",

    "wave_3d": """
// 3D Wave Equation — proper second-order PDE with optional point source driving
// Channel R = displacement u, Channel G = velocity v = du/dt
//   dv/dt = c^2 * lap(u) - damping * v + source_driving
//   du/dt = v
// Integration: symplectic Euler (update v first, then u with new v)
//
// Stability (CFL): explicit 3D wave eq. requires  c * u_dt * h_inv <= 1/sqrt(3).
// With the default h_inv = size/REF_SIZE (REF_SIZE=128) and u_dt ≈ 0.5, the
// rule is stable for c <~ 0.58 / h_inv. The preset uses c=0.5 which is safely
// inside this bound at all supported grid sizes; higher c values will ring.
//
// Two code paths: USE_SHARED_MEM=1 cooperatively loads a 10^3 tile
// (8^3 core + 1-voxel halo) into shared memory so the 7 stencil reads
// per cell come from on-chip memory instead of 7 imageLoad ops.
// USE_SHARED_MEM=0 keeps the original direct-fetch path for nouveau.
// Only the displacement channel (R) needs to be tiled — velocity is
// per-cell and read from the self_val sample.

#if USE_SHARED_MEM
#define WTILE 10                  // 8 + 2*1 halo
#define WTILE3 (WTILE * WTILE * WTILE)
shared float s_u[WTILE3];
int wtile_idx(int x, int y, int z) {
    return z * WTILE * WTILE + y * WTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    // Cooperative tile load. Threads outside the simulation domain still
    // participate in the load (and contribute boundary values via fetch())
    // before any early exit — otherwise the barrier would deadlock and
    // the halo would have undefined entries.
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 group_origin = ivec3(gl_WorkGroupID) * 8;
    ivec3 tile_origin = group_origin - ivec3(1);
    for (int i = local_flat; i < WTILE3; i += 512) {
        int tz = i / (WTILE * WTILE);
        int ty = (i / WTILE) % WTILE;
        int tx = i % WTILE;
        s_u[i] = fetch(tile_origin + ivec3(tx, ty, tz)).r;
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);  // shift past halo
    vec4 self_val = fetch(pos);   // need both .r and .g; .g not in tile
    float u = self_val.r;
    float v = self_val.g;

    float sum_u = 0.0;
    sum_u += s_u[wtile_idx(tp.x + 1, tp.y,     tp.z    )];
    sum_u += s_u[wtile_idx(tp.x - 1, tp.y,     tp.z    )];
    sum_u += s_u[wtile_idx(tp.x,     tp.y + 1, tp.z    )];
    sum_u += s_u[wtile_idx(tp.x,     tp.y - 1, tp.z    )];
    sum_u += s_u[wtile_idx(tp.x,     tp.y,     tp.z + 1)];
    sum_u += s_u[wtile_idx(tp.x,     tp.y,     tp.z - 1)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    vec4 self_val = fetch(pos);
    float u = self_val.r;  // displacement
    float v = self_val.g;  // velocity

    // 3D discrete Laplacian, scaled by h^2 for resolution-independence
    float sum_u = 0.0;
    sum_u += fetch(pos + ivec3( 1, 0, 0)).r;
    sum_u += fetch(pos + ivec3(-1, 0, 0)).r;
    sum_u += fetch(pos + ivec3( 0, 1, 0)).r;
    sum_u += fetch(pos + ivec3( 0,-1, 0)).r;
    sum_u += fetch(pos + ivec3( 0, 0, 1)).r;
    sum_u += fetch(pos + ivec3( 0, 0,-1)).r;
#endif
    float lap = (sum_u - 6.0 * u) * h_sq;

    float c         = u_param0;  // wave speed
    float damping   = u_param1;  // damping coefficient
    float drive_freq = u_param2; // driving frequency (0 = no driving)
    float drive_amp  = u_param3; // driving amplitude

    // Point source driving at grid center: sin(freq * time) * amplitude
    // Creates sustained resonance patterns instead of decaying initial pulse
    float driving = 0.0;
    int mid = int(u_size) / 2;
    if (drive_amp > 0.001 && drive_freq > 0.001) {
        int dx = pos.x - mid;
        int dy = pos.y - mid;
        int dz = pos.z - mid;
        float dist2 = float(dx*dx + dy*dy + dz*dz);
        float src_r = 2.0 * h_inv;  // source radius scales with grid
        if (dist2 < src_r * src_r) {
            driving = drive_amp * sin(drive_freq * float(u_frame) * u_dt);
        }
    }

    // Symplectic Euler: update velocity first, then position with new velocity
    float new_v = v + (c * c * lap - damping * v + driving) * u_dt;
    float new_u = u + new_v * u_dt;

    // Generous clamp to prevent float overflow (not [0,1] — waves are signed)
    new_u = clamp(new_u, -100.0, 100.0);
    new_v = clamp(new_v, -100.0, 100.0);

    imageStore(u_dst, pos, vec4(new_u, new_v, 0.0, 0.0));
}
""",

    "crystal_growth": """
// Phase-field crystal growth with anisotropic surface energy
// Kobayashi-type model: cubic harmonics anisotropy on the interface normal.
// R = phase field φ [0=liquid, 1=solid]
// G = supersaturation u (dimensionless undercooling, drives growth)
// u_param0 = undercooling (drives growth speed)
// u_param1 = diffusion rate (thermal diffusivity)
// u_param2 = anisotropy strength ε (cubic harmonic amplitude)
// u_param3 = mode: 0=compact crystal, 1=dendritic (noise-driven branching)
//
// USE_SHARED_MEM=1 cooperatively loads a 10^3 vec2 tile of (phi, u_field)
// so the 6 stencil reads (for both lap_phi+lap_u and the phi-gradient)
// come from on-chip shared memory instead of 6 imageLoad ops per cell.

#if USE_SHARED_MEM
#define CGTILE 10
#define CGTILE3 (CGTILE * CGTILE * CGTILE)
shared vec2 s_pu[CGTILE3];
int cg_idx(int x, int y, int z) {
    return z * CGTILE * CGTILE + y * CGTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    // Cooperative tile load. Every thread participates BEFORE any early
    // exit so the barrier can't deadlock and halo entries are populated
    // even when the workgroup straddles the grid boundary.
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < CGTILE3; i += 512) {
        int tz = i / (CGTILE * CGTILE);
        int ty = (i / CGTILE) % CGTILE;
        int tx = i % CGTILE;
        s_pu[i] = fetch(tile_origin + ivec3(tx, ty, tz)).rg;
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec2 c = s_pu[cg_idx(tp.x, tp.y, tp.z)];
    float phi     = c.r;
    float u_field = c.g;

    vec2 nxp = s_pu[cg_idx(tp.x + 1, tp.y,     tp.z    )];
    vec2 nxm = s_pu[cg_idx(tp.x - 1, tp.y,     tp.z    )];
    vec2 nyp = s_pu[cg_idx(tp.x,     tp.y + 1, tp.z    )];
    vec2 nym = s_pu[cg_idx(tp.x,     tp.y - 1, tp.z    )];
    vec2 nzp = s_pu[cg_idx(tp.x,     tp.y,     tp.z + 1)];
    vec2 nzm = s_pu[cg_idx(tp.x,     tp.y,     tp.z - 1)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_data = fetch(pos);
    float phi = self_data.r;      // phase: 0=liquid, 1=solid
    float u_field = self_data.g;  // supersaturation

    vec2 nxp = fetch(pos + ivec3( 1, 0, 0)).rg;
    vec2 nxm = fetch(pos + ivec3(-1, 0, 0)).rg;
    vec2 nyp = fetch(pos + ivec3( 0, 1, 0)).rg;
    vec2 nym = fetch(pos + ivec3( 0,-1, 0)).rg;
    vec2 nzp = fetch(pos + ivec3( 0, 0, 1)).rg;
    vec2 nzm = fetch(pos + ivec3( 0, 0,-1)).rg;
#endif

    float undercooling = u_param0;
    float D = u_param1;
    float eps_strength = u_param2;
    float mode = u_param3;

    float lap_phi = (nxp.r + nxm.r + nyp.r + nym.r + nzp.r + nzm.r - 6.0 * phi)     * h_sq;
    float lap_u   = (nxp.g + nxm.g + nyp.g + nym.g + nzp.g + nzm.g - 6.0 * u_field) * h_sq;

    // Interface normal from ∇φ (scale gradient by h_inv for correct magnitude)
    float gx = (nxp.r - nxm.r) * 0.5 * h_inv;
    float gy = (nyp.r - nym.r) * 0.5 * h_inv;
    float gz = (nzp.r - nzm.r) * 0.5 * h_inv;
    float grad_mag = sqrt(gx*gx + gy*gy + gz*gz + 1e-8);
    vec3 n = vec3(gx, gy, gz) / grad_mag;

    // Cubic harmonics anisotropy: β(n̂) = 1 + ε(nx⁴+ny⁴+nz⁴ - 3/5)
    float nx2 = n.x*n.x, ny2 = n.y*n.y, nz2 = n.z*n.z;
    float beta = 1.0 + eps_strength * (nx2*nx2 + ny2*ny2 + nz2*nz2 - 0.6);

    if (mode > 0.5) {
        // Dendritic: add interface noise for Mullins-Sekerka tip-splitting.
        // Must be *temporal* — a static per-cell hash biases each interface
        // cell in the same direction every step, suppressing tip-splitting.
        float noise = hash_temporal(pos, 1);
        beta += eps_strength * 0.5 * (noise - 0.5) * smoothstep(0.01, 0.2, grad_mag);
    }
    beta = max(beta, 0.01);

    // Double-well driving + capillary anisotropy
    float driving = phi * (1.0 - phi) * (phi - 0.5 + undercooling + u_field * 0.5);
    float b2 = beta * beta;
    float dphi = b2 * lap_phi + b2 * driving * 30.0;
    float new_phi = clamp(phi + dphi * u_dt, 0.0, 1.0);

    // Supersaturation: diffuses + depleted by solidification (latent heat)
    float du = D * lap_u - (new_phi - phi) * 0.5 / max(u_dt, 0.001);
    float new_u = u_field + du * u_dt;

    // Boundary feed: replenish supersaturation at edges
    if (u_boundary == 1) {
        if (pos.x == 0 || pos.x == u_size-1 ||
            pos.y == 0 || pos.y == u_size-1 ||
            pos.z == 0 || pos.z == u_size-1) {
            new_u = mix(new_u, undercooling, 0.05 * u_dt);
        }
    }

    new_u = clamp(new_u, -1.0, 2.0);
    imageStore(u_dst, pos, vec4(new_phi, new_u, 0.0, 0.0));
}
""",

    "lenia_3d": """
// 3D Lenia — continuous cellular automata with kernel-based growth
// u_param0 = growth center mu, u_param1 = growth width sigma
// u_param2 = kernel radius R, u_param3 = kernel ring position [0,1]
//
// Two code paths: USE_SHARED_MEM=1 uses shared memory tiling,
// USE_SHARED_MEM=0 uses direct imageLoad for driver compatibility.
//
// MAX_R caps both code paths identically so dynamics match regardless
// of shared-mem availability; at size 512 (h_inv=4) a radius=2 preset
// saturates at R=7 rather than 8.

#define MAX_R 7

#if USE_SHARED_MEM
#define TILE (8 + 2 * MAX_R)
#define TILE3 (TILE * TILE * TILE)
shared float s_tile[TILE3];
int tile_idx(int x, int y, int z) {
    return z * TILE * TILE + y * TILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

    float mu = u_param0;
    float sigma = u_param1;
    float radius = u_param2;
    float ring_pos = clamp(u_param3, 0.1, 0.9);
    if (ring_pos < 0.05) ring_pos = 0.5;

    // Shared cap — applied identically in both paths so dynamics match.
    // At size ≥ 4x REF_SIZE (radius*h_inv > MAX_R) the kernel saturates
    // at R = MAX_R voxels. See commit log.
    int R = min(int(radius * h_inv), MAX_R);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 group_origin = ivec3(gl_WorkGroupID) * 8;

    ivec3 tile_origin = group_origin - ivec3(R);
    int tile_size = 8 + 2 * R;
    int tile_count = tile_size * tile_size * tile_size;
    for (int i = local_flat; i < tile_count; i += 512) {
        int tz = i / (tile_size * tile_size);
        int ty = (i / tile_size) % tile_size;
        int tx = i % tile_size;
        int store_x = tx + (MAX_R - R);
        int store_y = ty + (MAX_R - R);
        int store_z = tz + (MAX_R - R);
        s_tile[tile_idx(store_x, store_y, store_z)] = fetch(tile_origin + ivec3(tx, ty, tz)).r;
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tile_pos = local + ivec3(MAX_R);
    float self = s_tile[tile_idx(tile_pos.x, tile_pos.y, tile_pos.z)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    float self = fetch(pos).r;
#endif

    float weighted_sum = 0.0;
    float weight_total = 0.0;
    float fR = float(R);

    for (int dz = -R; dz <= R; dz++)
    for (int dy = -R; dy <= R; dy++)
    for (int dx = -R; dx <= R; dx++) {
        float dist2 = float(dx*dx + dy*dy + dz*dz);
        if (dist2 < 0.25) continue;
        float dist = sqrt(dist2);
        if (dist > fR + 0.5) continue;

        float r_norm = dist / fR;
        float kring = (r_norm - ring_pos) / 0.15;
        float kernel = exp(-0.5 * kring * kring);

#if USE_SHARED_MEM
        float v = s_tile[tile_idx(tile_pos.x + dx, tile_pos.y + dy, tile_pos.z + dz)];
#else
        float v = fetch(pos + ivec3(dx, dy, dz)).r;
#endif
        weighted_sum += v * kernel;
        weight_total += kernel;
    }

    float potential = weight_total > 0.0 ? weighted_sum / weight_total : 0.0;
    // Growth function: 2*exp(-0.5*z^2) - 1 with z = (potential - mu)/sigma.
    // Hand-squaring the z-score avoids pow(x, 2.0); most GLSL drivers expand
    // pow into exp2(log2(x)*2), which is ~10x slower than a single multiply
    // and inside Lenia's hot loop this lands every cell every step.
    float gz = (potential - mu) / max(sigma, 0.001);
    float growth = 2.0 * exp(-0.5 * gz * gz) - 1.0;

    float result = self + u_dt * growth;
    result = clamp(result, 0.0, 1.0);

    imageStore(u_dst, pos, vec4(result, 0.0, 0.0, 0.0));
}
""",

    "lenia_multi_3d": """
// Multi-channel Lenia — 3 channels with cross-channel kernel coupling
// Two code paths: USE_SHARED_MEM=1 uses vec3 shared memory tiling,
// USE_SHARED_MEM=0 uses direct imageLoad for driver compatibility.
//
// MAX_R caps both code paths identically so dynamics match.

#define MAX_R 3

#if USE_SHARED_MEM
#define TILE (8 + 2 * MAX_R)
#define TILE3 (TILE * TILE * TILE)
shared vec3 s_tile[TILE3];
int tile_idx(int x, int y, int z) {
    return z * TILE * TILE + y * TILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

    float mu = u_param0;
    float sigma = max(u_param1, 0.001);
    float radius = u_param2;
    float cross = u_param3;

    int R = min(int(radius * h_inv), MAX_R);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 group_origin = ivec3(gl_WorkGroupID) * 8;

    ivec3 tile_origin = group_origin - ivec3(R);
    int tile_size = 8 + 2 * R;
    int tile_count = tile_size * tile_size * tile_size;
    for (int i = local_flat; i < tile_count; i += 512) {
        int tz = i / (tile_size * tile_size);
        int ty = (i / tile_size) % tile_size;
        int tx = i % tile_size;
        int sx = tx + (MAX_R - R);
        int sy = ty + (MAX_R - R);
        int sz = tz + (MAX_R - R);
        s_tile[tile_idx(sx, sy, sz)] = fetch(tile_origin + ivec3(tx, ty, tz)).rgb;
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(MAX_R);
    vec3 self_rgb = s_tile[tile_idx(tp.x, tp.y, tp.z)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    vec3 self_rgb = fetch(pos).rgb;
    ivec3 tp = pos;  // alias for sampling loop
#endif

    vec3 sum_inner = vec3(0.0), sum_outer = vec3(0.0);
    float w_inner = 0.0, w_outer = 0.0;
    float fR = float(R);

    for (int dz = -R; dz <= R; dz++)
    for (int dy = -R; dy <= R; dy++)
    for (int dx = -R; dx <= R; dx++) {
        float dist2 = float(dx*dx + dy*dy + dz*dz);
        if (dist2 < 0.25) continue;
        float dist = sqrt(dist2);
        if (dist > fR + 0.5) continue;

        float r_norm = dist / fR;
        float ki = (r_norm - 0.3) / 0.12;
        float ko = (r_norm - 0.7) / 0.12;
        float k_inner = exp(-0.5 * ki * ki);
        float k_outer = exp(-0.5 * ko * ko);

#if USE_SHARED_MEM
        vec3 nb = s_tile[tile_idx(tp.x + dx, tp.y + dy, tp.z + dz)];
#else
        vec3 nb = fetch(pos + ivec3(dx, dy, dz)).rgb;
#endif
        sum_inner += nb * k_inner;
        sum_outer += nb * k_outer;
        w_inner += k_inner;
        w_outer += k_outer;
    }

    sum_inner /= max(w_inner, 1.0);
    sum_outer /= max(w_outer, 1.0);

    float pot_a = sum_inner.r + cross * (sum_outer.g + sum_outer.b) * 0.5;
    float pot_b = sum_inner.g + cross * (sum_outer.b + sum_outer.r) * 0.5;
    float pot_c = sum_inner.b + cross * (sum_outer.r + sum_outer.g) * 0.5;

    float mu_offset = 0.1 * cross;
    float ga_z = (pot_a - mu) / sigma;
    float gb_z = (pot_b - mu * (1.0 + mu_offset)) / sigma;
    float gc_z = (pot_c - mu * (1.0 - mu_offset)) / sigma;
    float ga = 2.0 * exp(-0.5 * ga_z * ga_z) - 1.0;
    float gb = 2.0 * exp(-0.5 * gb_z * gb_z) - 1.0;
    float gc = 2.0 * exp(-0.5 * gc_z * gc_z) - 1.0;

    vec3 result = self_rgb + u_dt * vec3(ga, gb, gc);
    result = clamp(result, 0.0, 1.0);

    float activity = (abs(ga) + abs(gb) + abs(gc)) / 3.0;

    imageStore(u_dst, pos, vec4(result, activity));
}
""",

    "predator_prey_3d": """
// Rosenzweig-MacArthur predator-prey with Holling type II functional response
// R = prey density u (logistic growth, consumed by predators)
// G = predator density v (grows by consuming prey, dies naturally)
// B = interaction intensity (visualization: where predation is active)
// Produces: traveling pursuit waves, prey refugia, boom-bust oscillations,
//           spiral waves near Hopf bifurcation
// u_param0 = predation rate a (attack efficiency)
// u_param1 = prey growth rate r (intrinsic reproduction)
// u_param2 = predator mortality d (natural death rate)
// u_param3 = conversion efficiency e (prey biomass → predator biomass)
//            Controls Hopf bifurcation: e < a*h → oscillations, e > a*h → equilibrium
//
// USE_SHARED_MEM=1 loads a 10^3 vec2 tile of (u, v) so the 6 stencil reads
// come from on-chip shared memory instead of 6 imageLoad ops per cell.

#if USE_SHARED_MEM
#define PPTILE 10
#define PPTILE3 (PPTILE * PPTILE * PPTILE)
shared vec2 s_uv[PPTILE3];
int pp_idx(int x, int y, int z) {
    return z * PPTILE * PPTILE + y * PPTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < PPTILE3; i += 512) {
        int tz = i / (PPTILE * PPTILE);
        int ty = (i / PPTILE) % PPTILE;
        int tx = i % PPTILE;
        s_uv[i] = fetch(tile_origin + ivec3(tx, ty, tz)).rg;
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec2 c = s_uv[pp_idx(tp.x, tp.y, tp.z)];
    float u = c.r;
    float v = c.g;

    vec2 sum_uv = vec2(0.0);
    sum_uv += s_uv[pp_idx(tp.x + 1, tp.y,     tp.z    )];
    sum_uv += s_uv[pp_idx(tp.x - 1, tp.y,     tp.z    )];
    sum_uv += s_uv[pp_idx(tp.x,     tp.y + 1, tp.z    )];
    sum_uv += s_uv[pp_idx(tp.x,     tp.y - 1, tp.z    )];
    sum_uv += s_uv[pp_idx(tp.x,     tp.y,     tp.z + 1)];
    sum_uv += s_uv[pp_idx(tp.x,     tp.y,     tp.z - 1)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_val = fetch(pos);
    float u = self_val.r;  // prey density
    float v = self_val.g;  // predator density

    // Laplacian
    vec2 sum_uv = vec2(0.0);
    sum_uv += fetch(pos + ivec3( 1, 0, 0)).rg;
    sum_uv += fetch(pos + ivec3(-1, 0, 0)).rg;
    sum_uv += fetch(pos + ivec3( 0, 1, 0)).rg;
    sum_uv += fetch(pos + ivec3( 0,-1, 0)).rg;
    sum_uv += fetch(pos + ivec3( 0, 0, 1)).rg;
    sum_uv += fetch(pos + ivec3( 0, 0,-1)).rg;
#endif
    float lap_u = (sum_uv.x - 6.0 * u) * h_sq;
    float lap_v = (sum_uv.y - 6.0 * v) * h_sq;

    float a  = u_param0;  // predation rate
    float r  = u_param1;  // prey growth rate
    float d  = u_param2;  // predator mortality
    float e  = u_param3;  // conversion efficiency

    float Du = 0.5;       // prey diffusion (fixed)
    float Dv = 0.15;      // predator diffusion (less mobile)

    float K = 1.0;        // prey carrying capacity
    float h = 0.6;        // handling time (functional response saturation)

    // Holling type II functional response: f(u) = a*u / (1 + a*h*u)
    float func_resp = a * u / (1.0 + a * h * max(u, 0.0));

    // Rosenzweig-MacArthur dynamics:
    // du/dt = r*u*(1 - u/K) - v*f(u) + Du*∇²u
    // dv/dt = e*v*f(u) - d*v + Dv*∇²v
    float du = r * u * (1.0 - u / K) - v * func_resp + Du * lap_u;
    float dv = e * v * func_resp - d * v + Dv * lap_v;

    float new_u = max(u + du * u_dt, 0.0);
    float new_v = max(v + dv * u_dt, 0.0);

    // Prevent runaway (numerical stability)
    new_u = min(new_u, K * 3.0);
    new_v = min(new_v, K * 3.0);

    // Interaction channel: where predation is actively happening
    float interaction = v * func_resp;
    imageStore(u_dst, pos, vec4(new_u, new_v, interaction, 0.0));
}
""",

    "kuramoto_3d": """
// 3D Kuramoto coupled oscillators with adaptive frequencies
// R = phase (0 to 1, representing 0 to 2*pi)
// G = natural frequency (adapts via Hebbian-like learning)
// B = local order parameter (coherence magnitude)
// Cells synchronize with neighbors; frequencies adapt toward synchronized clusters
//
// USE_SHARED_MEM=1 loads a 10^3 vec2 tile of (phase, nat_freq) so the 26
// Moore-neighbourhood stencil reads come from on-chip shared memory.
// This is the biggest win of any tiled rule: 26 imageLoad → 26 shared reads.

#if USE_SHARED_MEM
#define KMTILE 10
#define KMTILE3 (KMTILE * KMTILE * KMTILE)
shared vec2 s_pn[KMTILE3];
int km_idx(int x, int y, int z) {
    return z * KMTILE * KMTILE + y * KMTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < KMTILE3; i += 512) {
        int tz = i / (KMTILE * KMTILE);
        int ty = (i / KMTILE) % KMTILE;
        int tx = i % KMTILE;
        s_pn[i] = fetch(tile_origin + ivec3(tx, ty, tz)).rg;
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec2 sc = s_pn[km_idx(tp.x, tp.y, tp.z)];
    float phase    = sc.r;  // 0..1 representing 0..2*pi
    float nat_freq = sc.g;  // natural frequency (adapts)
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_val = fetch(pos);
    float phase = self_val.r;         // 0..1 representing 0..2*pi
    float nat_freq = self_val.g;      // natural frequency (adapts)
#endif

    float coupling = u_param0;        // coupling strength K
    float noise_amp = u_param1;       // phase noise amplitude
    float freq_scale = u_param2;      // natural frequency multiplier
    float adaptation = u_param3;      // frequency adaptation rate

    // Sum sin/cos(neighbor_phase - my_phase) for coupling + coherence
    float phase_2pi = phase * 6.283185;
    float coupling_sum = 0.0;
    float cos_sum = 0.0;
    float mean_freq = 0.0;
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
#if USE_SHARED_MEM
        vec2 nb = s_pn[km_idx(tp.x + dx, tp.y + dy, tp.z + dz)];
        float nb_phase_2pi = nb.r * 6.283185;
        float dphi = nb_phase_2pi - phase_2pi;
        coupling_sum += sin(dphi);
        cos_sum += cos(dphi);
        mean_freq += nb.g;
#else
        vec4 nb = fetch(pos + ivec3(dx, dy, dz));
        float nb_phase_2pi = nb.r * 6.283185;
        float dphi = nb_phase_2pi - phase_2pi;
        coupling_sum += sin(dphi);
        cos_sum += cos(dphi);
        mean_freq += nb.g;
#endif
    }
    // 26 Moore neighbors
    coupling_sum /= 26.0;
    cos_sum /= 26.0;
    mean_freq /= 26.0;

    // Local order parameter: R = |<e^{i(θ_j - θ_i)}>|
    float coherence = sqrt(coupling_sum * coupling_sum + cos_sum * cos_sum);

    // Temporal noise — different every step (creates ongoing perturbation)
    float hash = hash_temporal(pos, 0);
    float noise = (hash - 0.5) * 2.0 * noise_amp;

    // Phase evolution: d(phase)/dt = omega + K * coupling + noise
    float d_phase = (nat_freq * freq_scale + coupling * coupling_sum + noise) * u_dt;
    float new_phase = fract(phase + d_phase);  // wrap to [0, 1)

    // Frequency adaptation: when neighbors are coherent, pull ω toward <ω_j>
    // Creates chimera states — some clusters synchronize, others drift freely.
    // Clamp natural frequency to a bounded range: the adaptation law is linear
    // and coherence-gated but parameter extremes + long runs can drift nat_freq
    // far outside the texture’s usable range, producing aliased phase updates.
    float new_freq = nat_freq + adaptation * coherence * (mean_freq - nat_freq) * u_dt;
    new_freq = clamp(new_freq, -2.0, 2.0);

    imageStore(u_dst, pos, vec4(new_phase, new_freq, coherence, 0.0));
}
""",

    "bz_3d": """
// 3D Belousov-Zhabotinsky via Complex Ginzburg-Landau equation (CGLE)
// The CGLE is the normal form of oscillating chemical reactions (like BZ)
// near a Hopf bifurcation. R = Re(A), G = Im(A) where A is complex amplitude.
// Produces spiral defect chaos and scroll wave turbulence in 3D.
//
// USE_SHARED_MEM=1 loads a 10^3 vec2 tile of (u, v) so the 6 stencil reads
// for the coupled Laplacian come from on-chip shared memory.

#if USE_SHARED_MEM
#define BZTILE 10
#define BZTILE3 (BZTILE * BZTILE * BZTILE)
shared vec2 s_uv[BZTILE3];
int bz_idx(int x, int y, int z) {
    return z * BZTILE * BZTILE + y * BZTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < BZTILE3; i += 512) {
        int tz = i / (BZTILE * BZTILE);
        int ty = (i / BZTILE) % BZTILE;
        int tx = i % BZTILE;
        s_uv[i] = fetch(tile_origin + ivec3(tx, ty, tz)).rg;
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec2 c = s_uv[bz_idx(tp.x, tp.y, tp.z)];
    float u = c.r;  // Re(A)
    float v = c.g;  // Im(A)

    vec2 sum_uv = vec2(0.0);
    sum_uv += s_uv[bz_idx(tp.x + 1, tp.y,     tp.z    )];
    sum_uv += s_uv[bz_idx(tp.x - 1, tp.y,     tp.z    )];
    sum_uv += s_uv[bz_idx(tp.x,     tp.y + 1, tp.z    )];
    sum_uv += s_uv[bz_idx(tp.x,     tp.y - 1, tp.z    )];
    sum_uv += s_uv[bz_idx(tp.x,     tp.y,     tp.z + 1)];
    sum_uv += s_uv[bz_idx(tp.x,     tp.y,     tp.z - 1)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_val = fetch(pos);
    float u = self_val.r;  // Re(A)
    float v = self_val.g;  // Im(A)

    // Laplacian of both components (von Neumann 6-neighbor)
    vec2 sum_uv = vec2(0.0);
    sum_uv += fetch(pos + ivec3( 1, 0, 0)).rg;
    sum_uv += fetch(pos + ivec3(-1, 0, 0)).rg;
    sum_uv += fetch(pos + ivec3( 0, 1, 0)).rg;
    sum_uv += fetch(pos + ivec3( 0,-1, 0)).rg;
    sum_uv += fetch(pos + ivec3( 0, 0, 1)).rg;
    sum_uv += fetch(pos + ivec3( 0, 0,-1)).rg;
#endif
    float lap_u = (sum_uv.x - 6.0 * u) * h_sq;
    float lap_v = (sum_uv.y - 6.0 * v) * h_sq;

    float alpha = u_param0;  // dispersion (linear cross-diffusion)
    float beta  = u_param1;  // nonlinear frequency shift
    float D     = u_param2;  // diffusion coefficient
    float mu    = u_param3;  // growth rate (bifurcation parameter, usually ~1)

    float rho2 = u * u + v * v;

    // CGLE: dA/dt = mu*A + (1+i*alpha)*D*lap(A) - (1+i*beta)*|A|^2*A
    // Reaction: linear growth + cubic saturation
    float du_react = mu * u - (u - beta * v) * rho2;
    float dv_react = mu * v - (v + beta * u) * rho2;

    // Diffusion: cross-coupled due to dispersion alpha
    float du_diff = D * (lap_u - alpha * lap_v);
    float dv_diff = D * (alpha * lap_u + lap_v);

    float new_u = u + (du_react + du_diff) * u_dt;
    float new_v = v + (dv_react + dv_diff) * u_dt;

    // Store phase in B channel for visualization: atan2(v,u)/2pi mapped to [0,1]
    float phase = atan(new_v, new_u) / 6.283185 + 0.5;

    imageStore(u_dst, pos, vec4(new_u, new_v, phase, 0.0));
}
""",

    "barkley_3d": """
// Barkley excitable medium — propagating waves with refractory period
// Models BZ reaction, cardiac tissue, neural excitation, forest fire waves.
// u = fast activator (excitation front) [0,1]
// v = slow inhibitor (recovery/refractory) [0,~1]
// u is bistable with threshold controlled by v: sharp wavefronts
// v tracks u slowly: creates refractory tail behind the wave
// Stochastic nucleation creates sporadic new wavefronts (realistic BZ)
// u_param0 = a (excitability: higher = easier to excite)
// u_param1 = b (threshold shift: controls spiral tip meander)
// u_param2 = epsilon (timescale ratio: smaller = sharper waves)
// u_param3 = D_u (diffusion of activator)
//
// USE_SHARED_MEM=1 loads a 10^3 vec2 tile of (u, v) so the 6 stencil reads
// come from on-chip shared memory.

#if USE_SHARED_MEM
#define BKTILE 10
#define BKTILE3 (BKTILE * BKTILE * BKTILE)
shared vec2 s_uv[BKTILE3];
int bk_idx(int x, int y, int z) {
    return z * BKTILE * BKTILE + y * BKTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < BKTILE3; i += 512) {
        int tz = i / (BKTILE * BKTILE);
        int ty = (i / BKTILE) % BKTILE;
        int tx = i % BKTILE;
        s_uv[i] = fetch(tile_origin + ivec3(tx, ty, tz)).rg;
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec2 c = s_uv[bk_idx(tp.x, tp.y, tp.z)];
    float u = c.r;  // activator
    float v = c.g;  // inhibitor

    vec2 sum_uv = vec2(0.0);
    sum_uv += s_uv[bk_idx(tp.x + 1, tp.y,     tp.z    )];
    sum_uv += s_uv[bk_idx(tp.x - 1, tp.y,     tp.z    )];
    sum_uv += s_uv[bk_idx(tp.x,     tp.y + 1, tp.z    )];
    sum_uv += s_uv[bk_idx(tp.x,     tp.y - 1, tp.z    )];
    sum_uv += s_uv[bk_idx(tp.x,     tp.y,     tp.z + 1)];
    sum_uv += s_uv[bk_idx(tp.x,     tp.y,     tp.z - 1)];
    float lap_u = (sum_uv.x - 6.0 * u) * h_sq;
    float lap_v = (sum_uv.y - 6.0 * v) * h_sq;
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_val = fetch(pos);
    float u = self_val.r;  // activator (excitation)
    float v = self_val.g;  // inhibitor (recovery)

    // 6-neighbor Laplacian of (u, v) — same additive grouping as tiled path
    // so USE_SHARED_MEM=1 and =0 are byte-exact equivalent.
    vec2 sum_uv = vec2(0.0);
    sum_uv += fetch(pos + ivec3( 1, 0, 0)).rg;
    sum_uv += fetch(pos + ivec3(-1, 0, 0)).rg;
    sum_uv += fetch(pos + ivec3( 0, 1, 0)).rg;
    sum_uv += fetch(pos + ivec3( 0,-1, 0)).rg;
    sum_uv += fetch(pos + ivec3( 0, 0, 1)).rg;
    sum_uv += fetch(pos + ivec3( 0, 0,-1)).rg;
    float lap_u = (sum_uv.x - 6.0 * u) * h_sq;
    float lap_v = (sum_uv.y - 6.0 * v) * h_sq;
#endif

    float a = u_param0;       // excitability (0.6-1.0)
    float b = u_param1;       // threshold shift (0.01-0.1)
    float epsilon = u_param2; // timescale separation (0.02-0.1)
    float D_u = u_param3;     // diffusion

    // Barkley kinetics: du/dt = ε⁻¹ · u(1-u)(u - v_thresh) + D∇²u
    //                   dv/dt = u - v + small v-diffusion
    // Self-excitation boost: when v is low (fresh/resting tissue), u excitation
    // is amplified → sharper wavefronts and more sustained propagation
    float v_thresh = (v + b) / max(a, 1e-3);  // guard: u_param0 is user-settable
    float self_excite = 1.0 + 0.5 * max(0.0, 1.0 - 3.0 * v);  // up to 1.5× when v≈0
    float du = self_excite * u * (1.0 - u) * (u - v_thresh) / max(epsilon, 0.001);
    float dv = u - v;

    float new_u = u + (du + D_u * lap_u) * u_dt;
    // Small v-diffusion maintains spatial heterogeneity in recovery
    float new_v = v + (dv + D_u * 0.05 * lap_v) * u_dt;

    // Stochastic nucleation: rare random excitation of resting cells
    // Creates ongoing wave sources like pacemaker sites in real BZ
    float nuc = hash_temporal(pos, 0);
    if (nuc > 0.998 && u < 0.1 && v < 0.1) {
        new_u = 1.0;
    }

    new_u = clamp(new_u, 0.0, 1.0);
    new_v = clamp(new_v, 0.0, 1.5);

    // Phase for visualization
    float phase = new_u - new_v * 0.5;

    imageStore(u_dst, pos, vec4(new_u, new_v, phase, 0.0));
}
""",

    "morphogen_3d": """
// 3D Turing morphogenesis — activator-inhibitor with tissue growth
// R = activator a, G = inhibitor h, B = tissue density rho
// Classic Gierer-Meinhardt with proper decay: inhibitor diffuses faster,
// creating local activation + long-range inhibition → Turing patterns
//
// USE_SHARED_MEM=1 loads a 10^3 vec3 tile of (a, h, rho) so the 6 stencil
// reads (for all three Laplacians) come from on-chip shared memory.
// Tile cost: 10^3 * 12 bytes = 12000 bytes shared per workgroup.

#if USE_SHARED_MEM
#define MGTILE 10
#define MGTILE3 (MGTILE * MGTILE * MGTILE)
shared vec3 s_ahr[MGTILE3];
int mg_idx(int x, int y, int z) {
    return z * MGTILE * MGTILE + y * MGTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < MGTILE3; i += 512) {
        int tz = i / (MGTILE * MGTILE);
        int ty = (i / MGTILE) % MGTILE;
        int tx = i % MGTILE;
        s_ahr[i] = fetch(tile_origin + ivec3(tx, ty, tz)).rgb;
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec3 c = s_ahr[mg_idx(tp.x, tp.y, tp.z)];
    float a   = c.r;  // activator
    float h   = c.g;  // inhibitor
    float rho = c.b;  // tissue density / growth

    vec3 sum_f = vec3(0.0);
    sum_f += s_ahr[mg_idx(tp.x + 1, tp.y,     tp.z    )];
    sum_f += s_ahr[mg_idx(tp.x - 1, tp.y,     tp.z    )];
    sum_f += s_ahr[mg_idx(tp.x,     tp.y + 1, tp.z    )];
    sum_f += s_ahr[mg_idx(tp.x,     tp.y - 1, tp.z    )];
    sum_f += s_ahr[mg_idx(tp.x,     tp.y,     tp.z + 1)];
    sum_f += s_ahr[mg_idx(tp.x,     tp.y,     tp.z - 1)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_val = fetch(pos);
    float a = self_val.r;   // activator
    float h = self_val.g;   // inhibitor
    float rho = self_val.b; // tissue density / growth

    // Laplacians (inhibitor diffuses much faster — key to Turing instability)
    vec3 sum_f = vec3(0.0);
    sum_f += fetch(pos + ivec3( 1, 0, 0)).rgb;
    sum_f += fetch(pos + ivec3(-1, 0, 0)).rgb;
    sum_f += fetch(pos + ivec3( 0, 1, 0)).rgb;
    sum_f += fetch(pos + ivec3( 0,-1, 0)).rgb;
    sum_f += fetch(pos + ivec3( 0, 0, 1)).rgb;
    sum_f += fetch(pos + ivec3( 0, 0,-1)).rgb;
#endif
    float lap_a = (sum_f.r - 6.0 * a) * h_sq;
    float lap_h = (sum_f.g - 6.0 * h) * h_sq;
    float lap_rho = (sum_f.b - 6.0 * rho) * h_sq;

    float Da      = u_param0;  // activator diffusion (small)
    float Dh      = u_param1;  // inhibitor diffusion (large — key ratio!)
    float react   = u_param2;  // reaction strength
    float growth  = u_param3;  // tissue growth rate

    // Gierer-Meinhardt with proper decay terms:
    // da/dt = Da*lap(a) + react*(a^2/(h+0.001) - mu_a*a) + sigma_a
    // dh/dt = Dh*lap(h) + react*(a^2 - mu_h*h)
    // mu_a = decay rate of activator, mu_h = decay rate of inhibitor
    // sigma_a = basal production (small, prevents dead state)
    float mu_a = 1.0;      // activator decay
    float mu_h = 1.0;      // inhibitor decay
    float sigma_a = 0.01;  // basal activator production
    float a2 = a * a;
    // Saturating activator production: a^2/(h*(1+kappa*a^2)) prevents blowup
    float kappa = 0.1;  // saturation constant
    float production = a2 / (h + 0.001) / (1.0 + kappa * a2);
    float d_a = Da * lap_a + react * (production - mu_a * a) + sigma_a;
    float d_h = Dh * lap_h + react * (a2 - mu_h * h);

    // Tissue grows where activator is high, decays slowly elsewhere
    float d_rho = growth * (a - 0.3) * rho + 0.002 * lap_rho;

    float new_a = clamp(a + d_a * u_dt, 0.0, 5.0);
    float new_h = clamp(h + d_h * u_dt, 0.0, 10.0);
    float new_rho = clamp(rho + d_rho * u_dt, 0.0, 1.0);

    imageStore(u_dst, pos, vec4(new_a, new_h, new_rho, 0.0));
}
""",

    "flocking_3d": """
// 3D Flocking / Active Matter (Vicsek-style model on continuous field)
// R = density of agents, G = velocity_x, B = velocity_y, A = velocity_z
// Agents align with neighbors (flocking), diffuse, and self-propel
//
// USE_SHARED_MEM=1 loads a 10^3 vec4 tile of (rho, vx, vy, vz). Both
// the 26-neighbour Moore sum and the 6-point density gradient read
// from the tile; only the semi-Lagrangian back-trace (which can
// wander off-tile) stays on the direct-fetch path.
// Tile cost: 10^3 * 16 bytes = 16000 bytes shared per workgroup.

#if USE_SHARED_MEM
#define FKTILE 10
#define FKTILE3 (FKTILE * FKTILE * FKTILE)
shared vec4 s_rv[FKTILE3];
int fk_idx(int x, int y, int z) {
    return z * FKTILE * FKTILE + y * FKTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < FKTILE3; i += 512) {
        int tz = i / (FKTILE * FKTILE);
        int ty = (i / FKTILE) % FKTILE;
        int tx = i % FKTILE;
        s_rv[i] = fetch(tile_origin + ivec3(tx, ty, tz));
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec4 self_val = s_rv[fk_idx(tp.x, tp.y, tp.z)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_val = fetch(pos);
#endif
    float rho = self_val.r;    // density
    vec3 vel = self_val.gba;   // velocity field (vx, vy, vz)

    // Neighborhood averages (density and velocity)
    float sum_rho = 0.0;
    vec3 sum_vel = vec3(0.0);
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
#if USE_SHARED_MEM
        vec4 nb = s_rv[fk_idx(tp.x + dx, tp.y + dy, tp.z + dz)];
#else
        vec4 nb = fetch(pos + ivec3(dx, dy, dz));
#endif
        float w = nb.r;  // weight by neighbor density
        sum_rho += nb.r;
        sum_vel += nb.gba * w;
    }
    float avg_rho = sum_rho / 26.0;
    vec3 avg_vel = length(sum_vel) > 0.001 ? normalize(sum_vel) : vec3(0.0);

    float alignment = u_param0;    // flocking alignment strength
    float self_prop = u_param1;    // self-propulsion speed
    float diffusion = u_param2;    // density diffusion
    float repulsion = u_param3;    // crowding repulsion

    // Moore Laplacian of density, scaled for resolution-independence
    float lap_rho = (sum_rho - 26.0 * rho) * h_sq;

    // Velocity alignment: blend toward neighbor average
    // Density-velocity feedback: higher local density → stronger collective motion
    float density_boost = 1.0 + 2.0 * smoothstep(0.2, 0.6, avg_rho);
    vec3 target_vel = avg_vel * self_prop * density_boost;
    vec3 new_vel = mix(vel, target_vel, alignment * u_dt);

    // Repulsion from high density (prevents all agents piling up)
    float pressure = repulsion * (avg_rho - 0.3);
    // Gradient of density (central differences, scaled by h_inv).
    // Reuse the tile when present — these are among the 26 neighbours.
#if USE_SHARED_MEM
    float grad_x = (s_rv[fk_idx(tp.x + 1, tp.y,     tp.z    )].r -
                    s_rv[fk_idx(tp.x - 1, tp.y,     tp.z    )].r) * 0.5 * h_inv;
    float grad_y = (s_rv[fk_idx(tp.x,     tp.y + 1, tp.z    )].r -
                    s_rv[fk_idx(tp.x,     tp.y - 1, tp.z    )].r) * 0.5 * h_inv;
    float grad_z = (s_rv[fk_idx(tp.x,     tp.y,     tp.z + 1)].r -
                    s_rv[fk_idx(tp.x,     tp.y,     tp.z - 1)].r) * 0.5 * h_inv;
#else
    float grad_x = (fetch(pos + ivec3(1,0,0)).r - fetch(pos + ivec3(-1,0,0)).r) * 0.5 * h_inv;
    float grad_y = (fetch(pos + ivec3(0,1,0)).r - fetch(pos + ivec3(0,-1,0)).r) * 0.5 * h_inv;
    float grad_z = (fetch(pos + ivec3(0,0,1)).r - fetch(pos + ivec3(0,0,-1)).r) * 0.5 * h_inv;
#endif
    new_vel -= pressure * vec3(grad_x, grad_y, grad_z) * u_dt;

    // Temporal noise — different every step for ongoing random exploration
    float hash1 = hash_temporal(pos, 0);
    float hash2 = hash_temporal(pos, 1);
    float hash3 = hash_temporal(pos, 2);
    new_vel += (vec3(hash1, hash2, hash3) - 0.5) * 0.1 * u_dt;

    // Clamp velocity magnitude
    float spd = length(new_vel);
    if (spd > self_prop * 1.5) new_vel = new_vel / spd * self_prop * 1.5;

    // Semi-Lagrangian advection: back-trace to find where density came from
    // Unconditionally stable, preserves features better than Euler
    vec3 departure = vec3(pos) - vel * u_dt;
    float advected_rho = fetch_interp(departure).r;
    // Blend advected density with diffusion
    float new_rho = advected_rho + diffusion * lap_rho * u_dt;
    new_rho = clamp(new_rho, 0.0, 1.0);

    imageStore(u_dst, pos, vec4(new_rho, new_vel));
}
""",

    "cahn_hilliard": """
// Cahn-Hilliard phase separation (spinodal decomposition)
// Produces interconnected sponge-like structures from a mixed binary fluid.
// R = order parameter c ∈ [-1, 1] (two phases)
// G = chemical potential μ (computed each step for Laplacian-of-Laplacian)
// u_param0 = mobility M (transport rate)
// u_param1 = epsilon² (interface energy / width control)
// u_param2 = noise strength (thermal fluctuations)
// u_param3 = asymmetry (shifts the double-well: favors one phase)
//
// USE_SHARED_MEM=1 loads a 10^3 vec2 tile (c, μ); both Laplacians are
// read from on-chip memory.

#if USE_SHARED_MEM
#define CHTILE 10
#define CHTILE3 (CHTILE * CHTILE * CHTILE)
shared vec2 s_ch[CHTILE3];
int ch_idx(int x, int y, int z) {
    return z * CHTILE * CHTILE + y * CHTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < CHTILE3; i += 512) {
        int tz = i / (CHTILE * CHTILE);
        int ty = (i / CHTILE) % CHTILE;
        int tx = i % CHTILE;
        s_ch[i] = fetch(tile_origin + ivec3(tx, ty, tz)).rg;
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec2 sc = s_ch[ch_idx(tp.x, tp.y, tp.z)];
    float c = sc.r;
    float mu = sc.g;

    vec2 nxp = s_ch[ch_idx(tp.x + 1, tp.y,     tp.z    )];
    vec2 nxm = s_ch[ch_idx(tp.x - 1, tp.y,     tp.z    )];
    vec2 nyp = s_ch[ch_idx(tp.x,     tp.y + 1, tp.z    )];
    vec2 nym = s_ch[ch_idx(tp.x,     tp.y - 1, tp.z    )];
    vec2 nzp = s_ch[ch_idx(tp.x,     tp.y,     tp.z + 1)];
    vec2 nzm = s_ch[ch_idx(tp.x,     tp.y,     tp.z - 1)];

    // Preserve exact FP-associativity grouping of the direct-fetch path
    float lap_c = 0.0;
    lap_c += nxp.r + nxm.r;
    lap_c += nyp.r + nym.r;
    lap_c += nzp.r + nzm.r;
    lap_c = (lap_c - 6.0 * c) * h_sq;

    float lap_mu = 0.0;
    lap_mu += nxp.g + nxm.g;
    lap_mu += nyp.g + nym.g;
    lap_mu += nzp.g + nzm.g;
    lap_mu = (lap_mu - 6.0 * mu) * h_sq;
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_data = fetch(pos);
    float c = self_data.r;     // order parameter
    float mu = self_data.g;    // chemical potential from previous step

    // Von Neumann Laplacian of c (resolution-independent)
    float lap_c = 0.0;
    lap_c += fetch(pos + ivec3(1,0,0)).r + fetch(pos + ivec3(-1,0,0)).r;
    lap_c += fetch(pos + ivec3(0,1,0)).r + fetch(pos + ivec3(0,-1,0)).r;
    lap_c += fetch(pos + ivec3(0,0,1)).r + fetch(pos + ivec3(0,0,-1)).r;
    lap_c = (lap_c - 6.0 * c) * h_sq;

    // Laplacian of chemical potential (for ∇²μ → gives ∇⁴c term)
    float lap_mu = 0.0;
    lap_mu += fetch(pos + ivec3(1,0,0)).g + fetch(pos + ivec3(-1,0,0)).g;
    lap_mu += fetch(pos + ivec3(0,1,0)).g + fetch(pos + ivec3(0,-1,0)).g;
    lap_mu += fetch(pos + ivec3(0,0,1)).g + fetch(pos + ivec3(0,0,-1)).g;
    lap_mu = (lap_mu - 6.0 * mu) * h_sq;
#endif

    float mobility = u_param0;
    float eps2 = u_param1;
    float noise_str = u_param2;
    float asymmetry = u_param3;

    // Chemical potential: μ = f'(c) - ε²∇²c
    // f(c) = 0.25*(c²-1)² + asymmetry*c → f'(c) = c³ - c + asymmetry
    float new_mu = (c * c * c - c + asymmetry) - eps2 * lap_c;

    // Cahn-Hilliard: ∂c/∂t = M·∇²μ
    float new_c = c + mobility * lap_mu * u_dt;

    // Thermal noise for nucleation in metastable regions
    float hash = hash_temporal(pos, 0);
    new_c += (hash - 0.5) * noise_str * u_dt;

    new_c = clamp(new_c, -1.0, 1.0);

    imageStore(u_dst, pos, vec4(new_c, new_mu, 0.0, 0.0));
}
""",

    "erosion_3d": """
// Erosion & Sediment Transport with gravity flow
// Fluid flows downhill over terrain, erodes at interfaces, sediment
// advects with flow and deposits where slow. Creates channels and caves.
// R = solid density [0,1] (1=rock, 0=empty)
// G = fluid (water) amount [0,1]
// B = dissolved sediment carried by fluid [0,1]
// A = flow speed (diagnostic)
// u_param0 = erosion rate
// u_param1 = deposition rate
// u_param2 = fluid diffusion
// u_param3 = gravity strength
//
// USE_SHARED_MEM=1 loads a 10^3 vec4 tile — all 6 face neighbours are
// read twice (once for fluid/sediment Laplacian, once for the gravity
// flow + lateral-pressure pass) so the tile saves many imageLoads.

#if USE_SHARED_MEM
#define ERTILE 10
#define ERTILE3 (ERTILE * ERTILE * ERTILE)
shared vec4 s_er[ERTILE3];
int er_idx(int x, int y, int z) {
    return z * ERTILE * ERTILE + y * ERTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < ERTILE3; i += 512) {
        int tz = i / (ERTILE * ERTILE);
        int ty = (i / ERTILE) % ERTILE;
        int tx = i % ERTILE;
        s_er[i] = fetch(tile_origin + ivec3(tx, ty, tz));
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec4 self_data = s_er[er_idx(tp.x, tp.y, tp.z)];

    vec4 nxp = s_er[er_idx(tp.x + 1, tp.y,     tp.z    )];
    vec4 nxm = s_er[er_idx(tp.x - 1, tp.y,     tp.z    )];
    vec4 nyp = s_er[er_idx(tp.x,     tp.y + 1, tp.z    )];
    vec4 nym = s_er[er_idx(tp.x,     tp.y - 1, tp.z    )];
    vec4 nzp = s_er[er_idx(tp.x,     tp.y,     tp.z + 1)];
    vec4 nzm = s_er[er_idx(tp.x,     tp.y,     tp.z - 1)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_data = fetch(pos);
    // Fetch all 6 face neighbors
    vec4 nxp = fetch(pos + ivec3( 1,0,0));
    vec4 nxm = fetch(pos + ivec3(-1,0,0));
    vec4 nyp = fetch(pos + ivec3(0, 1,0));
    vec4 nym = fetch(pos + ivec3(0,-1,0));
    vec4 nzp = fetch(pos + ivec3(0,0, 1));
    vec4 nzm = fetch(pos + ivec3(0,0,-1));
#endif
    float solid = self_data.r;
    float fluid = self_data.g;
    float sediment = self_data.b;

    float erosion_rate = u_param0;
    float deposition_rate = u_param1;
    float diffusion = u_param2;
    float gravity = u_param3;

    // Fluid Laplacian (lateral diffusion/spreading, resolution-independent)
    float lap_fluid = (nxp.g + nxm.g + nyp.g + nym.g + nzp.g + nzm.g - 6.0 * fluid) * h_sq;

    // Gravity-driven vertical flow
    // Space available below: can receive fluid if cell has room
    float space_below = max(0.0, 1.0 - nym.r - nym.g);
    float flow_down = gravity * fluid * space_below;

    // Can receive fluid from above
    float space_here = max(0.0, 1.0 - solid - fluid);
    float flow_in = gravity * nyp.g * space_here;

    // Lateral pressure: fluid flows from high-fluid to low-fluid areas
    // Lateral pressure: fluid flows from high-fluid to low-fluid areas
    float lat_grad_x = (nxp.g - nxm.g) * 0.5 * h_inv;
    float lat_grad_z = (nzp.g - nzm.g) * 0.5 * h_inv;
    float lat_flow = -fluid * (lat_grad_x + lat_grad_z) * 0.2;

    // Velocity estimate (for erosion/deposition scaling)
    float vel_vert = abs(flow_down - flow_in);
    float vel_lat = sqrt(lat_grad_x * lat_grad_x + lat_grad_z * lat_grad_z) * fluid;
    float velocity = vel_vert + vel_lat;

    // Fluid update
    float new_fluid = fluid + diffusion * lap_fluid * u_dt;
    new_fluid += (-flow_down + flow_in + lat_flow) * u_dt;

    // Erosion: fluid dissolves rock at interfaces
    // Stronger with more fluid and higher flow speed
    // Only erode where there IS solid and adjacent fluid
    float erode = erosion_rate * fluid * (velocity + 0.01) * solid * u_dt;
    float new_solid = solid - erode;

    // Sediment: eroded material enters fluid, advects with it
    float lap_sed = (nxp.b + nxm.b + nyp.b + nym.b + nzp.b + nzm.b - 6.0 * sediment) * h_sq;
    float new_sediment = sediment + erode;
    new_sediment += diffusion * 0.5 * lap_sed * u_dt;

    // Sediment follows gravity too
    float sed_space_below = max(0.0, 1.0 - nym.r);
    float sed_down = gravity * 0.5 * sediment * sed_space_below;
    float sed_in = gravity * 0.5 * nyp.b * max(0.0, 1.0 - solid);
    new_sediment += (-sed_down + sed_in) * u_dt;

    // Deposition: sediment settles where flow is slow + solid support below
    float support = step(0.3, nym.r);
    float deposit = deposition_rate * sediment * support * max(0.0, 1.0 - velocity * 3.0) * u_dt;
    new_solid += deposit;
    new_sediment -= deposit;

    // Water source replenishment at top boundary (rain)
    if (u_boundary == 1 && pos.y >= u_size - 2) {
        new_fluid = max(new_fluid, 0.6);
    }

    new_solid = clamp(new_solid, 0.0, 1.0);
    new_fluid = clamp(new_fluid, 0.0, 1.0);
    new_sediment = clamp(new_sediment, 0.0, 1.0);

    imageStore(u_dst, pos, vec4(new_solid, new_fluid, new_sediment, velocity));
}
""",

    "mycelium_3d": """
// Mycelium / Fungal Network
// Hyphal tips explore, branch, fuse on contact, transport nutrients.
// R = biomass (0=empty, >0.5=established hypha, small positive=tip)
// G = nutrient concentration (diffuses, consumed by biomass)
// B = signal/pheromone (tips emit, guides branching)
// A = tip marker (1.0=active tip, 0=established or empty)
// u_param0 = growth rate (tip extension speed)
// u_param1 = branch probability factor
// u_param2 = nutrient consumption rate
// u_param3 = nutrient diffusion
//
// USE_SHARED_MEM=1 loads a 10^3 vec4 tile for the 26-neighbor Moore scan.

#if USE_SHARED_MEM
#define MYTILE 10
#define MYTILE3 (MYTILE * MYTILE * MYTILE)
shared vec4 s_my[MYTILE3];
int my_idx(int x, int y, int z) {
    return z * MYTILE * MYTILE + y * MYTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < MYTILE3; i += 512) {
        int tz = i / (MYTILE * MYTILE);
        int ty = (i / MYTILE) % MYTILE;
        int tx = i % MYTILE;
        s_my[i] = fetch(tile_origin + ivec3(tx, ty, tz));
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec4 self_data = s_my[my_idx(tp.x, tp.y, tp.z)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_data = fetch(pos);
#endif
    float biomass = self_data.r;
    float nutrient = self_data.g;
    float signal = self_data.b;
    float tip = self_data.a;

    float growth = u_param0;
    float branch_factor = u_param1;
    float consumption = u_param2;
    float diffusion = u_param3;

    // Neighborhood scan
    float sum_bio = 0.0, sum_nut = 0.0, sum_sig = 0.0;
    int bio_count = 0;
    float max_tip = 0.0;
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
#if USE_SHARED_MEM
        vec4 nb = s_my[my_idx(tp.x + dx, tp.y + dy, tp.z + dz)];
#else
        vec4 nb = fetch(pos + ivec3(dx, dy, dz));
#endif
        sum_bio += nb.r;
        sum_nut += nb.g;
        sum_sig += nb.b;
        if (nb.r > 0.1) bio_count++;
        max_tip = max(max_tip, nb.a);
    }

    // Nutrient diffusion (Laplacian, resolution-independent)
    float lap_nut = (sum_nut - 26.0 * nutrient) * h_sq;
    // Signal diffusion + decay
    float lap_sig = (sum_sig - 26.0 * signal) * h_sq;

    float new_bio = biomass;
    float new_nut = nutrient + diffusion * lap_nut * u_dt;
    float new_sig = signal + diffusion * 0.5 * lap_sig * u_dt - signal * 0.05 * u_dt;
    float new_tip = 0.0;

    // Pseudo-random from position + nutrient
    float hash = hash_temporal(pos, 0);

    if (biomass > 0.5) {
        // Established hypha: consume nutrient, maintain
        new_nut -= consumption * biomass * u_dt;
        // Emit signal
        new_sig += biomass * 0.02 * u_dt;
        // Die if nutrient depleted
        if (nutrient < 0.01) new_bio -= 0.1 * u_dt;
    } else if (biomass > 0.01) {
        // Growing region
        new_bio += growth * nutrient * u_dt;
        new_nut -= consumption * 0.5 * u_dt;
        if (new_bio > 0.5) new_bio = 1.0;  // solidify
    } else {
        // Empty cell: can be colonized by adjacent tip
        if (max_tip > 0.3 && nutrient > 0.05) {
            // Tip extends into this cell
            // More likely toward nutrient gradient (chemotaxis)
            float nut_grad = nutrient - sum_nut / 26.0;
            float grow_prob = growth * (0.3 + 0.7 * clamp(nut_grad * 10.0 + 0.5, 0.0, 1.0));
            // Branching: new tips from existing tips (probability)
            float branch_chance = branch_factor * 0.1 * max_tip;
            float total_prob = min(grow_prob + branch_chance, 1.0);
            if (hash < total_prob * u_dt && bio_count >= 1 && bio_count <= 3) {
                new_bio = 0.1;  // start growing
                new_tip = 1.0;  // this is now a tip
            }
        }
    }

    // Tips that have neighbors filling in behind them lose tip status
    if (tip > 0.3 && bio_count >= 4) new_tip = 0.0;
    // Active tip if we're a tip or newly created
    if (biomass > 0.01 && biomass < 0.5 && bio_count <= 3) new_tip = max(new_tip, 0.5);

    new_bio = clamp(new_bio, 0.0, 1.0);
    new_nut = clamp(new_nut, 0.0, 1.0);
    new_sig = clamp(new_sig, 0.0, 1.0);

    imageStore(u_dst, pos, vec4(new_bio, new_nut, new_sig, new_tip));
}
""",

    "em_wave_3d": """
// Electromagnetic wave propagation (simplified Maxwell)
// Uses E and B fields packed into RGBA:
// R = Ez (electric field z-component)
// G = Bx (magnetic field x)
// B = By (magnetic field y)
// A = conductivity/medium marker (0=vacuum, >0=conductor absorbs)
// 2D TE-like mode extended to 3D: Ez, Bx, By propagation
// u_param0 = wave speed (c)
// u_param1 = damping in conductors
// u_param2 = source frequency
// u_param3 = source amplitude
//
// USE_SHARED_MEM=1 loads a 10^3 vec4 tile; all six axial derivatives of
// Ez, Bx, By come from on-chip memory.

#if USE_SHARED_MEM
#define EMTILE 10
#define EMTILE3 (EMTILE * EMTILE * EMTILE)
shared vec4 s_em[EMTILE3];
int em_idx(int x, int y, int z) {
    return z * EMTILE * EMTILE + y * EMTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < EMTILE3; i += 512) {
        int tz = i / (EMTILE * EMTILE);
        int ty = (i / EMTILE) % EMTILE;
        int tx = i % EMTILE;
        s_em[i] = fetch(tile_origin + ivec3(tx, ty, tz));
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec4 self_data = s_em[em_idx(tp.x, tp.y, tp.z)];

    vec4 nxp = s_em[em_idx(tp.x + 1, tp.y,     tp.z    )];
    vec4 nxm = s_em[em_idx(tp.x - 1, tp.y,     tp.z    )];
    vec4 nyp = s_em[em_idx(tp.x,     tp.y + 1, tp.z    )];
    vec4 nym = s_em[em_idx(tp.x,     tp.y - 1, tp.z    )];
    vec4 nzp = s_em[em_idx(tp.x,     tp.y,     tp.z + 1)];
    vec4 nzm = s_em[em_idx(tp.x,     tp.y,     tp.z - 1)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_data = fetch(pos);
    vec4 nxp = fetch(pos + ivec3( 1, 0, 0));
    vec4 nxm = fetch(pos + ivec3(-1, 0, 0));
    vec4 nyp = fetch(pos + ivec3( 0, 1, 0));
    vec4 nym = fetch(pos + ivec3( 0,-1, 0));
    vec4 nzp = fetch(pos + ivec3( 0, 0, 1));
    vec4 nzm = fetch(pos + ivec3( 0, 0,-1));
#endif
    float Ez = self_data.r;
    float Bx = self_data.g;
    float By = self_data.b;
    float medium = self_data.a;

    float c = u_param0;
    float damping = u_param1;
    float freq = u_param2;
    float amplitude = u_param3;

    // Curl of B → updates E: dEz/dt = c²(dBy/dx - dBx/dy + dBy/dz - dBx/dz)
    // Scale derivatives by h_inv for resolution-independence
    float dBy_dx = (nxp.b - nxm.b) * 0.5 * h_inv;
    float dBx_dy = (nyp.g - nym.g) * 0.5 * h_inv;
    float dBy_dz = (nzp.b - nzm.b) * 0.5 * h_inv;
    float dBx_dz = (nzp.g - nzm.g) * 0.5 * h_inv;

    float new_Ez = Ez + c * c * (dBy_dx - dBx_dy + dBy_dz - dBx_dz) * u_dt;

    // Curl of E → updates B: dBx/dt = -dEz/dy + dEz/dz, dBy/dt = dEz/dx - dEz/dz
    // Scale derivatives by h_inv for resolution-independence
    float dEz_dx = (nxp.r - nxm.r) * 0.5 * h_inv;
    float dEz_dy = (nyp.r - nym.r) * 0.5 * h_inv;
    float dEz_dz = (nzp.r - nzm.r) * 0.5 * h_inv;

    float new_Bx = Bx - dEz_dy * u_dt + dEz_dz * u_dt;
    float new_By = By + dEz_dx * u_dt - dEz_dz * u_dt;

    // Absorbing conductor: damp fields
    if (medium > 0.0) {
        float d = damping * medium;
        new_Ez *= exp(-d * u_dt);
        new_Bx *= exp(-d * u_dt);
        new_By *= exp(-d * u_dt);
    }

    // Dipole source at center — soft oscillator driven by local field sign
    int mid = u_size / 2;
    if (abs(pos.x - mid) <= 1 && abs(pos.y - mid) <= 1 && abs(pos.z - mid) <= 1) {
        // Drive Ez toward a sinusoidal target using feedback:
        // target oscillates based on current phase of the local field
        float target = amplitude * sin(freq * 6.28 * Ez);
        new_Ez += (target - Ez) * freq * u_dt;
    }

    new_Ez = clamp(new_Ez, -2.0, 2.0);
    new_Bx = clamp(new_Bx, -2.0, 2.0);
    new_By = clamp(new_By, -2.0, 2.0);

    imageStore(u_dst, pos, vec4(new_Ez, new_Bx, new_By, medium));
}
""",

    "viscous_fingers_3d": """
// Viscous Fingering (Saffman-Taylor instability / Hele-Shaw)
// Low-viscosity fluid invades high-viscosity fluid under pressure.
// R = saturation of invading fluid [0,1] (0=defending fluid, 1=invader)
// G = pressure field
// B = permeability field (evolves: invader dissolves medium, increasing permeability)
// A = interface marker (for visualization)
// u_param0 = injection pressure
// u_param1 = viscosity ratio (M = μ_defending / μ_invading)
// u_param2 = noise (porous medium heterogeneity)
// u_param3 = surface tension (stabilizes/smooths fingers)
//
// USE_SHARED_MEM=1 loads a 10^3 vec4 tile — two separate 6-face passes
// (pressure-mobility relaxation and saturation advection) both pull
// the same neighbours, so the tile is reused heavily.

#if USE_SHARED_MEM
#define VFTILE 10
#define VFTILE3 (VFTILE * VFTILE * VFTILE)
shared vec4 s_vf[VFTILE3];
int vf_idx(int x, int y, int z) {
    return z * VFTILE * VFTILE + y * VFTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < VFTILE3; i += 512) {
        int tz = i / (VFTILE * VFTILE);
        int ty = (i / VFTILE) % VFTILE;
        int tx = i % VFTILE;
        s_vf[i] = fetch(tile_origin + ivec3(tx, ty, tz));
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec4 self_data = s_vf[vf_idx(tp.x, tp.y, tp.z)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_data = fetch(pos);
#endif
    float sat = self_data.r;       // invader saturation
    float pressure = self_data.g;
    float perm = self_data.b;      // local permeability [0,1]
    float iface = self_data.a;

    float inject_p = u_param0;
    float visc_ratio = u_param1;
    float noise_str = u_param2;
    float surf_tension = u_param3;

    // Effective mobility: depends on saturation (which fluid) AND local permeability
    float eff_visc = mix(visc_ratio, 1.0, sat);  // interpolate viscosity
    float mobility = perm / max(eff_visc, 0.01);

    // Pressure Laplacian (Darcy flow: ∇·(k/μ ∇p) = 0, iterative relaxation).
    // Uses mobility-weighted averaging (sum_p_mob / sum_mob) rather than a
    // plain Laplacian since mobility varies with saturation.
    float sum_p_mob = 0.0;
    float sum_mob = 0.0;
    float lap_perm = 0.0;
    for (int i = 0; i < 6; i++) {
        ivec3 off = ivec3(0);
        int axis = i / 2; int dir = (i % 2) * 2 - 1;
        off[axis] = dir;
#if USE_SHARED_MEM
        vec4 nb = s_vf[vf_idx(tp.x + off.x, tp.y + off.y, tp.z + off.z)];
#else
        vec4 nb = fetch(pos + off);
#endif
        float nb_sat = nb.r;
        float nb_perm = nb.b;
        float nb_mob = nb_perm / max(mix(visc_ratio, 1.0, nb_sat), 0.01);
        float avg_mob = 0.5 * (mobility + nb_mob);
        sum_p_mob += avg_mob * nb.g;
        sum_mob += avg_mob;
        lap_perm += nb_perm;
    }
    lap_perm -= 6.0 * perm;
    lap_perm *= h_sq;  // resolution-independent

    // Relaxation toward pressure equilibrium weighted by mobility
    float new_pressure = mix(pressure, sum_p_mob / max(sum_mob, 0.001), 0.2);

    // Saturation: invader advances where pressure gradient overcomes capillary pressure
    float lap_sat = 0.0;
    float grad_p_dot_grad_sat = 0.0;
    for (int i = 0; i < 6; i++) {
        ivec3 off = ivec3(0);
        int axis = i / 2; int dir = (i % 2) * 2 - 1;
        off[axis] = dir;
#if USE_SHARED_MEM
        vec4 nb = s_vf[vf_idx(tp.x + off.x, tp.y + off.y, tp.z + off.z)];
#else
        vec4 nb = fetch(pos + off);
#endif
        lap_sat += nb.r;
        // Upwind: fluid moves from high to low pressure
        float dp = nb.g - pressure;
        if (dp > 0.0) {
            // Neighbor has higher pressure → pushes fluid toward us
            grad_p_dot_grad_sat += dp * (nb.r - sat);
        }
    }
    lap_sat -= 6.0 * sat;
    lap_sat *= h_sq;  // resolution-independent

    // Pseudo-noise for porous medium heterogeneity
    float hash = fract(sin(dot(vec3(pos)*0.17, vec3(12.9898, 78.233, 45.5432))) * 43758.5453);
    float perm_noise = 1.0 + (hash - 0.5) * noise_str;

    float new_sat = sat;
    // Invasion: mobility-weighted pressure-driven flow
    new_sat += mobility * perm_noise * grad_p_dot_grad_sat * u_dt;
    // Surface tension: Laplacian smoothing resists curvature
    new_sat += surf_tension * lap_sat * u_dt;

    // Permeability evolution: invading fluid dissolves medium at the interface
    // Creates positive feedback: dissolution widens channels → more flow → more dissolution
    float at_interface = sat * (1.0 - sat) * 4.0;  // peaks at sat=0.5
    float new_perm = perm + 0.1 * at_interface * u_dt;
    // Slow permeability diffusion (spreading of dissolution)
    new_perm += 0.02 * lap_perm * u_dt;
    new_perm = clamp(new_perm, 0.1, 1.0);

    // Source: inject at center
    int mid = u_size / 2;
    float dist_to_center = length(vec3(pos) - vec3(mid));
    if (dist_to_center < float(u_size) / 32.0) {
        new_sat = 1.0;
        new_pressure = inject_p;
    }

    // Boundary pressure (drain at edges for clamped mode)
    if (pos.x == 0 || pos.x == u_size-1 ||
        pos.y == 0 || pos.y == u_size-1 ||
        pos.z == 0 || pos.z == u_size-1) {
        new_pressure = 0.0;
    }

    // Interface detection
    float new_iface = (sat > 0.1 && sat < 0.9) ? 1.0 : 0.0;

    new_sat = clamp(new_sat, 0.0, 1.0);

    imageStore(u_dst, pos, vec4(new_sat, new_pressure, new_perm, new_iface));
}
""",

    "fire_3d": """
// Fire / Combustion front propagation
// R = fuel [0,1] (1=unburned, 0=ash)
// G = temperature [0,1] (normalized: 0=cold, 1=max heat)
// B = oxygen [0,1]
// A = ember marker (hot flying particles)
// u_param0 = ignition temperature threshold
// u_param1 = heat output (combustion energy)
// u_param2 = heat diffusion
// u_param3 = wind strength (upward bias + some lateral)
//
// USE_SHARED_MEM=1 loads a 10^3 vec4 tile: temp and oxygen Laplacians
// plus the wind-advection read (y±1) all come from the tile. The ember
// +Y neighbour read is also on the tile.

#if USE_SHARED_MEM
#define FRTILE 10
#define FRTILE3 (FRTILE * FRTILE * FRTILE)
shared vec4 s_fr[FRTILE3];
int fr_idx(int x, int y, int z) {
    return z * FRTILE * FRTILE + y * FRTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < FRTILE3; i += 512) {
        int tz = i / (FRTILE * FRTILE);
        int ty = (i / FRTILE) % FRTILE;
        int tx = i % FRTILE;
        s_fr[i] = fetch(tile_origin + ivec3(tx, ty, tz));
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec4 self_data = s_fr[fr_idx(tp.x, tp.y, tp.z)];

    // Pre-fetch the 6 face neighbours from the tile
    vec4 nxp = s_fr[fr_idx(tp.x + 1, tp.y,     tp.z    )];
    vec4 nxm = s_fr[fr_idx(tp.x - 1, tp.y,     tp.z    )];
    vec4 nyp = s_fr[fr_idx(tp.x,     tp.y + 1, tp.z    )];
    vec4 nym = s_fr[fr_idx(tp.x,     tp.y - 1, tp.z    )];
    vec4 nzp = s_fr[fr_idx(tp.x,     tp.y,     tp.z + 1)];
    vec4 nzm = s_fr[fr_idx(tp.x,     tp.y,     tp.z - 1)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_data = fetch(pos);
    vec4 nxp = fetch(pos + ivec3( 1, 0, 0));
    vec4 nxm = fetch(pos + ivec3(-1, 0, 0));
    vec4 nyp = fetch(pos + ivec3( 0, 1, 0));
    vec4 nym = fetch(pos + ivec3( 0,-1, 0));
    vec4 nzp = fetch(pos + ivec3( 0, 0, 1));
    vec4 nzm = fetch(pos + ivec3( 0, 0,-1));
#endif
    float fuel = self_data.r;
    float temp = self_data.g;
    float oxygen = self_data.b;
    float ember = self_data.a;

    float ignition = u_param0;
    float heat_output = u_param1;
    float diffusion = u_param2;
    float wind = u_param3;

    // Temperature Laplacian (heat diffusion)
    float lap_temp = (nxp.g + nxm.g + nyp.g + nym.g + nzp.g + nzm.g - 6.0 * temp) * h_sq;

    // Oxygen Laplacian
    float lap_oxy = (nxp.b + nxm.b + nyp.b + nym.b + nzp.b + nzm.b - 6.0 * oxygen) * h_sq;

    // Wind: bias heat transport upward (+Y) and slight lateral
    float heat_below = nym.g;
    float heat_above = nyp.g;
    float wind_advect = wind * (heat_below - heat_above) * 0.5;

    float new_fuel = fuel;
    float new_temp = temp + diffusion * lap_temp * u_dt + wind_advect * u_dt;
    float new_oxy = oxygen + diffusion * 0.8 * lap_oxy * u_dt;
    float new_ember = ember;

    // Combustion: fuel burns when temp > ignition and oxygen present
    if (fuel > 0.01 && temp > ignition && oxygen > 0.05) {
        float burn_rate = fuel * (temp - ignition) * oxygen * heat_output;
        new_fuel -= burn_rate * u_dt;
        new_temp += burn_rate * 2.0 * u_dt;  // exothermic
        new_oxy -= burn_rate * 0.5 * u_dt;   // consume oxygen

        // Ember generation (stochastic)
        float hash = hash_temporal(pos, 0);
        if (hash < burn_rate * 0.1 * u_dt) new_ember = 1.0;
    }

    // Ember transport (rises and cools)
    float ember_below = nym.a;
    new_ember = max(new_ember, ember_below * 0.8);  // embers rise
    new_ember -= 0.05 * u_dt;  // cool/fade
    // Embers can ignite fuel
    if (new_ember > 0.3 && fuel > 0.5) new_temp += new_ember * 0.3 * u_dt;

    // Radiative cooling
    new_temp -= temp * 0.02 * u_dt;

    // Oxygen replenishment from boundaries
    if (u_boundary == 1) {
        if (pos.x == 0 || pos.x == u_size-1 ||
            pos.y == 0 || pos.y == u_size-1 ||
            pos.z == 0 || pos.z == u_size-1) {
            new_oxy = mix(new_oxy, 1.0, 0.1 * u_dt);
        }
    }

    new_fuel = clamp(new_fuel, 0.0, 1.0);
    new_temp = clamp(new_temp, 0.0, 1.0);
    new_oxy = clamp(new_oxy, 0.0, 1.0);
    new_ember = clamp(new_ember, 0.0, 1.0);

    imageStore(u_dst, pos, vec4(new_fuel, new_temp, new_oxy, new_ember));
}
""",

    "physarum_3d": """
// Physarum / Slime Mold (3D chemotaxis network)
// Agents deposit trail, sense trail gradient, move toward strongest signal.
// R = trail pheromone concentration [0,1]
// G = agent density [0,1]
// B = nutrient (food sources)
// A = trail deposition rate marker
// u_param0 = sensor distance (how far agents look)
// u_param1 = turn strength (chemotaxis response)
// u_param2 = trail decay rate
// u_param3 = trail diffusion
//
// USE_SHARED_MEM=1 loads a 10^3 vec4 tile for the 6 axial reads used by
// trail/agent Laplacians and trail+food gradients. fetch_interp for the
// semi-Lagrangian back-trace stays off-tile (stepsize > 1 voxel).

#if USE_SHARED_MEM
#define PHTILE 10
#define PHTILE3 (PHTILE * PHTILE * PHTILE)
shared vec4 s_ph[PHTILE3];
int ph_idx(int x, int y, int z) {
    return z * PHTILE * PHTILE + y * PHTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < PHTILE3; i += 512) {
        int tz = i / (PHTILE * PHTILE);
        int ty = (i / PHTILE) % PHTILE;
        int tx = i % PHTILE;
        s_ph[i] = fetch(tile_origin + ivec3(tx, ty, tz));
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec4 self_data = s_ph[ph_idx(tp.x, tp.y, tp.z)];

    vec4 nxp = s_ph[ph_idx(tp.x + 1, tp.y,     tp.z    )];
    vec4 nxm = s_ph[ph_idx(tp.x - 1, tp.y,     tp.z    )];
    vec4 nyp = s_ph[ph_idx(tp.x,     tp.y + 1, tp.z    )];
    vec4 nym = s_ph[ph_idx(tp.x,     tp.y - 1, tp.z    )];
    vec4 nzp = s_ph[ph_idx(tp.x,     tp.y,     tp.z + 1)];
    vec4 nzm = s_ph[ph_idx(tp.x,     tp.y,     tp.z - 1)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_data = fetch(pos);
    vec4 nxp = fetch(pos + ivec3( 1, 0, 0));
    vec4 nxm = fetch(pos + ivec3(-1, 0, 0));
    vec4 nyp = fetch(pos + ivec3( 0, 1, 0));
    vec4 nym = fetch(pos + ivec3( 0,-1, 0));
    vec4 nzp = fetch(pos + ivec3( 0, 0, 1));
    vec4 nzm = fetch(pos + ivec3( 0, 0,-1));
#endif
    float trail = self_data.r;
    float agents = self_data.g;
    float food = self_data.b;

    float sensor_dist = max(1.0, u_param0);
    float turn = u_param1;
    float decay = u_param2;
    float diffusion = u_param3;

    // Trail Laplacian (diffusion) — preserve pairwise grouping for FP assoc.
    float lap_trail = 0.0;
    lap_trail += nxp.r + nxm.r;
    lap_trail += nyp.r + nym.r;
    lap_trail += nzp.r + nzm.r;
    lap_trail -= 6.0 * trail;
    lap_trail *= h_sq;  // resolution-independent

    // Agent density Laplacian (agents diffuse/move)
    float lap_agents = 0.0;
    lap_agents += nxp.g + nxm.g;
    lap_agents += nyp.g + nym.g;
    lap_agents += nzp.g + nzm.g;
    lap_agents -= 6.0 * agents;
    lap_agents *= h_sq;  // resolution-independent

    // Trail gradient (agents move toward strongest trail, scaled by h_inv)
    float gx = (nxp.r - nxm.r) * 0.5 * h_inv;
    float gy = (nyp.r - nym.r) * 0.5 * h_inv;
    float gz = (nzp.r - nzm.r) * 0.5 * h_inv;

    // Also sense food
    float fx = (nxp.b - nxm.b) * 0.5 * h_inv;
    float fy = (nyp.b - nym.b) * 0.5 * h_inv;
    float fz = (nzp.b - nzm.b) * 0.5 * h_inv;

    // Combined chemotactic gradient (trail + food attraction)
    float tot_gx = gx + fx * 2.0;
    float tot_gy = gy + fy * 2.0;
    float tot_gz = gz + fz * 2.0;

    // Random noise for exploration
    float hash = hash_temporal(pos, 0);

    // Update
    float new_trail = trail + diffusion * lap_trail * u_dt;
    new_trail += agents * 0.3 * u_dt;     // agents deposit trail
    new_trail -= decay * trail * u_dt;     // evaporation
    new_trail += food * 0.1 * u_dt;       // food attracts via scent

    float new_agents = agents;
    // Semi-Lagrangian chemotactic advection: agents back-trace along gradient
    vec3 chemotax_vel = vec3(tot_gx, tot_gy, tot_gz) * turn;
    vec3 agent_departure = vec3(pos) - chemotax_vel * u_dt;
    new_agents = fetch_interp(agent_departure).g;
    new_agents += 0.1 * lap_agents * u_dt;  // base diffusion
    new_agents += (hash - 0.5) * 0.02 * u_dt;     // exploration noise

    // Food consumption: agents at food sources multiply
    float new_food = food;
    if (agents > 0.01 && food > 0.01) {
        float eat = min(agents * 0.1 * u_dt, food);
        new_food -= eat;
        new_agents += eat * 0.5;  // grow from food
    }

    new_trail = clamp(new_trail, 0.0, 1.0);
    new_agents = clamp(new_agents, 0.0, 1.0);
    new_food = clamp(new_food, 0.0, 1.0);

    imageStore(u_dst, pos, vec4(new_trail, new_agents, new_food, 0.0));
}
""",

    "fracture_3d": """
// Elastic Stress / Fracture propagation
// Material under internal stress. Cracks propagate from stress concentration.
// R = displacement (deformation field)
// G = stress magnitude
// B = integrity [0,1] (1=intact, 0=broken — irreversible)
// A = strain energy (accumulated)
// u_param0 = stress wave speed
// u_param1 = fracture threshold
// u_param2 = stress diffusion
// u_param3 = initial stress intensity
//
// USE_SHARED_MEM=1 loads a 10^3 vec4 tile for both the face Laplacians
// (disp, stress) and the 26-neighbor broken-neighbor count.

#if USE_SHARED_MEM
#define FCTILE 10
#define FCTILE3 (FCTILE * FCTILE * FCTILE)
shared vec4 s_fc[FCTILE3];
int fc_idx(int x, int y, int z) {
    return z * FCTILE * FCTILE + y * FCTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < FCTILE3; i += 512) {
        int tz = i / (FCTILE * FCTILE);
        int ty = (i / FCTILE) % FCTILE;
        int tx = i % FCTILE;
        s_fc[i] = fetch(tile_origin + ivec3(tx, ty, tz));
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec4 self_data = s_fc[fc_idx(tp.x, tp.y, tp.z)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_data = fetch(pos);
#endif
    float disp = self_data.r;
    float stress = self_data.g;
    float integrity = self_data.b;
    float strain = self_data.a;

    float wave_speed = u_param0;
    float frac_thresh = u_param1;
    float diffusion = u_param2;
    float intensity = u_param3;

    // Displacement Laplacian (elastic wave equation) — preserve pairwise grouping
    float lap_disp = 0.0;
#if USE_SHARED_MEM
    lap_disp += s_fc[fc_idx(tp.x+1, tp.y,   tp.z  )].r + s_fc[fc_idx(tp.x-1, tp.y,   tp.z  )].r;
    lap_disp += s_fc[fc_idx(tp.x,   tp.y+1, tp.z  )].r + s_fc[fc_idx(tp.x,   tp.y-1, tp.z  )].r;
    lap_disp += s_fc[fc_idx(tp.x,   tp.y,   tp.z+1)].r + s_fc[fc_idx(tp.x,   tp.y,   tp.z-1)].r;
#else
    lap_disp += fetch(pos + ivec3(1,0,0)).r + fetch(pos + ivec3(-1,0,0)).r;
    lap_disp += fetch(pos + ivec3(0,1,0)).r + fetch(pos + ivec3(0,-1,0)).r;
    lap_disp += fetch(pos + ivec3(0,0,1)).r + fetch(pos + ivec3(0,0,-1)).r;
#endif
    lap_disp -= 6.0 * disp;
    lap_disp *= h_sq;  // resolution-independent

    // Stress Laplacian (stress diffusion / redistribution)
    float lap_stress = 0.0;
#if USE_SHARED_MEM
    lap_stress += s_fc[fc_idx(tp.x+1, tp.y,   tp.z  )].g + s_fc[fc_idx(tp.x-1, tp.y,   tp.z  )].g;
    lap_stress += s_fc[fc_idx(tp.x,   tp.y+1, tp.z  )].g + s_fc[fc_idx(tp.x,   tp.y-1, tp.z  )].g;
    lap_stress += s_fc[fc_idx(tp.x,   tp.y,   tp.z+1)].g + s_fc[fc_idx(tp.x,   tp.y,   tp.z-1)].g;
#else
    lap_stress += fetch(pos + ivec3(1,0,0)).g + fetch(pos + ivec3(-1,0,0)).g;
    lap_stress += fetch(pos + ivec3(0,1,0)).g + fetch(pos + ivec3(0,-1,0)).g;
    lap_stress += fetch(pos + ivec3(0,0,1)).g + fetch(pos + ivec3(0,0,-1)).g;
#endif
    lap_stress -= 6.0 * stress;
    lap_stress *= h_sq;  // resolution-independent

    // Count broken neighbors (stress concentrates at crack tips)
    float broken_neighbors = 0.0;
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
#if USE_SHARED_MEM
        float nb_int = s_fc[fc_idx(tp.x + dx, tp.y + dy, tp.z + dz)].b;
#else
        float nb_int = fetch(pos + ivec3(dx, dy, dz)).b;
#endif
        if (nb_int < 0.5) broken_neighbors += 1.0;
    }

    float new_disp = disp;
    float new_stress = stress;
    float new_integrity = integrity;
    float new_strain = strain;

    if (integrity > 0.5) {
        // Intact material: elastic wave propagation
        new_disp += wave_speed * wave_speed * lap_disp * integrity * u_dt;
        // Stress redistributes + concentrates near cracks
        new_stress += diffusion * lap_stress * u_dt;
        // Stress concentration at crack tips (more broken neighbors = more stress)
        new_stress += broken_neighbors * intensity * 0.01 * u_dt;
        // Strain accumulates
        new_strain += abs(new_stress) * u_dt;

        // Fracture: material breaks when stress exceeds threshold
        // Lower threshold near existing cracks (stress concentration factor)
        float effective_thresh = frac_thresh * (1.0 - broken_neighbors * 0.03);
        if (abs(new_stress) > effective_thresh) {
            new_integrity = 0.0;  // BREAK (irreversible)
            // Release energy as displacement wave
            new_disp += new_stress * 0.5;
            new_stress *= 0.3;  // partial stress release
        }

        // Damping
        new_disp *= 0.999;
    } else {
        // Broken: no elastic response, stress passes through weakly
        new_stress += diffusion * 0.1 * lap_stress * u_dt;
        new_stress *= 0.95;  // rapid stress dissipation in void
        new_disp *= 0.9;     // heavy damping
    }

    new_disp = clamp(new_disp, -2.0, 2.0);
    new_stress = clamp(new_stress, -2.0, 2.0);
    new_strain = clamp(new_strain, 0.0, 10.0);

    imageStore(u_dst, pos, vec4(new_disp, new_stress, new_integrity, new_strain));
}
""",

    "galaxy_3d": """
// Galaxy / N-body gravity (density field with multi-scale self-gravity)
// Multi-scale gravity: samples density at distances 1, 2, 4 cells
// to approximate longer-range 1/r² forces. Produces realistic filaments.
// R = density ρ [0+]
// G = velocity_x
// B = velocity_y  
// A = velocity_z
// u_param0 = gravitational strength G
// u_param1 = pressure (resists collapse, Jeans criterion)
// u_param2 = diffusion (numerical viscosity)
// u_param3 = expansion rate (cosmological)
void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_data = fetch(pos);
    float rho = self_data.r;
    vec3 vel = self_data.gba;

    float G_const = u_param0;
    float pressure = u_param1;
    float diffusion = u_param2;
    float expansion = u_param3;

    // Fetch nearest 6 neighbors (reused for Laplacian, gradient, and divergence)
    vec4 xp1 = fetch(pos + ivec3(1,0,0));
    vec4 xm1 = fetch(pos + ivec3(-1,0,0));
    vec4 yp1 = fetch(pos + ivec3(0,1,0));
    vec4 ym1 = fetch(pos + ivec3(0,-1,0));
    vec4 zp1 = fetch(pos + ivec3(0,0,1));
    vec4 zm1 = fetch(pos + ivec3(0,0,-1));

    // Density and velocity Laplacians (resolution-independent)
    float lap_rho = (xp1.r + xm1.r + yp1.r + ym1.r + zp1.r + zm1.r - 6.0 * rho) * h_sq;
    vec3 lap_vel = (xp1.gba + xm1.gba + yp1.gba + ym1.gba + zp1.gba + zm1.gba - 6.0 * vel) * h_sq;

    // Multi-scale gravity: ∇ρ at scales 1, 2, 4 with 1/r weighting
    // Scale offsets scale with resolution for consistent physics
    vec3 grav_force = vec3(0.0);

    // Scale 1 (nearest): weight 1.0
    vec3 grad1 = vec3(xp1.r - xm1.r, yp1.r - ym1.r, zp1.r - zm1.r) * 0.5 * h_inv;
    grav_force += grad1;

    // Scale 2: weight 0.5 (1/r falloff, gradient over wider span)
    int s2 = max(1, int(round(2.0 * h_inv)));
    vec3 grad2 = vec3(
        fetch(pos + ivec3(s2,0,0)).r - fetch(pos + ivec3(-s2,0,0)).r,
        fetch(pos + ivec3(0,s2,0)).r - fetch(pos + ivec3(0,-s2,0)).r,
        fetch(pos + ivec3(0,0,s2)).r - fetch(pos + ivec3(0,0,-s2)).r
    ) * 0.5 / float(s2) * h_inv;
    grav_force += grad2 * 0.5;

    // Scale 4: weight 0.25 (longer range filament formation)
    int s4 = max(1, int(round(4.0 * h_inv)));
    vec3 grad4 = vec3(
        fetch(pos + ivec3(s4,0,0)).r - fetch(pos + ivec3(-s4,0,0)).r,
        fetch(pos + ivec3(0,s4,0)).r - fetch(pos + ivec3(0,-s4,0)).r,
        fetch(pos + ivec3(0,0,s4)).r - fetch(pos + ivec3(0,0,-s4)).r
    ) * 0.5 / float(s4) * h_inv;
    grav_force += grad4 * 0.25;

    grav_force *= G_const * rho;

    // Pressure force: F_press = -P * ∇ρ / ρ (resists compression)
    vec3 press_force = -pressure * grad1 / max(rho, 0.001);

    // Update velocity: gravity pulls mass together, pressure resists
    vec3 new_vel = vel + (grav_force + press_force) * u_dt;

    // Physical viscous dissipation: ν∇²v
    new_vel += diffusion * 0.5 * lap_vel * u_dt;

    // Cosmological expansion: gentle outward push
    vec3 from_center = (vec3(pos) - vec3(u_size) * 0.5) / float(u_size);
    new_vel += expansion * from_center * 0.01 * u_dt;

    // Semi-Lagrangian advection: back-trace density along velocity field
    vec3 departure = vec3(pos) - vel * u_dt;
    float advected_rho = fetch_interp(departure).r;
    float new_rho = advected_rho + diffusion * lap_rho * u_dt;

    new_rho = max(new_rho, 0.001);  // floor density (vacuum energy)
    new_rho = min(new_rho, 5.0);

    // Clamp velocity
    float spd = length(new_vel);
    if (spd > 2.0) new_vel = new_vel / spd * 2.0;

    imageStore(u_dst, pos, vec4(new_rho, new_vel));
}
""",

    "lichen_3d": """
// Lichen / Coral Competition — multiple species compete for space
// R = species A biomass [0,1]
// G = species B biomass [0,1]
// B = nutrient/light resource [0,1]
// A = species C biomass [0,1]
// Three species with different strategies:
//   A: fast grower, low resilience (pioneer)
//   B: slow grower, high resilience (competitor)
//   C: medium growth, mobile (nomad, can spread further)
// u_param0 = growth rate multiplier
// u_param1 = competition strength
// u_param2 = resource regeneration rate
// u_param3 = diffusion (spread rate)
//
// USE_SHARED_MEM=1 loads a 10^3 vec4 tile — all four channels are
// Laplacian-averaged so the full vec4 is the natural unit.

#if USE_SHARED_MEM
#define LCTILE 10
#define LCTILE3 (LCTILE * LCTILE * LCTILE)
shared vec4 s_lc[LCTILE3];
int lc_idx(int x, int y, int z) {
    return z * LCTILE * LCTILE + y * LCTILE + x;
}
#endif

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if USE_SHARED_MEM
    ivec3 local = ivec3(gl_LocalInvocationID);
    int local_flat = int(gl_LocalInvocationIndex);
    ivec3 tile_origin = ivec3(gl_WorkGroupID) * 8 - ivec3(1);
    for (int i = local_flat; i < LCTILE3; i += 512) {
        int tz = i / (LCTILE * LCTILE);
        int ty = (i / LCTILE) % LCTILE;
        int tx = i % LCTILE;
        s_lc[i] = fetch(tile_origin + ivec3(tx, ty, tz));
    }
    barrier();

    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;
    ivec3 tp = local + ivec3(1);
    vec4 self_data = s_lc[lc_idx(tp.x, tp.y, tp.z)];
#else
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_data = fetch(pos);
#endif
    float a = self_data.r;  // species A (pioneer)
    float b = self_data.g;  // species B (competitor)
    float res = self_data.b;
    float c = self_data.a;  // species C (nomad)

    float growth_mult = u_param0;
    float competition = u_param1;
    float regen = u_param2;
    float diffusion = u_param3;

    // Laplacians for each species + resource
    float lap_a = 0.0, lap_b = 0.0, lap_c = 0.0, lap_r = 0.0;
    for (int i = 0; i < 6; i++) {
        ivec3 off = ivec3(0);
        int axis = i / 2; int dir = (i % 2) * 2 - 1;
        off[axis] = dir;
#if USE_SHARED_MEM
        vec4 nb = s_lc[lc_idx(tp.x + off.x, tp.y + off.y, tp.z + off.z)];
#else
        vec4 nb = fetch(pos + off);
#endif
        lap_a += nb.r; lap_b += nb.g; lap_r += nb.b; lap_c += nb.a;
    }
    lap_a -= 6.0 * a; lap_b -= 6.0 * b; lap_r -= 6.0 * res; lap_c -= 6.0 * c;
    lap_a *= h_sq; lap_b *= h_sq; lap_r *= h_sq; lap_c *= h_sq;  // resolution-independent

    float total_bio = a + b + c;
    float space = max(0.0, 1.0 - total_bio);

    // Species A: fast pioneer — grows 3x faster but dies under competition
    float grow_a = growth_mult * 3.0 * a * res * space;
    float die_a = a * (competition * (b * 1.5 + c * 0.5) + 0.02);  // vulnerable to B

    // Species B: slow competitor — grows slowly but dominates in encounters
    float grow_b = growth_mult * 1.0 * b * res * space;
    float die_b = b * (competition * (c * 0.8) + 0.005);  // very resilient, only C threatens

    // Species C: nomad — medium growth, spreads faster (higher diffusion)
    float grow_c = growth_mult * 1.5 * c * res * space;
    float die_c = c * (competition * (a * 0.3 + b * 0.3) + 0.01);

    float new_a = a + (grow_a - die_a) * u_dt + diffusion * lap_a * u_dt;
    float new_b = b + (grow_b - die_b) * u_dt + diffusion * 0.5 * lap_b * u_dt;
    float new_c = c + (grow_c - die_c) * u_dt + diffusion * 2.0 * lap_c * u_dt;

    // Resource: regenerates, consumed by all species
    float consumption = (a * 3.0 + b * 1.0 + c * 1.5) * growth_mult;
    float new_res = res + regen * (1.0 - res) * u_dt
                    - consumption * res * u_dt
                    + diffusion * 0.3 * lap_r * u_dt;

    // Noise for symmetry breaking
    float hash = hash_temporal(pos, 0);
    new_a += (hash - 0.5) * 0.001 * u_dt;

    new_a = clamp(new_a, 0.0, 1.0);
    new_b = clamp(new_b, 0.0, 1.0);
    new_c = clamp(new_c, 0.0, 1.0);
    new_res = clamp(new_res, 0.0, 1.0);

    // Enforce carrying capacity
    float new_total = new_a + new_b + new_c;
    if (new_total > 1.0) {
        new_a /= new_total;
        new_b /= new_total;
        new_c /= new_total;
    }

    imageStore(u_dst, pos, vec4(new_a, new_b, new_res, new_c));
}
""",

    "schrodinger_3d": """
// Time-Dependent Schrödinger Equation in 3D
// Solves: iħ ∂Ψ/∂t = -(ħ²/2m)∇²Ψ + V(r)Ψ
// Split into real/imaginary:
//   ∂ψ_R/∂t = +(ħ/2m)∇²ψ_I - V·ψ_I
//   ∂ψ_I/∂t = -(ħ/2m)∇²ψ_R + V·ψ_R
//
// Channel R = ψ_R  (real part of wavefunction)
// Channel G = ψ_I  (imaginary part of wavefunction)
// Channel B = V(r)  (potential energy, static — set at init)
// Channel A = |Ψ|²  (probability density, computed for rendering)
//
// Yee/FDTD staggered leapfrog for exact probability conservation:
//   ψ_R lives at integer timesteps:      t = 0, dt, 2dt, ...
//   ψ_I lives at half-integer timesteps:  t = dt/2, 3dt/2, ...
//   Even frames: advance ψ_I using H·ψ_R (all ψ_R neighbors available)
//   Odd frames:  advance ψ_R using H·ψ_I (all ψ_I neighbors available)
//
// This preserves the symplectic norm Σ(ψ_R²(n) + ψ_I(n+½)·ψ_I(n-½))
// EXACTLY — no accumulating drift. The naive norm Σ(ψ_R² + ψ_I²)
// oscillates by O(dt) but never grows.

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_val = fetch(pos);
    float psi_r = self_val.r;
    float psi_i = self_val.g;
    float V     = self_val.b;

    float hbar_2m = u_param0;  // ħ/2m kinetic coefficient
    float V_scale = u_param1;  // potential strength multiplier
    float Vs = V * V_scale;

    if (u_frame % 2 == 0) {
        // ── Even frame: advance ψ_I ──
        // ∂ψ_I/∂t = -hbar_2m · ∇²ψ_R + V · ψ_R
        float sum_r = 0.0;
        sum_r += fetch(pos + ivec3( 1, 0, 0)).r;
        sum_r += fetch(pos + ivec3(-1, 0, 0)).r;
        sum_r += fetch(pos + ivec3( 0, 1, 0)).r;
        sum_r += fetch(pos + ivec3( 0,-1, 0)).r;
        sum_r += fetch(pos + ivec3( 0, 0, 1)).r;
        sum_r += fetch(pos + ivec3( 0, 0,-1)).r;
        float lap_r = (sum_r - 6.0 * psi_r) * h_sq;
        psi_i += (-hbar_2m * lap_r + Vs * psi_r) * u_dt;
    } else {
        // ── Odd frame: advance ψ_R ──
        // ∂ψ_R/∂t = +hbar_2m · ∇²ψ_I - V · ψ_I
        float sum_i = 0.0;
        sum_i += fetch(pos + ivec3( 1, 0, 0)).g;
        sum_i += fetch(pos + ivec3(-1, 0, 0)).g;
        sum_i += fetch(pos + ivec3( 0, 1, 0)).g;
        sum_i += fetch(pos + ivec3( 0,-1, 0)).g;
        sum_i += fetch(pos + ivec3( 0, 0, 1)).g;
        sum_i += fetch(pos + ivec3( 0, 0,-1)).g;
        float lap_i = (sum_i - 6.0 * psi_i) * h_sq;
        psi_r += (hbar_2m * lap_i - Vs * psi_i) * u_dt;
    }

    // Safety clamp: Yee leapfrog is symplectic but unbounded if dt violates
    // the von Neumann stability criterion (dt < h²/(2·hbar_2m)). Cap |ψ|
    // to keep visualization and downstream metrics finite when a user
    // pushes parameters past the stability limit.
    psi_r = clamp(psi_r, -1e3, 1e3);
    psi_i = clamp(psi_i, -1e3, 1e3);

    float prob = psi_r * psi_r + psi_i * psi_i;
    imageStore(u_dst, pos, vec4(psi_r, psi_i, V, prob));
}
""",

    "schrodinger_poisson_3d": """
// Schrödinger + Poisson self-consistent field (Hartree mean-field)
// The potential V in channel B is updated each step via Jacobi relaxation
// toward ∇²V = -α|Ψ|², creating self-interaction feedback.
//
// Channel R = ψ_R, G = ψ_I, B = V (updated), A = |Ψ|²
//
// Uses Yee/FDTD leapfrog for the Schrödinger part (symplectic).
// Poisson relaxation runs every frame regardless of even/odd parity.

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_val = fetch(pos);
    float psi_r = self_val.r;
    float psi_i = self_val.g;
    float V     = self_val.b;
    float prob  = self_val.a;

    // ── Poisson relaxation: update V toward ∇²V = -α|Ψ|² ──
    float sum_V = 0.0;
    sum_V += fetch(pos + ivec3( 1, 0, 0)).b;
    sum_V += fetch(pos + ivec3(-1, 0, 0)).b;
    sum_V += fetch(pos + ivec3( 0, 1, 0)).b;
    sum_V += fetch(pos + ivec3( 0,-1, 0)).b;
    sum_V += fetch(pos + ivec3( 0, 0, 1)).b;
    sum_V += fetch(pos + ivec3( 0, 0,-1)).b;

    float alpha = u_param2;  // coupling strength
    // Clamp SOR relaxation to its convergence interval (0, 2). Outside that
    // range Jacobi/SOR diverges. ω=1 is plain Jacobi, ω∈(1,2) accelerates.
    float omega = clamp(u_param3, 0.0, 1.95);

    // Jacobi step for ∇²V = -α|ψ|² discretized as (sum_V - 6V)/h² = -α|ψ|².
    // Solving for V: V = (sum_V + h²·α·|ψ|²) / 6. The h_sq factor is
    // essential for resolution independence — without it the coupling is
    // 4× too weak at size 64 and 4× too strong at size 256.
    float V_jacobi = (sum_V + h_sq * alpha * prob) / 6.0;
    V = mix(V, V_jacobi, omega);

    // ── Schrödinger Yee leapfrog ──
    float hbar_2m = u_param0;
    float V_scale = u_param1;
    float Vs = V * V_scale;

    if (u_frame % 2 == 0) {
        float sum_r = 0.0;
        sum_r += fetch(pos + ivec3( 1, 0, 0)).r;
        sum_r += fetch(pos + ivec3(-1, 0, 0)).r;
        sum_r += fetch(pos + ivec3( 0, 1, 0)).r;
        sum_r += fetch(pos + ivec3( 0,-1, 0)).r;
        sum_r += fetch(pos + ivec3( 0, 0, 1)).r;
        sum_r += fetch(pos + ivec3( 0, 0,-1)).r;
        float lap_r = (sum_r - 6.0 * psi_r) * h_sq;
        psi_i += (-hbar_2m * lap_r + Vs * psi_r) * u_dt;
    } else {
        float sum_i = 0.0;
        sum_i += fetch(pos + ivec3( 1, 0, 0)).g;
        sum_i += fetch(pos + ivec3(-1, 0, 0)).g;
        sum_i += fetch(pos + ivec3( 0, 1, 0)).g;
        sum_i += fetch(pos + ivec3( 0,-1, 0)).g;
        sum_i += fetch(pos + ivec3( 0, 0, 1)).g;
        sum_i += fetch(pos + ivec3( 0, 0,-1)).g;
        float lap_i = (sum_i - 6.0 * psi_i) * h_sq;
        psi_r += (hbar_2m * lap_i - Vs * psi_i) * u_dt;
    }

    psi_r = clamp(psi_r, -1e3, 1e3);
    psi_i = clamp(psi_i, -1e3, 1e3);
    V    = clamp(V,    -1e3, 1e3);

    prob = psi_r * psi_r + psi_i * psi_i;
    imageStore(u_dst, pos, vec4(psi_r, psi_i, V, prob));
}
""",

    "schrodinger_molecule_3d": """
// Two-center molecular Schrödinger equation (Born-Oppenheimer)
// Electron wavefunction evolves in a two-nucleus Coulomb potential:
//   V(r) = -Z/|r - R1| - Z/|r - R2|
// Nuclei are placed symmetrically along x-axis at ±separation/2.
//
// Channel R = ψ_R, G = ψ_I, B = V (two-center Coulomb), A = |Ψ|²
// Uses Yee/FDTD leapfrog (symplectic).

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_val = fetch(pos);
    float psi_r = self_val.r;
    float psi_i = self_val.g;

    float hbar_2m  = u_param0;
    float Z        = u_param1;
    float sep      = u_param2;
    float r_soft   = u_param3;

    // Nuclear positions: centered in grid, separated along x
    float mid = float(u_size) * 0.5;
    float half_sep = sep * 0.5;
    vec3 R1 = vec3(mid - half_sep, mid, mid);
    vec3 R2 = vec3(mid + half_sep, mid, mid);
    vec3 r = vec3(pos) + 0.5;

    // Softened Coulomb: V = -Z/sqrt(|r-R|² + r_soft²)
    float d1 = length(r - R1);
    float d2 = length(r - R2);
    float V = -Z / sqrt(d1 * d1 + r_soft * r_soft)
              -Z / sqrt(d2 * d2 + r_soft * r_soft);

    // Yee leapfrog
    if (u_frame % 2 == 0) {
        float sum_r = 0.0;
        sum_r += fetch(pos + ivec3( 1, 0, 0)).r;
        sum_r += fetch(pos + ivec3(-1, 0, 0)).r;
        sum_r += fetch(pos + ivec3( 0, 1, 0)).r;
        sum_r += fetch(pos + ivec3( 0,-1, 0)).r;
        sum_r += fetch(pos + ivec3( 0, 0, 1)).r;
        sum_r += fetch(pos + ivec3( 0, 0,-1)).r;
        float lap_r = sum_r - 6.0 * psi_r;
        lap_r *= h_sq;
        psi_i += (-hbar_2m * lap_r + V * psi_r) * u_dt;
    } else {
        float sum_i = 0.0;
        sum_i += fetch(pos + ivec3( 1, 0, 0)).g;
        sum_i += fetch(pos + ivec3(-1, 0, 0)).g;
        sum_i += fetch(pos + ivec3( 0, 1, 0)).g;
        sum_i += fetch(pos + ivec3( 0,-1, 0)).g;
        sum_i += fetch(pos + ivec3( 0, 0, 1)).g;
        sum_i += fetch(pos + ivec3( 0, 0,-1)).g;
        float lap_i = sum_i - 6.0 * psi_i;
        lap_i *= h_sq;
        psi_r += (hbar_2m * lap_i - V * psi_i) * u_dt;
    }

    psi_r = clamp(psi_r, -1e3, 1e3);
    psi_i = clamp(psi_i, -1e3, 1e3);

    float prob = psi_r * psi_r + psi_i * psi_i;
    imageStore(u_dst, pos, vec4(psi_r, psi_i, V, prob));
}
""",
}

# ── Element CA: separate compute header with SSBO for element properties ──

ELEMENT_COMPUTE_HEADER = """
#version 430
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

layout(rgba32f, binding=0) uniform image3D u_src;
layout(rgba32f, binding=1) uniform image3D u_dst;

// Element property table: 120 elements x 16 floats each (0=vacuum, 1-118=elements, 119=wall)
// Layout per element:
//   [0] atomic_number  [1] mass      [2] electronegativity  [3] valence_electrons
//   [4] melting_point  [5] boiling_pt [6] density           [7] thermal_cond
//   [8] color_r        [9] color_g    [10] color_b          [11] phase_at_25C
//   [12] group         [13] period    [14] category_id      [15] reserved
layout(std430, binding=2) buffer ElementTable {
    float elements[];  // 120 * 16 floats
};

uniform int u_size;
uniform float u_dt;
uniform float u_param0;  // ambient temperature
uniform float u_param1;  // gravity strength
uniform float u_param2;  // reaction rate multiplier
uniform float u_param3;  // unused
uniform int u_boundary;  // 0 = toroidal (wrap), 1 = clamped (Dirichlet, zero outside), 2 = mirror (Neumann)

// Cell channels:
// R = element ID (0=vacuum, 1-118=element), encoded as float
// G = temperature (°C)
// B = phase (0=solid, 1=liquid, 2=gas)
// A = velocity_y (for gravity/buoyancy)

vec4 fetch(ivec3 p) {
    if (u_boundary == 1) {
        if (any(lessThan(p, ivec3(0))) || any(greaterThanEqual(p, ivec3(u_size))))
            return vec4(0.0);
        return imageLoad(u_src, p);
    }
    if (u_boundary == 2) {
        // Mirror (Neumann zero-flux): treat the wall as identical to its inner neighbor.
        p = clamp(p, ivec3(0), ivec3(u_size - 1));
        return imageLoad(u_src, p);
    }
    p = (p + u_size) % u_size;
    return imageLoad(u_src, p);
}

// Get element property by atomic number and property index
float elem_prop(int z, int prop) {
    if (z < 0 || z > 118) return 0.0;
    return elements[z * 16 + prop];
}

float get_mass(int z)       { return elem_prop(z, 1); }
float get_eneg(int z)       { return elem_prop(z, 2); }
float get_valence(int z)    { return elem_prop(z, 3); }
float get_mp(int z)         { return elem_prop(z, 4); }
float get_bp(int z)         { return elem_prop(z, 5); }
float get_density(int z)    { return elem_prop(z, 6); }
float get_thermal(int z)    { return elem_prop(z, 7); }
float get_category(int z)   { return elem_prop(z, 14); }

// Determine phase from temperature and element properties
float compute_phase(int z, float temp) {
    float mp = get_mp(z);
    float bp = get_bp(z);
    if (temp < mp) return 0.0;       // solid
    if (temp < bp) return 1.0;       // liquid
    return 2.0;                       // gas
}
"""

ELEMENT_CA_RULE = """
// Multi-element CA with real physical properties
// Movement uses matched thresholds: a cell vacates itself at the same threshold
// that a vacuum cell uses to accept an incoming atom, preventing duplication/loss.
void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

    vec4 self_val = fetch(pos);
    int self_id = int(round(self_val.r));
    float self_temp = self_val.g;
    float self_phase = self_val.b;
    float self_vy = self_val.a;

    // Wall element (id=119): indestructible, no physics, just stay put
    if (self_id == 119) {
        imageStore(u_dst, pos, self_val);
        return;
    }

    // Movement threshold — must be the same for sender and receiver
    const float MOVE_THRESHOLD = 0.3;

    // Vacuum stays vacuum (but can be filled by moving atoms)
    if (self_id == 0) {
        // Check if a neighbor wants to move here (gravity, buoyancy, diffusion)

        // Liquid/solid falling from above?
        vec4 above = fetch(pos + ivec3(0, 1, 0));
        int above_id = int(round(above.r));
        float above_phase = above.b;
        float above_vy = above.a;
        if (above_id > 0 && above_phase < 1.5 && above_vy < -MOVE_THRESHOLD) {
            imageStore(u_dst, pos, vec4(float(above_id), above.g, above_phase, above_vy * 0.8));
            return;
        }

        // Gas rising from below?
        vec4 below = fetch(pos + ivec3(0, -1, 0));
        int below_id = int(round(below.r));
        float below_phase = below.b;
        float below_vy = below.a;
        if (below_id > 0 && below_phase > 1.5 && below_vy > MOVE_THRESHOLD) {
            imageStore(u_dst, pos, vec4(float(below_id), below.g, below_phase, below_vy * 0.8));
            return;
        }

        // Liquid spreading sideways: check 4 horizontal neighbors for liquid wanting to spread
        for (int axis = 0; axis < 3; axis += 2) {  // X and Z axes only
            for (int dir = -1; dir <= 1; dir += 2) {
                ivec3 offset = ivec3(0);
                offset[axis] = dir;
                vec4 nb = fetch(pos + offset);
                int nb_id = int(round(nb.r));
                if (nb_id > 0 && nb.b > 0.5 && nb.b < 1.5) {
                    // Liquid neighbor — check if there's liquid above it (hydrostatic pressure)
                    vec4 nb_above = fetch(pos + offset + ivec3(0, 1, 0));
                    if (int(round(nb_above.r)) > 0) {
                        float hash = fract(sin(float(pos.x * 374761 + pos.y * 668265 + pos.z * 928114 + axis * 13) * 0.0001) * 43758.5453);
                        if (hash < 0.2 * u_dt) {
                            imageStore(u_dst, pos, vec4(float(nb_id), nb.g, nb.b, 0.0));
                            return;
                        }
                    }
                }
            }
        }

        // Gas diffusion: check all 6 neighbors for gas-phase atoms
        for (int axis = 0; axis < 3; axis++) {
            for (int dir = -1; dir <= 1; dir += 2) {
                ivec3 offset = ivec3(0);
                offset[axis] = dir;
                vec4 nb = fetch(pos + offset);
                int nb_id = int(round(nb.r));
                if (nb_id > 0 && nb.b > 1.5) {
                    float mass = get_mass(nb_id);
                    float diffuse_rate = 1.0 / max(sqrt(mass), 1.0);
                    float hash = fract(sin(float(pos.x * 374761 + pos.y * 668265 + pos.z * 928114 + axis * 13) * 0.0001) * 43758.5453);
                    if (hash < diffuse_rate * u_dt * 0.3) {
                        imageStore(u_dst, pos, vec4(float(nb_id), nb.g, nb.b, 0.0));
                        return;
                    }
                }
            }
        }

        // Stay vacuum, but conduct ambient temperature
        float ambient = u_param0;
        imageStore(u_dst, pos, vec4(0.0, mix(self_temp, ambient, 0.1 * u_dt), 0.0, 0.0));
        return;
    }

    // Non-vacuum cell: apply physics

    // 1. Thermal conduction — average temperature with neighbors, weighted by thermal conductivity
    float temp_sum = 0.0;
    float weight_sum = 0.0;
    for (int axis = 0; axis < 3; axis++) {
        for (int dir = -1; dir <= 1; dir += 2) {
            ivec3 offset = ivec3(0);
            offset[axis] = dir;
            vec4 nb = fetch(pos + offset);
            int nb_id = int(round(nb.r));
            float k_self = get_thermal(self_id);
            float k_nb = nb_id > 0 ? get_thermal(nb_id) : 0.01;
            float k_avg = (k_self + k_nb) * 0.5;
            // Normalize conductivity to reasonable rate
            float rate = k_avg * 0.001;
            temp_sum += nb.g * rate;
            weight_sum += rate;
        }
    }
    float new_temp = self_temp;
    if (weight_sum > 0.0) {
        new_temp = mix(self_temp, temp_sum / weight_sum, min(u_dt * 0.5, 0.4));
    }

    // 2. Phase transitions
    float new_phase = compute_phase(self_id, new_temp);

    // 3. Gravity and movement
    float new_vy = self_vy;
    float density = get_density(self_id);
    float gravity = u_param1;

    if (new_phase > 0.5) {
        // Liquid or gas: affected by gravity/buoyancy
        if (new_phase > 1.5) {
            // Gas: buoyancy upward
            new_vy += gravity * 0.5 * u_dt;
        } else {
            // Liquid: falls with gravity
            new_vy -= gravity * density * 0.1 * u_dt;
        }
        new_vy = clamp(new_vy, -5.0, 5.0);
        new_vy *= (1.0 - 0.1 * u_dt); // drag

        // Liquid falling: vacate if destination (below) is vacuum
        vec4 below = fetch(pos + ivec3(0, -1, 0));
        int below_id = int(round(below.r));
        if (new_vy < -MOVE_THRESHOLD && below_id == 0 && new_phase < 1.5) {
            imageStore(u_dst, pos, vec4(0.0, new_temp, 0.0, 0.0));
            return;
        }

        // Gas rising: vacate if destination (above) is vacuum
        vec4 above = fetch(pos + ivec3(0, 1, 0));
        int above_id = int(round(above.r));
        if (new_vy > MOVE_THRESHOLD && above_id == 0 && new_phase > 1.5) {
            imageStore(u_dst, pos, vec4(0.0, new_temp, 0.0, 0.0));
            return;
        }

        // Liquid spreading: vacate sideways if pressured from above
        if (new_phase > 0.5 && new_phase < 1.5) {
            vec4 my_above = fetch(pos + ivec3(0, 1, 0));
            if (int(round(my_above.r)) > 0) {
                // Under pressure — try to spread sideways into vacuum
                for (int axis = 0; axis < 3; axis += 2) {
                    for (int dir = -1; dir <= 1; dir += 2) {
                        ivec3 offset = ivec3(0);
                        offset[axis] = dir;
                        vec4 side = fetch(pos + offset);
                        if (int(round(side.r)) == 0) {
                            float hash = fract(sin(float(pos.x * 571 + pos.y * 887 + pos.z * 233 + axis * 17) * 0.0001) * 43758.5453);
                            if (hash < 0.2 * u_dt) {
                                imageStore(u_dst, pos, vec4(0.0, new_temp, 0.0, 0.0));
                                return;
                            }
                        }
                    }
                }
            }
        }
    } else {
        new_vy = 0.0; // Solids don't move
    }

    // 4. Chemical reactions with neighbors
    float react_mult = u_param2;
    for (int axis = 0; axis < 3; axis++) {
        for (int dir = -1; dir <= 1; dir += 2) {
            ivec3 offset = ivec3(0);
            offset[axis] = dir;
            vec4 nb = fetch(pos + offset);
            int nb_id = int(round(nb.r));
            if (nb_id == 0 || nb_id == self_id) continue;

            // Electronegativity difference drives reaction probability
            float en_self = get_eneg(self_id);
            float en_nb = get_eneg(nb_id);
            float en_diff = abs(en_self - en_nb);

            // Higher EN difference = more likely to react
            // Also need sufficient temperature (activation energy ~ mass-weighted)
            float activation = (get_mass(self_id) + get_mass(nb_id)) * 2.0;
            if (en_diff > 0.8 && new_temp > activation * 0.1) {
                float hash = fract(sin(float(pos.x * 127 + pos.y * 311 + pos.z * 523 + axis * 7) * 0.0001) * 43758.5453);
                float react_prob = en_diff * 0.02 * react_mult * u_dt;
                if (hash < react_prob) {
                    // Reaction! Release energy (exothermic if large EN diff)
                    float energy = en_diff * 200.0;
                    new_temp += energy;
                    // Switch phase (reaction heat may cause phase change)
                    new_phase = compute_phase(self_id, new_temp);
                }
            }
        }
    }

    new_temp = clamp(new_temp, -273.15, 10000.0);
    imageStore(u_dst, pos, vec4(float(self_id), new_temp, new_phase, new_vy));
}
"""


# ── Voxel rendering shaders ──────────────────────────────────────────

# GPU-side indirect buffer reset (avoids CPU->GPU write per chunk)
INDIRECT_RESET_SHADER = """
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
void main() {
    vertexCount = 36u;
    instanceCount = 0u;
    firstVertex = 0u;
    baseInstance = 0u;
    if (u_reset_total == 1)
        totalVisibleCount = 0u;
}
"""

# Compute shader: extract visible voxel positions into an SSBO
# A voxel is "visible" if it's alive/non-vacuum AND has at least one dead/vacuum neighbor
# Uses shared memory tiling (10x10x10 halo) for fast neighbor lookups
# and sampler3D (texelFetch) for texture-cached reads
# Packed format: 1 uint (4 bytes) per voxel
#   uint0: x(9) | y(9) | z(9) | solid_neighbors(5)  — supports up to 512³
# Note: prior layout used 10/10/10/2 (1024³ max, only 4 shading levels). The
# 5-bit shading field gives 32 levels, eliminating visible posterization on
# voxel iso-surfaces. The simulator maxes at 512³ grids so 9 bits suffice.
VOXEL_CULL_SHADER = """
#version 430
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

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
uniform float u_threshold;  // visibility threshold for non-element CAs
uniform int u_is_element_ca; // 1 = element mode, 0 = standard mode
uniform int u_max_voxels;
uniform int u_channel;      // which channel to read (0=R, 1=G, 2=B, 3=A)
uniform int u_use_abs;      // 1 = use abs(value) for alive test (wave mode)
// Chunk bounds for multi-pass rendering (default: full grid)
uniform ivec3 u_chunk_min;  // inclusive lower corner
uniform ivec3 u_chunk_max;  // exclusive upper corner

// Shared memory tile: 10x10x10 halo around the 8x8x8 workgroup.
// Only stores the scalar alive-test value (1 float per cell).
shared float s_tile[10][10][10];

float get_alive_value(vec4 c) {
    if (u_is_element_ca == 1) return c.r;
    float v = c[u_channel];
    if (u_use_abs == 1) v = abs(v);
    return v;
}

// Fetch alive-test value, returning 0.0 (dead) for out-of-bounds cells.
// This prevents boundary cells from falsely seeing themselves as neighbors.
float fetch_alive(ivec3 gp) {
    if (any(lessThan(gp, ivec3(0))) || any(greaterThanEqual(gp, ivec3(u_size))))
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
    ivec3 wg_origin = ivec3(gl_WorkGroupID) * 8 + u_chunk_min;

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
    if (any(greaterThanEqual(pos, u_chunk_max))) return;
    if (any(greaterThanEqual(pos, ivec3(u_size)))) return;

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
    // Cap the AO score in the 5-bit field range (0..31)
    uint shade_hint = uint(min(aoc, 31));

    // Append position only (1 uint = 4 bytes per voxel)
    atomicAdd(totalVisibleCount, 1u);  // frame-level total (not reset per chunk)
    uint idx = atomicAdd(instanceCount, 1u);
    if (idx >= uint(u_max_voxels)) return;

    // Pack: 9 bits x, 9 bits y, 9 bits z (supports up to 512³) + 5-bit shading
    voxels[idx] = (uint(pos.x) & 0x1FFu)
                | ((uint(pos.y) & 0x1FFu) <<  9)
                | ((uint(pos.z) & 0x1FFu) << 18)
                | ((shade_hint  & 0x1Fu)  << 27);
}
"""

# ── GPU-side metrics reduction shader ─────────────────────────────
# Computes alive_count, change_count, surface_count, nan_count using
# workgroup-local shared-memory reduction, then one global atomicAdd
# per counter per workgroup.  This avoids millions of global atomics
# that can hang Nouveau/Mesa at large grid sizes.
METRICS_REDUCE_SHADER = """
#version 430
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

layout(rgba32f, binding=0) readonly uniform image3D u_current;
layout(rgba32f, binding=1) readonly uniform image3D u_prev;

layout(std430, binding=5) buffer Metrics {
    uint alive_count;
    uint change_count;
    uint surface_count;
    uint nan_count;
};

uniform int u_size;
uniform float u_threshold;
uniform int u_channel;
uniform int u_mode;        // 0=discrete, 1=continuous, 2=wave, 3=element
uniform float u_change_thr;
uniform int u_has_prev;    // 0 = no previous snapshot, skip activity
uniform int u_boundary;    // 0 = toroidal, 1 = clamped (Dirichlet), 2 = mirror (Neumann)

shared uint s_alive;
shared uint s_change;
shared uint s_surface;
shared uint s_nan;

bool is_alive(vec4 cell) {
    float v = cell[u_channel];
    if (u_mode == 3) return abs(v) > 0.5 && abs(v - 119.0) > 0.5;
    if (u_mode == 2) return abs(v) > u_threshold;
    return v > u_threshold;
}

vec4 safe_load(ivec3 p) {
    if (u_boundary == 1) {
        // Clamped: out-of-bounds reads as dead (zero)
        if (any(lessThan(p, ivec3(0))) || any(greaterThanEqual(p, ivec3(u_size))))
            return vec4(0.0);
        return imageLoad(u_current, p);
    } else if (u_boundary == 2) {
        // Mirror: out-of-bounds reflects to its inner neighbor (zero-flux).
        return imageLoad(u_current, clamp(p, ivec3(0), ivec3(u_size - 1)));
    } else {
        // Toroidal: wrap
        return imageLoad(u_current, (p + ivec3(u_size)) % ivec3(u_size));
    }
}

void main() {
    uint lid = gl_LocalInvocationIndex;

    // Clear shared counters (one thread per workgroup)
    if (lid == 0u) {
        s_alive = 0u; s_change = 0u; s_surface = 0u; s_nan = 0u;
    }
    barrier();

    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (all(lessThan(pos, ivec3(u_size)))) {
        vec4 cur = imageLoad(u_current, pos);
        float val = cur[u_channel];

        // NaN / Inf check
        if (isnan(val) || isinf(val)) {
            atomicAdd(s_nan, 1u);
        } else {
            bool alive = is_alive(cur);

            if (alive) {
                atomicAdd(s_alive, 1u);

                // Surface check: alive cell with < 6 alive von-Neumann neighbours
                int solid = 0;
                if (is_alive(safe_load(pos + ivec3(-1, 0, 0)))) solid++;
                if (is_alive(safe_load(pos + ivec3( 1, 0, 0)))) solid++;
                if (is_alive(safe_load(pos + ivec3( 0,-1, 0)))) solid++;
                if (is_alive(safe_load(pos + ivec3( 0, 1, 0)))) solid++;
                if (is_alive(safe_load(pos + ivec3( 0, 0,-1)))) solid++;
                if (is_alive(safe_load(pos + ivec3( 0, 0, 1)))) solid++;
                if (solid < 6) atomicAdd(s_surface, 1u);
            }

            // Activity: compare with previous snapshot
            if (u_has_prev != 0) {
                vec4 prv = imageLoad(u_prev, pos);
                float prev_val = prv[u_channel];
                float diff = abs(val - prev_val);
                bool changed = diff > u_change_thr;
                if (u_mode == 3) {
                    float temp_diff = abs(cur[1] - prv[1]);
                    changed = changed || temp_diff > 1.0;
                }
                if (changed) atomicAdd(s_change, 1u);
            }
        }
    }

    // Flush workgroup totals to global memory (one atomic per counter)
    barrier();
    if (lid == 0u) {
        if (s_alive > 0u)   atomicAdd(alive_count, s_alive);
        if (s_change > 0u)  atomicAdd(change_count, s_change);
        if (s_surface > 0u) atomicAdd(surface_count, s_surface);
        if (s_nan > 0u)     atomicAdd(nan_count, s_nan);
    }
}
"""

# Vertex shader: instanced cube rendering
VOXEL_VERTEX_SHADER = """
#version 430

// Per-instance data from packed SSBO (1 uint per voxel — position only)
layout(std430, binding=3) buffer VoxelBuffer {
    uint voxels[];
};

// Element property table (for element CA color lookup)
layout(std430, binding=2) buffer ElementTable {
    float elements[];  // 120 * 16 floats
};

uniform sampler3D u_volume_tex;  // 3D volume texture for color sampling
uniform int u_size;
uniform mat4 u_view_proj;
uniform float u_voxel_gap;  // gap between voxels (0 = touching, 0.1 = 10% gap)
uniform int u_is_element_ca;
uniform int u_channel;      // which channel to read for color
uniform int u_use_abs;      // 1 = use abs(value) for wave mode
uniform float u_threshold;  // visibility threshold for value normalization

// Cube vertices: 36 vertices (12 triangles, 6 faces)
const vec3 cube_verts[36] = vec3[36](
    // -Z face
    vec3(0,0,0), vec3(1,0,0), vec3(1,1,0),  vec3(0,0,0), vec3(1,1,0), vec3(0,1,0),
    // +Z face
    vec3(0,0,1), vec3(1,1,1), vec3(1,0,1),  vec3(0,0,1), vec3(0,1,1), vec3(1,1,1),
    // -X face
    vec3(0,0,0), vec3(0,1,0), vec3(0,1,1),  vec3(0,0,0), vec3(0,1,1), vec3(0,0,1),
    // +X face
    vec3(1,0,0), vec3(1,1,1), vec3(1,1,0),  vec3(1,0,0), vec3(1,0,1), vec3(1,1,1),
    // -Y face
    vec3(0,0,0), vec3(0,0,1), vec3(1,0,1),  vec3(0,0,0), vec3(1,0,1), vec3(1,0,0),
    // +Y face
    vec3(0,1,0), vec3(1,1,0), vec3(1,1,1),  vec3(0,1,0), vec3(1,1,1), vec3(0,1,1)
);

const vec3 cube_normals[6] = vec3[6](
    vec3(0,0,-1), vec3(0,0,1), vec3(-1,0,0), vec3(1,0,0), vec3(0,-1,0), vec3(0,1,0)
);

// Face normal directions as ivec3 for neighbor lookup (matches cube_normals order)
const ivec3 face_dirs[6] = ivec3[6](
    ivec3(0,0,-1), ivec3(0,0,1), ivec3(-1,0,0), ivec3(1,0,0), ivec3(0,-1,0), ivec3(0,1,0)
);

out vec3 v_normal;
out vec3 v_world_pos;
out vec3 v_color;
out float v_value;

bool nb_alive(ivec3 p) {
    if (any(lessThan(p, ivec3(0))) || any(greaterThanEqual(p, ivec3(u_size)))) return false;
    vec4 nb = texelFetch(u_volume_tex, p, 0);
    if (u_is_element_ca == 1) return int(round(nb.r)) > 0;
    float v = nb[u_channel];
    if (u_use_abs == 1) v = abs(v);
    return v > u_threshold;
}

void main() {
    int instance_id = gl_InstanceID;
    int vert_id = gl_VertexID;

    // Unpack position from 1 uint (4 bytes) — 9 bits per axis + 5-bit shading
    uint pdata = voxels[instance_id];
    uint shade_hint = (pdata >> 27) & 0x1Fu;  // 0..31, smoothed AO score

    // Sample color from 3D texture at this voxel's grid coordinate (exact texel)
    ivec3 ipos = ivec3(int(pdata & 0x1FFu),
                        int((pdata >>  9) & 0x1FFu),
                        int((pdata >> 18) & 0x1FFu));

    // Per-face visibility: degenerate triangles for faces hidden by a neighbor
    int face_id = vert_id / 6;
    if (nb_alive(ipos + face_dirs[face_id])) {
        gl_Position = vec4(0.0);
        v_normal = vec3(0.0);
        v_world_pos = vec3(0.0);
        v_color = vec3(0.0);
        v_value = 0.0;
        return;
    }

    vec3 cell_pos = vec3(ipos);
    vec4 cell = texelFetch(u_volume_tex, ipos, 0);

    float value;
    vec3 color;

    if (u_is_element_ca == 1) {
        int z = int(round(cell.r));
        value = cell.r;
        // Color from element table
        color = vec3(
            elements[z * 16 + 8],
            elements[z * 16 + 9],
            elements[z * 16 + 10]
        );
        // Tint by temperature
        float temp = cell.g;
        float heat = clamp(temp / 2000.0, 0.0, 1.0);
        color = mix(color, vec3(1.0, 0.3, 0.1), heat * 0.5);
    } else {
        value = cell[u_channel];
        if (u_use_abs == 1) value = abs(value);
        // Normalize value to [0,1] relative to threshold
        value = clamp((value - u_threshold) / max(1.0 - u_threshold, 0.001), 0.0, 1.0);

        // For binary CAs (value ~= 1.0), use the AO-weighted shading hint
        // (5 bits, 0..31) to give 32 smooth darkening levels instead of the
        // previous 4. This eliminates visible posterization on iso-surfaces.
        if (value > 0.99) {
            float sn = float(shade_hint) / 31.0;  // 5-bit hint, 0..1
            // Map so isolated cells (sn ~ 0) are bright (0.85) and deeply
            // embedded cells (sn ~ 1) are dim (0.30). The exponent shapes
            // the response curve to emphasize edges.
            value = mix(0.85, 0.30, pow(sn, 0.7));
        }
        color = vec3(value);
    }

    // Cube vertex
    vec3 local_pos = cube_verts[vert_id];
    vec3 normal = cube_normals[face_id];

    // Scale and position: map grid coords to [0,1]^3 box
    float cell_size = 1.0 / float(u_size);
    float shrink = 1.0 - u_voxel_gap;
    vec3 offset = cell_pos * cell_size + cell_size * 0.5 * (1.0 - shrink);
    vec3 world_pos = offset + local_pos * cell_size * shrink;

    gl_Position = u_view_proj * vec4(world_pos, 1.0);
    v_normal = normal;
    v_world_pos = world_pos;
    v_color = color;
    v_value = value;
}
"""

VOXEL_FRAGMENT_SHADER = """
#version 430

in vec3 v_normal;
in vec3 v_world_pos;
in vec3 v_color;
in float v_value;
out vec4 fragColor;

uniform vec3 u_camera_pos;
uniform float u_brightness;
uniform int u_colormap;
uniform int u_is_element_ca;
uniform float u_alpha;  // transparency (1.0 = opaque)

vec3 colormap_fire(float t) {
    return vec3(clamp(t*3.0, 0.0, 1.0), clamp(t*3.0-1.0, 0.0, 1.0), clamp(t*3.0-2.0, 0.0, 1.0));
}
vec3 colormap_cool(float t) {
    return vec3(sin(t*1.5708)*0.3, t*0.8, 0.5+t*0.5);
}
vec3 colormap_neon(float t) {
    float h = t * 4.0;
    return vec3(clamp(abs(h-2.0)-1.0, 0.0, 1.0), clamp(2.0-abs(h-1.5), 0.0, 1.0), clamp(2.0-abs(h-3.0), 0.0, 1.0)) * (0.5+t*0.5);
}
vec3 colormap_discrete(float t) {
    // 16 maximally-distinct hues via golden-angle spacing.
    // 16 bins (vs the original 8) cuts banding in half on smooth fields.
    int idx = int(floor(t * 16.0));
    idx = clamp(idx, 0, 15);
    float hue = fract(float(idx) * 0.618033988 + 0.0);
    // HSV to RGB (S=0.75, V=0.95)
    float s = 0.75, v = 0.95;
    float c = v * s;
    float h = hue * 6.0;
    float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
    vec3 rgb;
    if      (h < 1.0) rgb = vec3(c, x, 0);
    else if (h < 2.0) rgb = vec3(x, c, 0);
    else if (h < 3.0) rgb = vec3(0, c, x);
    else if (h < 4.0) rgb = vec3(0, x, c);
    else if (h < 5.0) rgb = vec3(x, 0, c);
    else              rgb = vec3(c, 0, x);
    return rgb + vec3(v - c);
}

void main() {
    vec3 color;
    if (u_is_element_ca == 1) {
        color = v_color;  // element color from SSBO
    } else {
        float t = clamp(v_value, 0.0, 1.0);
        if (u_colormap == 0) color = colormap_fire(t);
        else if (u_colormap == 1) color = colormap_cool(t);
        else if (u_colormap == 3) color = colormap_neon(t);
        else if (u_colormap == 4) color = colormap_discrete(t);
        else color = vec3(t);
    }

    // Phong lighting
    vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
    vec3 normal = normalize(v_normal);
    float diffuse = max(dot(normal, light_dir), 0.0);
    float ambient = 0.2;
    vec3 view_dir = normalize(u_camera_pos - v_world_pos);
    float specular = pow(max(dot(reflect(-light_dir, normal), view_dir), 0.0), 32.0);

    vec3 lit = color * (ambient + diffuse * 0.7) * u_brightness + vec3(specular * 0.2);
    fragColor = vec4(lit, u_alpha);
}
"""

# (The previous OCCUPANCY_BUILD_SHADER and MINMAX_BUILD_SHADER were
# replaced by the fused ACCEL_FUSED_SHADER below — they performed two
# separate full passes over the source texture for what is now a
# single-pass dispatch.)

# ── Fused occupancy + min/max compute shader ─────────────────────────
# A single pass over each BLOCK_SIZE³ region that writes BOTH the
# occupancy bitmap (R8UI) and the min/max mipmap (RG16F).
#
# Source is the R16F view texture (channel-select + abs already baked
# in by VIEW_TEX_BUILD_SHADER). This reads 1/8th the bytes of the
# previous RGBA32F direct-read formulation:
#   512³ voxels × 2 B  =  256 MB   (vs 2 GB for RGBA32F)
# so the full accel rebuild (view + occupancy + minmax) costs
# 2 GB + 256 MB instead of the prior 2 × 2 GB. Roughly halves the
# per-sim-step accel bandwidth at large grids.
ACCEL_FUSED_SHADER = """
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
"""

# ── Reduced-precision view texture builder ──────────────────────────
# Bakes the per-voxel visualization scalar (channel select + abs / signed
# remap) into a single-channel R16F texture. The ray-marcher samples THIS
# texture instead of the RGBA32F simulation texture, giving 8× less
# bandwidth per sample and a 4× smaller footprint in the texture cache.
# For a 512³ grid: 256 MB R16F vs 2 × 512 MB RGBA32F ping-pong.
# R16F (vs R8) keeps values outside [0,1] intact with ~3-decimal precision,
# which is plenty for colormaps and iso thresholds.
VIEW_TEX_BUILD_SHADER = """
#version 430
layout(local_size_x=4, local_size_y=4, local_size_z=4) in;

layout(r16f, binding=0) writeonly uniform image3D u_view;
uniform sampler3D u_volume;
uniform int u_size;
uniform int u_channel;
uniform int u_use_abs;   // 0=raw, 1=abs, 2=signed→[0,1]

void main() {
    ivec3 p = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(p, ivec3(u_size)))) return;
    float v = texelFetch(u_volume, p, 0)[u_channel];
    if (u_use_abs == 1)      v = abs(v);
    else if (u_use_abs == 2) v = v * 0.5 + 0.5;
    imageStore(u_view, p, vec4(v, 0.0, 0.0, 0.0));
}
"""

# ── Half-resolution bilateral upsample shader ────────────────────────
# Upsamples a half-res volume render to full res using edge-aware bilateral filter
UPSAMPLE_VERTEX_SHADER = """
#version 430
in vec2 in_pos;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_pos * 0.5 + 0.5;
}
"""

UPSAMPLE_FRAGMENT_SHADER = """
#version 430
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_half_res;    // half-resolution volume render
uniform vec2 u_texel_size;       // 1.0 / half_res_dimensions

void main() {
    // Joint bilateral upsample (5-tap: center + 4 diagonal neighbors at the
    // adjacent half-res texel centers).
    //
    // PRIOR BUG: the 4 corner samples used offsets of ±0.5 * u_texel_size,
    // which sample BETWEEN half-res texels. With LINEAR filtering enabled
    // those reads collapse to bilinear-interpolated values that already
    // include the center, so the bilateral weights had nothing distinct to
    // compare against — the filter degenerated to a plain box blur.
    //
    // The correct neighbor offset on the half-res grid is ±1.0 texel.
    vec2 uv = v_uv;
    vec4 center = texture(u_half_res, uv);

    vec4 s00 = texture(u_half_res, uv + vec2(-1.0, -1.0) * u_texel_size);
    vec4 s10 = texture(u_half_res, uv + vec2( 1.0, -1.0) * u_texel_size);
    vec4 s01 = texture(u_half_res, uv + vec2(-1.0,  1.0) * u_texel_size);
    vec4 s11 = texture(u_half_res, uv + vec2( 1.0,  1.0) * u_texel_size);

    // Range (color) sigma: how perceptually different a sample must be before
    // its weight collapses. 0.08 keeps the filter sharp on iso-surfaces while
    // still smoothing within smooth gradients.
    const float sigma_color = 0.08;
    const float inv_2sig2   = 1.0 / (2.0 * sigma_color * sigma_color);

    // Squared color distance from center (Euclidean in linear RGB).
    vec3 d00 = s00.rgb - center.rgb;
    vec3 d10 = s10.rgb - center.rgb;
    vec3 d01 = s01.rgb - center.rgb;
    vec3 d11 = s11.rgb - center.rgb;

    float w00 = exp(-dot(d00, d00) * inv_2sig2);
    float w10 = exp(-dot(d10, d10) * inv_2sig2);
    float w01 = exp(-dot(d01, d01) * inv_2sig2);
    float w11 = exp(-dot(d11, d11) * inv_2sig2);
    // Center always carries a weight of 1 — guarantees a nonzero denominator
    // and keeps the result anchored on the original sample.
    float w_sum = 1.0 + w00 + w10 + w01 + w11;

    vec3 result = (center.rgb
                 + s00.rgb * w00
                 + s10.rgb * w10
                 + s01.rgb * w01
                 + s11.rgb * w11) / w_sum;

    fragColor = vec4(result, 1.0);
}
"""

# ── Compute-shader raymarcher ────────────────────────────────────────
# Alternative to the fragment shader: raymarches in a compute shader,
# writes to a screen-sized 2D image. Benefits: warp-level early exit,
# shared memory for occupancy, persistent threads.
COMPUTE_RAYMARCH_SHADER = """
#version 430
layout(local_size_x=8, local_size_y=8) in;

layout(rgba8, binding=0) writeonly uniform image2D u_output;

uniform sampler3D u_volume;
uniform int u_size;
uniform vec3 u_camera_pos;
uniform mat3 u_camera_rot;
uniform float u_fov;
uniform float u_density_scale;
uniform float u_brightness;
uniform int u_render_mode;
uniform float u_iso_threshold;
uniform int u_colormap;
uniform int u_vis_channel;
uniform int u_vis_abs;
uniform float u_aspect;
uniform int u_frame_id;
uniform ivec2 u_resolution;

// Occupancy
uniform usampler3D u_occupancy;
uniform int u_occ_size;
uniform int u_use_occupancy;

// Min/max mipmap
uniform sampler3D u_minmax_mip;
uniform int u_minmax_size;
uniform int u_use_minmax;

// Reduced-precision view texture (see FRAGMENT_SHADER for docs)
uniform sampler3D u_view_tex;

float sample_vol(vec3 p) {
    return texture(u_view_tex, p).r;
}

// Same colormaps as fragment shader (duplicated for compute path)
vec3 colormap_fire(float t) {
    return vec3(clamp(t*3.0,0.0,1.0), clamp(t*3.0-1.0,0.0,1.0), clamp(t*3.0-2.0,0.0,1.0));
}
vec3 colormap_cool(float t) {
    return vec3(clamp(sin(t*3.14159*0.5)*0.3,0.0,1.0), clamp(t*0.8,0.0,1.0), clamp(0.5+t*0.5,0.0,1.0));
}
vec3 colormap_neon(float t) {
    float h=t*4.0; vec3 c; c.r=clamp(abs(h-2.0)-1.0,0.0,1.0);
    c.g=clamp(2.0-abs(h-1.5),0.0,1.0); c.b=clamp(2.0-abs(h-3.0),0.0,1.0);
    return c*(0.5+t*0.5);
}
vec3 colormap_discrete(float t) {
    int idx=clamp(int(floor(t*16.0)),0,15); float hue=fract(float(idx)*0.618033988);
    float s=0.75,v=0.95,cc=v*s,h=hue*6.0,x=cc*(1.0-abs(mod(h,2.0)-1.0));
    vec3 rgb; if(h<1.0)rgb=vec3(cc,x,0);else if(h<2.0)rgb=vec3(x,cc,0);
    else if(h<3.0)rgb=vec3(0,cc,x);else if(h<4.0)rgb=vec3(0,x,cc);
    else if(h<5.0)rgb=vec3(x,0,cc);else rgb=vec3(cc,0,x);
    return rgb+vec3(v-cc);
}
vec3 colormap_spectral(float t) {
    float wl=380.0+t*400.0;
    // CIE 1931 2-deg colour-matching functions, fitted as sums of Gaussians
    // (Wyman, Sloan & Shirley 2013). Use squared-z-score form so the GLSL
    // compiler can keep this in fast multiply-add lanes (avoids pow()).
    float a, b, c, d, e, f;
    a=(wl-599.8)/37.9; b=(wl-442.0)/16.0; c=(wl-501.1)/20.4;
    float x_bar=1.056*exp(-0.5*a*a)+0.362*exp(-0.5*b*b)-0.065*exp(-0.5*c*c);
    d=(wl-568.8)/46.9; e=(wl-530.9)/16.3;
    float y_bar=0.821*exp(-0.5*d*d)+0.286*exp(-0.5*e*e);
    float p=(wl-437.0)/11.8; float q=(wl-459.0)/26.0;
    float z_bar=1.217*exp(-0.5*p*p)+0.681*exp(-0.5*q*q);
    vec3 rgb; rgb.r=3.2406*x_bar-1.5372*y_bar-0.4986*z_bar;
    rgb.g=-0.9689*x_bar+1.8758*y_bar+0.0415*z_bar;
    rgb.b=0.0557*x_bar-0.2040*y_bar+1.0570*z_bar;
    float edge=1.0; if(wl<420.0)edge=0.3+0.7*(wl-380.0)/40.0;
    else if(wl>700.0)edge=0.3+0.7*(780.0-wl)/80.0;
    return clamp(rgb*edge,0.0,1.0);
}
vec3 apply_colormap(float t) {
    if(u_colormap==0)return colormap_fire(t);if(u_colormap==1)return colormap_cool(t);
    if(u_colormap==3)return colormap_neon(t);if(u_colormap==4)return colormap_discrete(t);
    if(u_colormap==5)return colormap_spectral(t);return vec3(t);
}

vec2 intersect_box(vec3 ro, vec3 rd) {
    vec3 inv_rd=1.0/rd;
    vec3 t0=(vec3(0.0)-ro)*inv_rd, t1=(vec3(1.0)-ro)*inv_rd;
    vec3 tmin=min(t0,t1), tmax=max(t0,t1);
    return vec2(max(max(tmin.x,tmin.y),tmin.z), min(min(tmax.x,tmax.y),tmax.z));
}

vec3 compute_gradient_opt(vec3 p, float step, float cv) {
    return normalize(vec3(sample_vol(p+vec3(step,0,0))-cv,
                          sample_vol(p+vec3(0,step,0))-cv,
                          sample_vol(p+vec3(0,0,step))-cv));
}

float ign_hash(vec2 pixel, int frame) {
    float angle=float(frame%64)*6.2831853/64.0;
    float ca=cos(angle),sa=sin(angle);
    vec2 r=vec2(ca*pixel.x-sa*pixel.y, sa*pixel.x+ca*pixel.y);
    return fract(52.9829189*fract(0.06711056*r.x+0.00583715*r.y));
}

float adaptive_step(vec3 p) {
    if (u_use_minmax == 0) return 1.0;
    vec2 mm = texture(u_minmax_mip, p).rg;
    if (mm.g < 0.005) return 4.0;
    if (mm.g < 0.02) return 2.0;
    return 1.0;
}

// ── Occupancy DDA (mirrors FRAGMENT_SHADER::skip_empty_blocks) ─────
// Advances t to the next occupied brick along the ray, or t_end if none.
float skip_empty_blocks(vec3 ro, vec3 rd, float t_start, float t_end) {
    if (u_use_occupancy == 0) return t_start;
    float block_size = 1.0 / float(u_occ_size);
    float t = t_start;
    vec3 p = ro + rd * t;
    ivec3 cell = ivec3(clamp(p * float(u_occ_size),
                             vec3(0.0), vec3(float(u_occ_size) - 1.0)));
    if (texelFetch(u_occupancy, cell, 0).r > 0u) return t;

    vec3 inv_rd = 1.0 / rd;
    vec3 step_dir = sign(rd);
    vec3 next_boundary = vec3(cell) + max(step_dir, vec3(0.0));
    vec3 t_max_v = (next_boundary / float(u_occ_size) - ro) * inv_rd;
    vec3 t_delta = abs(block_size * inv_rd);

    int max_occ_steps = u_occ_size * 3;
    for (int i = 0; i < max_occ_steps && t < t_end; i++) {
        if (t_max_v.x < t_max_v.y) {
            if (t_max_v.x < t_max_v.z) {
                t = t_max_v.x; cell.x += int(step_dir.x); t_max_v.x += t_delta.x;
            } else {
                t = t_max_v.z; cell.z += int(step_dir.z); t_max_v.z += t_delta.z;
            }
        } else {
            if (t_max_v.y < t_max_v.z) {
                t = t_max_v.y; cell.y += int(step_dir.y); t_max_v.y += t_delta.y;
            } else {
                t = t_max_v.z; cell.z += int(step_dir.z); t_max_v.z += t_delta.z;
            }
        }
        if (any(lessThan(cell, ivec3(0))) ||
            any(greaterThanEqual(cell, ivec3(u_occ_size)))) return t_end;
        if (texelFetch(u_occupancy, cell, 0).r > 0u)
            return max(t - block_size * 0.5, t_start);
    }
    return t_end;
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= u_resolution.x || pixel.y >= u_resolution.y) return;

    vec2 uv = (vec2(pixel) + 0.5) / vec2(u_resolution);
    vec2 ndc = uv * 2.0 - 1.0;
    vec3 rd = normalize(u_camera_rot * vec3(ndc.x * tan(u_fov*0.5) * u_aspect,
                                            ndc.y * tan(u_fov*0.5), -1.0));
    vec3 ro = u_camera_pos;

    vec2 t_hit = intersect_box(ro, rd);
    if (t_hit.x > t_hit.y || t_hit.y < 0.0) {
        imageStore(u_output, pixel, vec4(0.02, 0.02, 0.04, 1.0));
        return;
    }

    float t_start = max(t_hit.x, 0.0);
    float t_end = t_hit.y;
    float base_step = 1.0 / float(u_size);
    int max_steps = u_size * 3;

    // Jitter
    t_start += ign_hash(vec2(pixel), u_frame_id) * base_step;

    // Occupancy DDA: skip initial empty bricks
    t_start = skip_empty_blocks(ro, rd, t_start, t_end);
    if (t_start >= t_end) {
        imageStore(u_output, pixel, vec4(0.02, 0.02, 0.04, 1.0));
        return;
    }

    if (u_render_mode == 0) {
        // Volume rendering
        vec3 accum_color = vec3(0.0);
        float accum_alpha = 0.0;
        float t = t_start;
        float empty_run = 0.0;

        for (int i = 0; i < max_steps && t < t_end; i++) {
            vec3 p = ro + rd * t;
            float val = sample_vol(p);

            if (val > 0.01) {
                empty_run = 0.0;
                vec3 col = apply_colormap(val);
                float alpha = val * u_density_scale * base_step;
                alpha = min(alpha, 0.95);
                accum_color += col * alpha * (1.0 - accum_alpha) * u_brightness;
                accum_alpha += alpha * (1.0 - accum_alpha);
                if (accum_alpha > 0.98) break;
                t += base_step;
            } else {
                float mult = adaptive_step(p);
                t += base_step * mult;
                empty_run += mult;
                if (empty_run > 8.0 && u_use_occupancy != 0) {
                    t = skip_empty_blocks(ro, rd, t, t_end);
                    empty_run = 0.0;
                }
            }
        }

        imageStore(u_output, pixel, vec4(accum_color + vec3(0.02,0.02,0.04)*(1.0-accum_alpha), 1.0));

    } else if (u_render_mode == 1) {
        // Iso-surface
        float t = t_start;
        float prev_val = 0.0;
        vec3 hit_color = vec3(0.02, 0.02, 0.04);

        for (int i = 0; i < max_steps && t < t_end; i++) {
            vec3 p = ro + rd * t;
            float val = sample_vol(p);

            if (val > u_iso_threshold && prev_val <= u_iso_threshold) {
                float t_lo = t - base_step, t_hi = t;
                for (int b = 0; b < 4; b++) {
                    float t_mid = (t_lo+t_hi)*0.5;
                    if (sample_vol(ro+rd*t_mid) > u_iso_threshold) t_hi=t_mid; else t_lo=t_mid;
                }
                p = ro + rd * t_hi;
                val = sample_vol(p);
                vec3 n = compute_gradient_opt(p, base_step, val);
                vec3 ld = normalize(vec3(0.5,1.0,0.3));
                float diff = max(dot(n,ld),0.0);
                float spec = pow(max(dot(reflect(-ld,n),-rd),0.0),32.0);
                hit_color = apply_colormap(val)*(0.15+diff*0.7)*u_brightness+vec3(spec*0.3);
                break;
            }
            prev_val = val;
            t += (val < u_iso_threshold*0.25) ? base_step*adaptive_step(p) : base_step;
        }

        imageStore(u_output, pixel, vec4(hit_color, 1.0));

    } else {
        // MIP
        float max_val = 0.0;
        float t = t_start;
        for (int i = 0; i < max_steps && t < t_end; i++) {
            vec3 p = ro + rd * t;
            float val = sample_vol(p);
            max_val = max(max_val, val);
            if (max_val > 0.99) break;
            t += (val < 0.01) ? base_step*adaptive_step(p) : base_step;
        }
        imageStore(u_output, pixel, vec4(apply_colormap(max_val)*u_brightness, 1.0));
    }
}
"""

# Ray marching fragment shader for volumetric rendering
VERTEX_SHADER = """
#version 430
in vec2 in_pos;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_pos * 0.5 + 0.5;
}
"""

FRAGMENT_SHADER = """
#version 430
in vec2 v_uv;
out vec4 fragColor;

uniform sampler3D u_volume;
uniform int u_size;
uniform vec3 u_camera_pos;
uniform mat3 u_camera_rot;
uniform float u_fov;
uniform float u_density_scale;
uniform float u_brightness;
uniform int u_render_mode;  // 0=volume, 1=iso_surface, 2=max_intensity
uniform float u_iso_threshold;
uniform float u_slice_pos;   // -1 = disabled, 0..1 = slice position
uniform int u_slice_axis;     // 0=X, 1=Y, 2=Z
uniform int u_colormap;       // 0=fire, 1=cool, 2=grayscale, 3=neon, 5=spectral
uniform int u_vis_channel;    // which RGBA channel to visualize (0-3)
uniform int u_vis_abs;        // 0=direct, 1=abs(value), 2=signed bipolar
uniform float u_aspect;       // viewport width / height
uniform int u_frame_id;       // frame counter for temporal jitter

// ── Occupancy acceleration ──────────────────────────────────────────
uniform usampler3D u_occupancy;     // 3D occupancy bitmap (R8UI, 1=occupied block)
uniform int u_occ_size;             // occupancy grid dimension (size/BLOCK_SIZE)
uniform int u_use_occupancy;        // 0=disabled, 1=enabled

// ── Min/Max mipmap hierarchy ────────────────────────────────────────
uniform sampler3D u_minmax_mip;     // 3D texture: R=min, G=max per 4³ block
uniform int u_minmax_size;          // mipmap grid dimension
uniform int u_use_minmax;           // 0=disabled, 1=enabled

// ── Reduced-precision view texture ──────────────────────────────────
// R16F scalar baked from (u_volume[channel] -> optional abs/signed remap)
// once per sim step. Sampling this is 8× less bandwidth than sampling
// the RGBA32F simulation texture, and eliminates the per-sample branch.
uniform sampler3D u_view_tex;

// ────────────────────────────────────────────────────────────────────
// Sample the selected visualization channel from the volume
float sample_vol(vec3 p) {
    return texture(u_view_tex, p).r;
}

// ── Colormaps ───────────────────────────────────────────────────────
vec3 colormap_fire(float t) {
    return vec3(
        clamp(t * 3.0, 0.0, 1.0),
        clamp(t * 3.0 - 1.0, 0.0, 1.0),
        clamp(t * 3.0 - 2.0, 0.0, 1.0)
    );
}

vec3 colormap_cool(float t) {
    return vec3(
        clamp(sin(t * 3.14159 * 0.5) * 0.3, 0.0, 1.0),
        clamp(t * 0.8, 0.0, 1.0),
        clamp(0.5 + t * 0.5, 0.0, 1.0)
    );
}

vec3 colormap_neon(float t) {
    float h = t * 4.0;
    vec3 c = vec3(0.0);
    c.r = clamp(abs(h - 2.0) - 1.0, 0.0, 1.0);
    c.g = clamp(2.0 - abs(h - 1.5), 0.0, 1.0);
    c.b = clamp(2.0 - abs(h - 3.0), 0.0, 1.0);
    return c * (0.5 + t * 0.5);
}

vec3 colormap_discrete(float t) {
    int idx = int(floor(t * 16.0));
    idx = clamp(idx, 0, 15);
    float hue = fract(float(idx) * 0.618033988 + 0.0);
    float s = 0.75, v = 0.95;
    float cc = v * s;
    float h = hue * 6.0;
    float x = cc * (1.0 - abs(mod(h, 2.0) - 1.0));
    vec3 rgb;
    if      (h < 1.0) rgb = vec3(cc, x, 0);
    else if (h < 2.0) rgb = vec3(x, cc, 0);
    else if (h < 3.0) rgb = vec3(0, cc, x);
    else if (h < 4.0) rgb = vec3(0, x, cc);
    else if (h < 5.0) rgb = vec3(x, 0, cc);
    else              rgb = vec3(cc, 0, x);
    return rgb + vec3(v - cc);
}

// Spectral colormap: physically-based hydrogen emission line colors
// Maps |ψ|² to visible spectrum via CIE-approximate RGB
vec3 colormap_spectral(float t) {
    // Wavelength 380-780nm mapped to [0,1]
    float wl = 380.0 + t * 400.0;
    // Attempt CIE 1931 approximate XYZ → linear sRGB
    float x_bar, y_bar, z_bar;
    // Gaussian approximation of CIE color matching functions
    float a, b, c, d, e, f;
    a = (wl - 599.8) / 37.9; b = (wl - 442.0) / 16.0; c = (wl - 501.1) / 20.4;
    x_bar = 1.056 * exp(-0.5 * a * a)
          + 0.362 * exp(-0.5 * b * b)
          - 0.065 * exp(-0.5 * c * c);
    d = (wl - 568.8) / 46.9; e = (wl - 530.9) / 16.3;
    y_bar = 0.821 * exp(-0.5 * d * d)
          + 0.286 * exp(-0.5 * e * e);
    f = (wl - 437.0) / 11.8; float g = (wl - 459.0) / 26.0;
    z_bar = 1.217 * exp(-0.5 * f * f)
          + 0.681 * exp(-0.5 * g * g);
    // XYZ to linear sRGB
    vec3 rgb;
    rgb.r =  3.2406 * x_bar - 1.5372 * y_bar - 0.4986 * z_bar;
    rgb.g = -0.9689 * x_bar + 1.8758 * y_bar + 0.0415 * z_bar;
    rgb.b =  0.0557 * x_bar - 0.2040 * y_bar + 1.0570 * z_bar;
    // Intensity ramp at edges of visible spectrum
    float edge = 1.0;
    if (wl < 420.0) edge = 0.3 + 0.7 * (wl - 380.0) / 40.0;
    else if (wl > 700.0) edge = 0.3 + 0.7 * (780.0 - wl) / 80.0;
    return clamp(rgb * edge, 0.0, 1.0);
}

vec3 apply_colormap(float t) {
    if (u_colormap == 0) return colormap_fire(t);
    if (u_colormap == 1) return colormap_cool(t);
    if (u_colormap == 3) return colormap_neon(t);
    if (u_colormap == 4) return colormap_discrete(t);
    if (u_colormap == 5) return colormap_spectral(t);
    return vec3(t);  // grayscale
}

// ── Ray-AABB intersection ───────────────────────────────────────────
vec2 intersect_box(vec3 ro, vec3 rd) {
    vec3 inv_rd = 1.0 / rd;
    vec3 t0 = (vec3(0.0) - ro) * inv_rd;
    vec3 t1 = (vec3(1.0) - ro) * inv_rd;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float t_enter = max(max(tmin.x, tmin.y), tmin.z);
    float t_exit = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(t_enter, t_exit);
}

// ── Optimized gradient — forward differences reusing center sample ──
vec3 compute_gradient_opt(vec3 p, float step, float center_val) {
    // 3 texture fetches instead of 6: use center_val from the main loop
    float dx = sample_vol(p + vec3(step, 0, 0)) - center_val;
    float dy = sample_vol(p + vec3(0, step, 0)) - center_val;
    float dz = sample_vol(p + vec3(0, 0, step)) - center_val;
    return normalize(vec3(dx, dy, dz));
}

// ── Blue-noise-quality hash for jitter (Interleaved Gradient Noise) ─
// From Jimenez 2014 (Next Generation Post Processing in Call of Duty)
float ign_hash(vec2 pixel, int frame) {
    // Rotate the IGN pattern each frame for temporal integration
    float angle = float(frame % 64) * 3.14159265 * 2.0 / 64.0;
    float ca = cos(angle), sa = sin(angle);
    vec2 rotated = vec2(ca * pixel.x - sa * pixel.y,
                        sa * pixel.x + ca * pixel.y);
    return fract(52.9829189 * fract(0.06711056 * rotated.x + 0.00583715 * rotated.y));
}

// ── Occupancy DDA: skip empty blocks in large steps ─────────────────
// Returns the t value where the ray enters the next occupied block,
// or t_end if no occupied blocks are found along the ray.
float skip_empty_blocks(vec3 ro, vec3 rd, float t_start, float t_end) {
    if (u_use_occupancy == 0) return t_start;

    float block_size = 1.0 / float(u_occ_size);
    float t = t_start;

    // Check if current position is already in an occupied block
    vec3 p = ro + rd * t;
    ivec3 block = ivec3(clamp(p * float(u_occ_size), vec3(0.0), vec3(float(u_occ_size) - 1.0)));
    uint occ = texelFetch(u_occupancy, block, 0).r;
    if (occ > 0u) return t;

    // DDA through occupancy grid
    vec3 inv_rd = 1.0 / rd;
    vec3 step_dir = sign(rd);
    vec3 pos = p * float(u_occ_size);
    ivec3 cell = ivec3(floor(pos));

    // tMax: how far along the ray to the next cell boundary in each axis
    vec3 next_boundary = vec3(cell) + max(step_dir, vec3(0.0));
    vec3 t_max_v = (next_boundary / float(u_occ_size) - ro) * inv_rd;
    vec3 t_delta = abs(block_size * inv_rd);

    // March through occupancy grid (max: 3 × occ_size steps)
    int max_occ_steps = u_occ_size * 3;
    for (int i = 0; i < max_occ_steps && t < t_end; i++) {
        // Find which axis crosses next
        if (t_max_v.x < t_max_v.y) {
            if (t_max_v.x < t_max_v.z) {
                t = t_max_v.x;
                cell.x += int(step_dir.x);
                t_max_v.x += t_delta.x;
            } else {
                t = t_max_v.z;
                cell.z += int(step_dir.z);
                t_max_v.z += t_delta.z;
            }
        } else {
            if (t_max_v.y < t_max_v.z) {
                t = t_max_v.y;
                cell.y += int(step_dir.y);
                t_max_v.y += t_delta.y;
            } else {
                t = t_max_v.z;
                cell.z += int(step_dir.z);
                t_max_v.z += t_delta.z;
            }
        }

        // Out of grid?
        if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, ivec3(u_occ_size))))
            return t_end;

        // Check occupancy
        occ = texelFetch(u_occupancy, cell, 0).r;
        if (occ > 0u) return max(t - block_size * 0.5, t_start);
    }
    return t_end;
}

// ── Min/max mipmap adaptive stepping ────────────────────────────────
// Returns a safe step multiplier: how many voxels can we skip forward
// knowing the maximum value in the local region is below threshold.
float adaptive_step_multiplier(vec3 p) {
    if (u_use_minmax == 0) return 1.0;

    // Sample the min/max mipmap (R=min, G=max of the 4³ block)
    vec2 mm = texture(u_minmax_mip, p).rg;
    float local_max = mm.g;

    // If the local max is near zero, we can take a large step (4 voxels)
    if (local_max < 0.005) return 4.0;
    // If local max is small, take a 2-voxel step
    if (local_max < 0.02) return 2.0;
    return 1.0;
}

void main() {
    // Ray origin and direction
    vec2 uv = v_uv * 2.0 - 1.0;
    vec3 rd = normalize(u_camera_rot * vec3(uv.x * tan(u_fov * 0.5) * u_aspect,
                                             uv.y * tan(u_fov * 0.5),
                                             -1.0));
    vec3 ro = u_camera_pos;

    // ── Slice mode (early return) ───────────────────────────────────
    if (u_slice_pos >= 0.0) {
        vec3 p = vec3(v_uv, 0.0);
        if (u_slice_axis == 0) p = vec3(u_slice_pos, v_uv);
        else if (u_slice_axis == 1) p = vec3(v_uv.x, u_slice_pos, v_uv.y);
        else p = vec3(v_uv, u_slice_pos);

        float val = sample_vol(p);
        vec3 col = apply_colormap(val);
        fragColor = vec4(col * u_brightness, 1.0);
        return;
    }

    // ── Ray-box intersection ────────────────────────────────────────
    vec2 t_hit = intersect_box(ro, rd);
    if (t_hit.x > t_hit.y || t_hit.y < 0.0) {
        fragColor = vec4(0.02, 0.02, 0.04, 1.0);
        return;
    }

    float t_start = max(t_hit.x, 0.0);
    float t_end = t_hit.y;
    float base_step = 1.0 / float(u_size);
    int max_steps = u_size * 3;

    // ── Interleaved Gradient Noise jitter (breaks banding) ──────────
    vec2 pixel = gl_FragCoord.xy;
    float jitter = ign_hash(pixel, u_frame_id) * base_step;
    t_start += jitter;

    // ── Occupancy DDA: skip initial empty space ─────────────────────
    t_start = skip_empty_blocks(ro, rd, t_start, t_end);
    if (t_start >= t_end) {
        fragColor = vec4(0.02, 0.02, 0.04, 1.0);
        return;
    }

    if (u_render_mode == 0) {
        // ── Volume rendering (emission-absorption) ──────────────────
        vec3 accum_color = vec3(0.0);
        float accum_alpha = 0.0;
        float t = t_start;
        float empty_run = 0.0;

        for (int i = 0; i < max_steps && t < t_end; i++) {
            vec3 p = ro + rd * t;
            float val = sample_vol(p);

            if (val > 0.01) {
                empty_run = 0.0;
                vec3 col = apply_colormap(val);
                float alpha = val * u_density_scale * base_step;
                alpha = min(alpha, 0.95);

                accum_color += col * alpha * (1.0 - accum_alpha) * u_brightness;
                accum_alpha += alpha * (1.0 - accum_alpha);

                if (accum_alpha > 0.98) break;
                t += base_step;
            } else {
                // Adaptive stepping: use min/max mipmap to skip known-empty regions
                float mult = adaptive_step_multiplier(p);
                t += base_step * mult;
                empty_run += mult;
                // If we've been in empty space for a while, try occupancy DDA skip
                if (empty_run > 8.0 && u_use_occupancy != 0) {
                    float t_skip = skip_empty_blocks(ro, rd, t, t_end);
                    t = t_skip;
                    empty_run = 0.0;
                }
            }
        }

        fragColor = vec4(accum_color + vec3(0.02, 0.02, 0.04) * (1.0 - accum_alpha), 1.0);

    } else if (u_render_mode == 1) {
        // ── Iso-surface rendering with optimized gradient ───────────
        float t = t_start;
        float prev_val = 0.0;
        vec3 hit_color = vec3(0.02, 0.02, 0.04);

        for (int i = 0; i < max_steps && t < t_end; i++) {
            vec3 p = ro + rd * t;
            float val = sample_vol(p);

            if (val > u_iso_threshold && prev_val <= u_iso_threshold) {
                // Bisection refinement: 4 steps for sub-voxel accuracy
                float t_lo = t - base_step;
                float t_hi = t;
                for (int b = 0; b < 4; b++) {
                    float t_mid = (t_lo + t_hi) * 0.5;
                    float v_mid = sample_vol(ro + rd * t_mid);
                    if (v_mid > u_iso_threshold) t_hi = t_mid;
                    else t_lo = t_mid;
                }
                p = ro + rd * t_hi;
                val = sample_vol(p);

                // Gradient from 3 forward differences (reuse center val)
                vec3 normal = compute_gradient_opt(p, base_step, val);
                vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
                float diffuse = max(dot(normal, light_dir), 0.0);
                float ambient = 0.15;
                float specular = pow(max(dot(reflect(-light_dir, normal), -rd), 0.0), 32.0);

                vec3 base_col = apply_colormap(val);
                hit_color = base_col * (ambient + diffuse * 0.7) * u_brightness + vec3(specular * 0.3);
                break;
            }
            prev_val = val;
            // Adaptive stepping for iso-surface too
            if (val < u_iso_threshold * 0.25) {
                float mult = adaptive_step_multiplier(p);
                t += base_step * mult;
            } else {
                t += base_step;
            }
        }

        fragColor = vec4(hit_color, 1.0);

    } else {
        // ── Maximum intensity projection with early termination ─────
        float max_val = 0.0;
        float t = t_start;

        for (int i = 0; i < max_steps && t < t_end; i++) {
            vec3 p = ro + rd * t;
            float val = sample_vol(p);
            max_val = max(max_val, val);
            // Early termination: if we hit a fully saturated value, stop
            if (max_val > 0.99) break;
            // Adaptive step in low-value regions
            if (val < 0.01) {
                float mult = adaptive_step_multiplier(p);
                t += base_step * mult;
            } else {
                t += base_step;
            }
        }

        vec3 col = apply_colormap(max_val) * u_brightness;
        fragColor = vec4(col, 1.0);
    }
}
"""


# ── Rule presets ──────────────────────────────────────────────────────

RULE_PRESETS = {
    "game_of_life_3d": {
        "label": "3D Game of Life",
        "shader": "game_of_life_3d",
        "params": {"Birth min": 6, "Birth max": 7, "Survive min": 5, "Survive max": 7},
        "param_ranges": {"Birth min": (0, 26), "Birth max": (0, 26),
                         "Survive min": (0, 26), "Survive max": (0, 26)},
        "dt": 1.0,
        "init": "random_dense",
        "init_variants": ["random_dense", "game_of_life_centered"],
        "description": "Classic discrete Life in 3D. B6-7/S5-7 — stable clusters.",
        "vis_channels": ["Value"],
        "vis_default": 0,
        "vis_abs": False,
        "render_mode": "voxel",
        "boundary": "toroidal",
    },

    "445_rule": {
        "label": "4/4/5 Crystal",
        "shader": "game_of_life_3d",
        "params": {"Birth min": 4, "Birth max": 4, "Survive min": 4, "Survive max": 4},
        "param_ranges": {"Birth min": (3, 6), "Birth max": (3, 7),
                         "Survive min": (3, 6), "Survive max": (3, 8)},
        "dt": 1.0,
        "init": "random_sparse",
        "description": "B4/S4 — strict rule grows diamond-like crystals.",
        "vis_channels": ["Value"],
        "vis_default": 0,
        "vis_abs": False,
        "render_mode": "voxel",
        "boundary": "toroidal",
    },

    "smoothlife_3d": {
        "label": "3D SmoothLife",
        "shader": "smoothlife_3d",
        "params": {"Birth center": 0.22, "Birth range": 0.05,
                   "Survive center": 0.22, "Survive range": 0.08},
        "param_ranges": {"Birth center": (0.05, 0.45), "Birth range": (0.005, 0.20),
                         "Survive center": (0.05, 0.45), "Survive range": (0.005, 0.20)},
        "dt": 0.15,
        "dt_range": (0.01, 0.8),
        "init": "random_smooth",
        "init_variants": ["random_smooth", "smoothlife_sparse"],
        "description": "Continuous analog of Life: cells survive within a density band.",
        "vis_channels": ["Value"],
        "vis_default": 0,
        "vis_abs": False,
        "boundary": "toroidal",
    },
    "reaction_diffusion_3d": {
        "label": "3D Gray-Scott (mitosis)",
        "shader": "reaction_diffusion_3d",
        "params": {"Feed rate": 0.028, "Kill rate": 0.062,
                   "U diffusion": 0.16, "V diffusion": 0.08},
        "param_ranges": {"Feed rate": (0.010, 0.050), "Kill rate": (0.040, 0.072),
                         "U diffusion": (0.05, 0.30), "V diffusion": (0.02, 0.15)},
        "dt": 0.5,
        "dt_range": (0.1, 2.5),
        "init": "gray_scott",
        "description": "Gray-Scott: two coupled species produce spots that divide (mitosis).",
        "vis_channels": ["U (substrate)", "V (catalyst)"],
        "vis_default": 1,
        "vis_abs": False,
        "boundary": "toroidal",
    },

    "gray_scott_worms": {
        "label": "Gray-Scott (worms)",
        "shader": "reaction_diffusion_3d",
        "params": {"Feed rate": 0.046, "Kill rate": 0.063,
                   "U diffusion": 0.16, "V diffusion": 0.04},
        "param_ranges": {"Feed rate": (0.030, 0.060), "Kill rate": (0.050, 0.072),
                         "U diffusion": (0.05, 0.30), "V diffusion": (0.01, 0.08)},
        "dt": 0.5,
        "dt_range": (0.1, 2.5),
        "init": "gray_scott",
        "init_variants": ["gray_scott", "gray_scott_worms_dense"],
        "description": "Gray-Scott worm regime — labyrinthine stripe patterns.",
        "vis_channels": ["U (substrate)", "V (catalyst)"],
        "vis_default": 1,
        "vis_abs": False,
        "boundary": "toroidal",
    },
    "wave_3d": {
        "label": "3D Wave",
        "shader": "wave_3d",
        "params": {"Wave speed": 0.35, "Damping": 0.02, "Drive freq": 0.0, "Drive amp": 0.0},
        "param_ranges": {"Wave speed": (0.05, 2.0), "Damping": (0.0, 0.3),
                         "Drive freq": (0.0, 15.0), "Drive amp": (0.0, 3.0)},
        "dt": 0.15,
        "dt_range": (0.02, 0.5),
        "init": "wave_pulse",
        "description": "3D wave equation — watch spherical waves propagate and reflect.",
        "vis_channels": ["Displacement", "Velocity"],
        "vis_default": 0,
        "vis_abs": True,
        "boundary": "clamped",
    },
    "crystal_growth": {
        "label": "Crystal Growth",
        "shader": "crystal_growth",
        "params": {"Undercooling": 0.3, "Diffusion": 0.15, "Anisotropy strength": 0.3, "Mode": 0},
        "param_ranges": {"Undercooling": (0.05, 0.9), "Diffusion": (0.05, 0.5),
                         "Anisotropy strength": (0.1, 1.0), "Mode": (0, 0)},
        "dt": 0.02,
        "dt_range": (0.005, 0.08),
        "init": "crystal_seed",
        "init_variants": ["crystal_seed", "crystal_multi_seed"],
        "description": "Compact crystal — low anisotropy gives smooth rounded growth front.",
        "vis_channels": ["Phase φ", "Supersaturation"],
        "vis_default": 0,
        "vis_abs": False,
        "render_mode": "voxel",
        "boundary": "clamped",
    },
    "crystal_dendritic": {
        "label": "Crystal (Dendritic)",
        "shader": "crystal_growth",
        "params": {"Undercooling": 0.4, "Diffusion": 0.12, "Anisotropy strength": 1.0, "Mode": 1},
        "param_ranges": {"Undercooling": (0.05, 0.95), "Diffusion": (0.02, 0.5),
                         "Anisotropy strength": (0.3, 3.0), "Mode": (1, 1)},
        "dt": 0.01,
        "dt_range": (0.003, 0.06),
        "init": "crystal_multi_seed",
        "init_variants": ["crystal_multi_seed", "crystal_seed"],
        "description": "Dendritic crystal — Mullins-Sekerka instability produces branching arms.",
        "vis_channels": ["Phase φ", "Supersaturation"],
        "vis_default": 0,
        "vis_abs": False,
        "render_mode": "voxel",
        "boundary": "clamped",
    },
    "crystal_faceted": {
        "label": "Crystal (Faceted)",
        "shader": "crystal_growth",
        "params": {"Undercooling": 0.35, "Diffusion": 0.1, "Anisotropy strength": 2.0, "Mode": 0},
        "param_ranges": {"Undercooling": (0.05, 0.8), "Diffusion": (0.02, 0.5),
                         "Anisotropy strength": (1.0, 5.0), "Mode": (0, 0)},
        "dt": 0.02,
        "dt_range": (0.005, 0.08),
        "init": "crystal_multi_seed",
        "init_variants": ["crystal_multi_seed", "crystal_seed"],
        "description": "Strongly faceted crystal — high anisotropy makes sharp angular edges.",
        "vis_channels": ["Phase φ", "Supersaturation"],
        "vis_default": 0,
        "vis_abs": False,
        "render_mode": "voxel",
        "boundary": "clamped",
    },
    "lenia_3d": {
        "label": "3D Lenia",
        "shader": "lenia_3d",
        "params": {"Growth center": 0.12, "Growth width": 0.03,
                   "Kernel radius": 4.0, "Ring position": 0.5},
        "param_ranges": {"Growth center": (0.005, 0.6), "Growth width": (0.003, 0.15),
                         "Kernel radius": (1.5, 6.0), "Ring position": (0.15, 0.85)},
        "dt": 0.1,
        "dt_range": (0.01, 0.5),
        "init": "lenia_blobs",
        "description": "Continuous CA with kernel-based growth — alien lifeforms.",
        "vis_channels": ["Value"],
        "vis_default": 0,
        "vis_abs": False,
        "boundary": "toroidal",
    },
    "lenia_geminium": {
        "label": "Lenia (Geminium)",
        "shader": "lenia_3d",
        "params": {"Growth center": 0.12, "Growth width": 0.025,
                   "Kernel radius": 7.0, "Ring position": 0.5},
        "param_ranges": {"Growth center": (0.005, 0.6), "Growth width": (0.005, 0.08),
                         "Kernel radius": (2.5, 7.0), "Ring position": (0.15, 0.85)},
        "dt": 0.08,
        "dt_range": (0.01, 0.5),
        "init": "lenia_blobs",
        "description": "Wider growth tolerance + large kernel — fat pulsing creatures that divide and merge.",
        "vis_channels": ["Value"],
        "vis_default": 0,
        "vis_abs": False,
        "boundary": "toroidal",
    },
    "lenia_multi": {
        "label": "Multi-channel Lenia",
        "shader": "lenia_multi_3d",
        "params": {"Growth center": 0.14, "Growth width": 0.025,
                   "Kernel radius": 4.0, "Cross coupling": 0.5},
        "param_ranges": {"Growth center": (0.02, 0.5), "Growth width": (0.003, 0.12),
                         "Kernel radius": (1.5, 6.0), "Cross coupling": (0.0, 2.0)},
        "dt": 0.1,
        "dt_range": (0.01, 0.5),
        "init": "lenia_multi",
        "init_variants": ["lenia_multi", "lenia_multi_colocated"],
        "description": "3-channel Lenia with cross-kernel coupling — creatures with internal organs and differentiation.",
        "vis_channels": ["Channel A", "Channel B", "Channel C", "Activity"],
        "vis_default": 0,
        "vis_abs": False,
        "boundary": "toroidal",
    },
    "predator_prey_3d": {
        "label": "3D Predator-Prey",
        "shader": "predator_prey_3d",
        "params": {"Predation rate": 5.0, "Prey growth": 2.0,
                   "Predator death": 0.3, "Conversion eff": 0.6},
        "param_ranges": {"Predation rate": (1.0, 15.0), "Prey growth": (0.2, 6.0),
                         "Predator death": (0.05, 1.2), "Conversion eff": (0.05, 2.0)},
        "dt": 0.05,
        "dt_range": (0.01, 0.15),
        "init": "predator_prey_separated",
        "init_variants": ["predator_prey_separated", "predator_prey"],
        "description": "Rosenzweig-MacArthur predator-prey — pursuit waves, refugia, boom-bust cycles.",
        "vis_channels": ["Prey", "Predator", "Interaction"],
        "vis_default": 0,
        "vis_abs": False,
        "boundary": "toroidal",
    },
    "kuramoto_3d": {
        "label": "3D Kuramoto Oscillators",
        "shader": "kuramoto_3d",
        "params": {"Coupling K": 0.5, "Noise": 0.02,
                   "Freq scale": 0.1, "Adaptation": 0.5},
        "param_ranges": {"Coupling K": (0.01, 1.5), "Noise": (0.01, 0.8),
                         "Freq scale": (0.05, 2.0), "Adaptation": (0.0, 3.0)},
        "dt": 0.1,
        "dt_range": (0.02, 0.3),
        "init": "kuramoto",
        "init_variants": ["kuramoto", "kuramoto_clusters"],
        "description": "Coupled oscillators synchronize — adaptive frequencies create chimera states.",
        "vis_channels": ["Phase", "Frequency", "Coherence"],
        "vis_default": 0,
        "vis_abs": False,
        "boundary": "toroidal",
    },

    "bz_spiral_waves": {
        "label": "BZ Spiral Waves",
        "shader": "bz_3d",
        "params": {"Alpha (dispersion)": 0.5, "Beta (nonlinear)": -1.0,
                   "Diffusion": 0.4, "Growth (mu)": 1.0},
        "param_ranges": {"Alpha (dispersion)": (0.05, 1.5), "Beta (nonlinear)": (-2.5, 0.0),
                         "Diffusion": (0.05, 1.5), "Growth (mu)": (0.3, 2.5)},
        "dt": 0.05,
        "dt_range": (0.01, 0.15),
        "init": "bz_spiral_seed",
        "init_variants": ["bz_spiral_seed", "bz_reaction"],
        "description": "CGLE in Benjamin-Feir stable regime — clean rotating scroll waves and target patterns.",
        "vis_channels": ["Re(A)", "Im(A)", "Phase"],
        "vis_default": 2,
        "vis_abs": False,
        "boundary": "toroidal",
    },
    "bz_turbulence": {
        "label": "BZ Amplitude Turbulence",
        "shader": "bz_3d",
        "params": {"Alpha (dispersion)": 3.0, "Beta (nonlinear)": -1.5,
                   "Diffusion": 0.2, "Growth (mu)": 1.2},
        "param_ranges": {"Alpha (dispersion)": (1.5, 6.0), "Beta (nonlinear)": (-4.0, -0.5),
                         "Diffusion": (0.02, 1.0), "Growth (mu)": (0.3, 4.0)},
        "dt": 0.03,
        "dt_range": (0.01, 0.1),
        "init": "bz_reaction",
        "init_variants": ["bz_reaction", "bz_spiral_seed", "bz_turbulence_high_amp"],
        "description": "Strongly Benjamin-Feir unstable — amplitude dies to zero, creates defect-mediated turbulence.",
        "vis_channels": ["Re(A)", "Im(A)", "Phase"],
        "vis_default": 2,
        "vis_abs": False,
        "boundary": "toroidal",
    },

    "bz_excitable": {
        "label": "BZ Excitable Medium",
        "shader": "barkley_3d",
        "params": {"Excitability a": 0.75, "Threshold b": 0.06,
                   "Epsilon": 0.05, "Diffusion": 0.3},
        "param_ranges": {"Excitability a": (0.3, 1.5), "Threshold b": (0.0, 0.25),
                         "Epsilon": (0.001, 0.2), "Diffusion": (0.02, 4.0)},
        "dt": 0.05,
        "dt_range": (0.005, 0.15),
        "init": "barkley_excitable",
        "init_variants": ["barkley_excitable", "bz_spiral_seed"],
        "description": "Barkley excitable medium — wavefronts with refractory tails, stochastic nucleation.",
        "vis_channels": ["Excitation u", "Recovery v", "Phase"],
        "vis_default": 0,
        "vis_abs": False,
        "boundary": "toroidal",
    },

    "morphogen_spots": {
        "label": "Morphogen (Spots)",
        "shader": "morphogen_3d",
        "params": {"Da (activator)": 0.01, "Dh (inhibitor)": 0.5,
                   "Reaction": 0.06, "Growth": 0.05},
        "param_ranges": {"Da (activator)": (0.001, 2.0), "Dh (inhibitor)": (0.1, 50.0),
                         "Reaction": (0.005, 0.5), "Growth": (0.002, 0.3)},
        "dt": 0.1,
        "dt_range": (0.1, 3.0),
        "init": "morphogen_hotspots",
        "init_variants": ["morphogen_hotspots", "morphogen"],
        "description": "High inhibitor diffusion (Dh/Da=50) — isolated activator peaks form a regular spotted pattern.",
        "vis_channels": ["Activator", "Inhibitor", "Tissue density"],
        "vis_default": 0,
        "vis_abs": False,
        "boundary": "toroidal",
    },

    "flocking_3d": {
        "label": "3D Flocking",
        "shader": "flocking_3d",
        "params": {"Alignment": 2.0, "Self-propel": 0.3,
                   "Diffusion": 0.0004, "Repulsion": 0.5},
        "param_ranges": {"Alignment": (0.05, 4.0), "Self-propel": (0.02, 2.0),
                         "Diffusion": (0.00005, 0.01), "Repulsion": (0.0, 4.0)},
        "dt": 0.1,
        "dt_range": (0.02, 0.8),
        "init": "flocking",
        "init_variants": ["flocking", "flocking_vortex"],
        "description": "Vicsek-style active matter — swarms, lanes, and vortex mills emerge.",
        "vis_channels": ["Density", "Vel X", "Vel Y", "Vel Z"],
        "vis_default": 0,
        "vis_abs": False,
        "boundary": "toroidal",
    },
    "element_ca": {
        "label": "Element Chemistry",
        "shader": "element_ca",
        "params": {"Temperature": 25.0, "Gravity": 2.0,
                   "Reaction rate": 1.0, "unused_0": 0},
        "param_ranges": {"Temperature": (0.0, 600.0), "Gravity": (0.5, 5.0),
                         "Reaction rate": (0.0, 5.0), "unused_0": (0, 1)},
        "dt": 0.2,
        "init": "element_mix",
        "init_variants": ["element_mix", "element_layered"],
        "description": "Multi-element CA: atoms with real physical properties interact.",
        "vis_channels": ["Element", "Temperature"],
        "vis_default": 0,
        "vis_abs": False,
        "is_element_ca": True,
        "render_mode": "voxel",
        "boundary": "clamped",
    },
    "element_na_water": {
        "label": "Sodium in Water",
        "shader": "element_ca",
        "params": {"Temperature": 25.0, "Gravity": 2.0,
                   "Reaction rate": 2.0, "unused_0": 0},
        "param_ranges": {"Temperature": (-20.0, 150.0), "Gravity": (0.5, 5.0),
                         "Reaction rate": (0.5, 5.0), "unused_0": (0, 1)},
        "dt": 0.2,
        "init": "sodium_water",
        "description": "Drop sodium into water — watch the exothermic reaction!",
        "vis_channels": ["Element", "Temperature"],
        "vis_default": 0,
        "vis_abs": False,
        "is_element_ca": True,
        "render_mode": "voxel",
        "boundary": "clamped",
    },
    "element_metals": {
        "label": "Metal Alloy Melt",
        "shader": "element_ca",
        "params": {"Temperature": 500.0, "Gravity": 2.0,
                   "Reaction rate": 0.5, "unused_0": 0},
        "param_ranges": {"Temperature": (200.0, 1500.0), "Gravity": (0.5, 5.0),
                         "Reaction rate": (0.0, 3.0), "unused_0": (0, 1)},
        "dt": 0.2,
        "init": "metal_layers",
        "description": "Layered metals at high temp — watch lower-MP metals melt first.",
        "vis_channels": ["Element", "Temperature"],
        "vis_default": 0,
        "vis_abs": False,
        "is_element_ca": True,
        "render_mode": "voxel",
        "boundary": "clamped",
    },
    "sandbox": {
        "label": "Sandbox (empty)",
        "shader": "element_ca",
        "params": {"Temperature": 25.0, "Gravity": 2.0,
                   "Reaction rate": 1.0, "unused_0": 0},
        "param_ranges": {"Temperature": (-100.0, 3000.0), "Gravity": (0.0, 10.0),
                         "Reaction rate": (0.0, 5.0), "unused_0": (0, 1)},
        "dt": 0.2,
        "init": "sandbox_empty",
        "description": "Empty sandbox — press B to enter brush mode and build!",
        "vis_channels": ["Element", "Temperature"],
        "vis_default": 0,
        "vis_abs": False,
        "is_element_ca": True,
        "render_mode": "voxel",
        "boundary": "clamped",
    },
    "phase_separation": {
        "label": "Phase Separation",
        "shader": "cahn_hilliard",
        "params": {"Mobility": 0.5, "Epsilon²": 0.2, "Noise": 0.05, "Asymmetry": 0.0},
        "param_ranges": {"Mobility": (0.1, 2.0), "Epsilon²": (0.05, 1.0),
                         "Noise": (0.001, 0.15), "Asymmetry": (-0.15, 0.15)},
        "dt": 0.05,
        "dt_range": (0.01, 0.1),
        "init": "phase_separation",
        "init_variants": ["phase_separation", "phase_separation_quench"],
        "description": "Cahn-Hilliard spinodal decomposition — binary fluid separates into sponge-like phases.",
        "vis_channels": ["Order param", "Chemical potential"],
        "vis_default": 0,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "toroidal",
    },
    "nucleation": {
        "label": "Nucleation (Droplets)",
        "shader": "cahn_hilliard",
        "params": {"Mobility": 0.5, "Epsilon²": 0.2, "Noise": 0.08, "Asymmetry": 0.3},
        "param_ranges": {"Mobility": (0.1, 2.0), "Epsilon²": (0.05, 1.0),
                         "Noise": (0.005, 0.2), "Asymmetry": (0.15, 0.5)},
        "dt": 0.05,
        "dt_range": (0.01, 0.1),
        "init": "phase_separation",
        "description": "Off-critical Cahn-Hilliard — minority phase nucleates as discrete droplets instead of a sponge.",
        "vis_channels": ["Order param", "Chemical potential"],
        "vis_default": 0,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "toroidal",
    },
    "erosion": {
        "label": "Erosion & Sediment",
        "shader": "erosion_3d",
        "params": {"Erosion": 1.0, "Deposition": 0.15, "Diffusion": 0.2, "Gravity": 2.0},
        "param_ranges": {"Erosion": (0.1, 8.0), "Deposition": (0.005, 3.0),
                         "Diffusion": (0.02, 2.0), "Gravity": (0.2, 8.0)},
        "dt": 0.5,
        "dt_range": (0.1, 3.0),
        "init": "erosion_terrain",
        "init_variants": ["erosion_terrain", "erosion_ridges"],
        "description": "Fluid erodes solid terrain, carries sediment, deposits — creates canyons and caves.",
        "vis_channels": ["Solid", "Fluid", "Sediment"],
        "vis_default": 0,
        "vis_abs": False,
        "render_mode": "voxel",
        "boundary": "clamped",
    },
    "mycelium": {
        "label": "Mycelium Network",
        "shader": "mycelium_3d",
        "params": {"Growth": 1.0, "Branching": 0.8, "Consumption": 0.1, "Diffusion": 0.15},
        "param_ranges": {"Growth": (0.01, 2.0), "Branching": (0.0, 2.0),
                         "Consumption": (0.01, 1.0), "Diffusion": (0.01, 0.5)},
        "dt": 0.1,
        "init": "mycelium",
        "init_variants": ["mycelium", "mycelium_foraging"],
        "description": "Fungal hyphal network — tips explore, branch, fuse, and transport nutrients.",
        "vis_channels": ["Biomass", "Nutrient", "Signal"],
        "vis_default": 0,
        "vis_abs": False,
        "render_mode": "voxel",
        "boundary": "clamped",
    },
    "em_wave": {
        "label": "EM Wave",
        "shader": "em_wave_3d",
        "params": {"Wave speed": 1.0, "Damping": 2.0, "Frequency": 3.0, "Amplitude": 0.5},
        "param_ranges": {"Wave speed": (0.05, 5.0), "Damping": (0.0, 15.0),
                         "Frequency": (0.1, 20.0), "Amplitude": (0.01, 5.0)},
        "dt": 0.05,
        "dt_range": (0.01, 0.1),
        "init": "em_wave",
        "description": "Electromagnetic wave propagation — dipole antenna radiates, reflects off conductors.",
        "vis_channels": ["Ez field", "Bx field", "By field"],
        "vis_default": 0,
        "vis_abs": True,
        "render_mode": "volumetric",
        "boundary": "clamped",
    },
    "viscous_fingers": {
        "label": "Viscous Fingers",
        "shader": "viscous_fingers_3d",
        "params": {"Injection": 1.0, "Viscosity ratio": 10.0, "Noise": 0.8, "Surface tension": 0.01},
        "param_ranges": {"Injection": (0.05, 8.0), "Viscosity ratio": (1.0, 100.0),
                         "Noise": (0.0, 4.0), "Surface tension": (0.0, 0.5)},
        "dt": 0.1,
        "dt_range": (0.03, 0.2),
        "init": "viscous_fingers",
        "description": "Saffman-Taylor instability — low-viscosity fluid invades, creating fractal fingers.",
        "vis_channels": ["Saturation", "Pressure", "Permeability", "Interface"],
        "vis_default": 0,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "clamped",
    },
    "fire": {
        "label": "Fire / Combustion",
        "shader": "fire_3d",
        "params": {"Ignition temp": 0.25, "Heat output": 2.0, "Diffusion": 0.4, "Wind": 0.5},
        "param_ranges": {"Ignition temp": (0.05, 0.5), "Heat output": (0.5, 8.0),
                         "Diffusion": (0.05, 0.5), "Wind": (0.0, 3.0)},
        "dt": 0.1,
        "dt_range": (0.2, 1.0),
        "init": "fire",
        "init_variants": ["fire", "fire_sparse"],
        "description": "Combustion front — fire spreads through fuel, heat rises, embers fly.",
        "vis_channels": ["Fuel", "Temperature", "Oxygen", "Embers"],
        "vis_default": 1,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "clamped",
    },
    "physarum": {
        "label": "Slime Mold (Physarum)",
        "shader": "physarum_3d",
        "params": {"Sensor dist": 3.0, "Turn strength": 1.0, "Decay": 0.05, "Diffusion": 0.1},
        "param_ranges": {"Sensor dist": (1.0, 10.0), "Turn strength": (0.1, 5.0),
                         "Decay": (0.01, 0.3), "Diffusion": (0.01, 0.5)},
        "dt": 0.1,
        "dt_range": (0.5, 5.0),
        "init": "physarum",
        "description": "Physarum polycephalum — chemotactic agents form optimal transport networks.",
        "vis_channels": ["Trail", "Agents", "Food"],
        "vis_default": 0,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "toroidal",
    },
    "fracture": {
        "label": "Elastic Fracture",
        "shader": "fracture_3d",
        "params": {"Wave speed": 1.0, "Fracture threshold": 0.5, "Diffusion": 0.2, "Stress": 0.5},
        "param_ranges": {"Wave speed": (0.2, 2.0), "Fracture threshold": (0.5, 1.5),
                         "Diffusion": (0.01, 0.5), "Stress": (0.1, 1.0)},
        "dt": 0.1,
        "dt_range": (0.01, 0.15),
        "init": "fracture",
        "description": "Material fracture — stress concentrates at crack tips, propagates and branches.",
        "vis_channels": ["Displacement", "Stress", "Integrity", "Strain"],
        "vis_default": 2,
        "vis_abs": False,
        "render_mode": "voxel",
        "boundary": "clamped",
    },
    "galaxy": {
        "label": "Galaxy Formation",
        "shader": "galaxy_3d",
        "params": {"Gravity": 5.0, "Pressure": 0.02, "Diffusion": 0.05, "Expansion": 0.0},
        "param_ranges": {"Gravity": (0.2, 15.0), "Pressure": (0.0005, 0.4),
                         "Diffusion": (0.0, 0.5), "Expansion": (-1.0, 1.0)},
        "dt": 0.05,
        "dt_range": (0.1, 2.5),
        "init": "galaxy",
        "init_variants": ["galaxy", "galaxy_filaments"],
        "description": "Self-gravitating density field — Jeans instability forms cosmic web filaments.",
        "vis_channels": ["Density", "Velocity X", "Velocity Y", "Velocity Z"],
        "vis_default": 0,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "toroidal",
    },
    "lichen": {
        "label": "Lichen Competition",
        "shader": "lichen_3d",
        "params": {"Growth": 1.0, "Competition": 1.0, "Regen": 0.3, "Diffusion": 0.1},
        "param_ranges": {"Growth": (0.1, 3.0), "Competition": (0.5, 8.0),
                         "Regen": (0.01, 1.0), "Diffusion": (0.01, 0.5)},
        "dt": 0.1,
        "dt_range": (0.03, 0.15),
        "init": "lichen",
        "init_variants": ["lichen", "lichen_dense"],
        "description": "Three species compete for space — pioneer, competitor, and nomad create territorial mosaic.",
        "vis_channels": ["Species A", "Species B", "Resource", "Species C"],
        "vis_default": 0,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "toroidal",
    },

    # ── Quantum Mechanics CAs ──

    "quantum_hydrogen": {
        "label": "Hydrogen Atom",
        "shader": "schrodinger_3d",
        "default_size": 192,
        "params": {"ħ/2m": 2.5, "V strength": 1.0, "Momentum": 0.0, "Potential": 0.0},
        "param_ranges": {"ħ/2m": (0.1, 10.0), "V strength": (0.0, 5.0),
                         "Momentum": (0.0, 3.0), "Potential": (0.0, 3.0)},
        "dt": 0.02,
        "dt_range": (0.005, 0.1),
        "init": "quantum_hydrogen",
        "init_variants": ["quantum_orbital"],
        "description": "Hydrogen atom — 1s+2p superposition in Coulomb potential. Watch the probability cloud oscillate between sphere and dumbbell.",
        "vis_channels": ["ψ real", "ψ imag", "Potential V", "Probability |Ψ|²"],
        "vis_default": 3,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "clamped",
    },
    "quantum_orbital": {
        "label": "Atomic Orbital",
        "shader": "schrodinger_3d",
        "default_size": 192,
        "params": {"ħ/2m": 2.5, "V strength": 1.0, "Momentum": 0.0, "Potential": 0.0},
        "param_ranges": {"ħ/2m": (0.1, 10.0), "V strength": (0.0, 5.0),
                         "Momentum": (0.0, 3.0), "Potential": (0.0, 3.0)},
        "dt": 0.02,
        "dt_range": (0.005, 0.1),
        "init": "orbital_1s",
        "init_variants": [
            "orbital_1s", "orbital_2s", "orbital_2p0", "orbital_2p1", "orbital_2p-1",
            "orbital_3s", "orbital_3p0", "orbital_3p1",
            "orbital_3d0", "orbital_3d1", "orbital_3d2", "orbital_3d-1", "orbital_3d-2",
            "orbital_4s", "orbital_4p0", "orbital_4d0", "orbital_4f0", "orbital_4f3",
            "quantum_orbital", "quantum_hydrogen",
        ],
        "description": "Hydrogen orbital browser — select any orbital from 1s through 4f. Eigenstates rotate phase; superpositions oscillate.",
        "vis_channels": ["ψ real", "ψ imag", "Potential V", "Probability |Ψ|²"],
        "vis_default": 3,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "clamped",
    },
    "quantum_wavepacket": {
        "label": "Quantum Wavepacket",
        "shader": "schrodinger_3d",
        "default_size": 192,
        "params": {"ħ/2m": 2.5, "V strength": 1.0, "Momentum": 0.0, "Potential": 3.0},
        "param_ranges": {"ħ/2m": (0.1, 10.0), "V strength": (0.0, 5.0),
                         "Momentum": (0.0, 3.0), "Potential": (0.0, 3.0)},
        "dt": 0.02,
        "dt_range": (0.005, 0.1),
        "init": "quantum_wavepacket",
        "description": "Gaussian wavepacket in a box — disperses, bounces, interferes with itself.",
        "vis_channels": ["ψ real", "ψ imag", "Potential V", "Probability |Ψ|²"],
        "vis_default": 3,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "clamped",
    },
    "quantum_harmonic": {
        "label": "Quantum Oscillator",
        "shader": "schrodinger_3d",
        "default_size": 192,
        "params": {"ħ/2m": 2.5, "V strength": 1.0, "Momentum": 0.0, "Potential": 1.0},
        "param_ranges": {"ħ/2m": (0.1, 10.0), "V strength": (0.0, 5.0),
                         "Momentum": (0.0, 3.0), "Potential": (0.0, 3.0)},
        "dt": 0.02,
        "dt_range": (0.005, 0.1),
        "init": "quantum_harmonic",
        "description": "Coherent state in harmonic trap — oscillates like a classical particle, showing quantum-classical correspondence.",
        "vis_channels": ["ψ real", "ψ imag", "Potential V", "Probability |Ψ|²"],
        "vis_default": 3,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "clamped",
    },
    "quantum_tunneling": {
        "label": "Quantum Tunneling",
        "shader": "schrodinger_3d",
        "default_size": 192,
        "params": {"ħ/2m": 2.5, "V strength": 1.0, "Momentum": 0.0, "Potential": 3.0},
        "param_ranges": {"ħ/2m": (0.1, 10.0), "V strength": (0.0, 10.0),
                         "Momentum": (0.0, 3.0), "Potential": (0.0, 3.0)},
        "dt": 0.02,
        "dt_range": (0.005, 0.1),
        "init": "quantum_tunneling",
        "description": "Wavepacket hits a potential barrier — part tunnels through, part reflects. Classically impossible.",
        "vis_channels": ["ψ real", "ψ imag", "Potential V", "Probability |Ψ|²"],
        "vis_default": 3,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "clamped",
    },
    "quantum_double_slit": {
        "label": "Double Slit",
        "shader": "schrodinger_3d",
        "default_size": 192,
        "params": {"ħ/2m": 2.5, "V strength": 1.0, "Momentum": 0.0, "Potential": 3.0},
        "param_ranges": {"ħ/2m": (0.1, 10.0), "V strength": (0.0, 10.0),
                         "Momentum": (0.0, 3.0), "Potential": (0.0, 3.0)},
        "dt": 0.02,
        "dt_range": (0.005, 0.1),
        "init": "quantum_double_slit",
        "description": "Wavepacket through a double slit — the defining quantum experiment. Watch interference fringes form.",
        "vis_channels": ["ψ real", "ψ imag", "Potential V", "Probability |Ψ|²"],
        "vis_default": 3,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "clamped",
    },
    "quantum_molecule": {
        "label": "Molecular Bond",
        "shader": "schrodinger_molecule_3d",
        "default_size": 192,
        "params": {"ħ/2m": 2.5, "Nuclear charge": 1.0, "Separation": 8.0, "Softening": 1.5},
        "param_ranges": {"ħ/2m": (0.1, 10.0), "Nuclear charge": (0.5, 4.0),
                         "Separation": (2.0, 20.0), "Softening": (0.5, 4.0)},
        "dt": 0.02,
        "dt_range": (0.005, 0.1),
        "init": "quantum_molecule",
        "init_variants": ["quantum_antibonding"],
        "description": "Electron in two-nucleus potential — watch bonding orbital form. Adjust separation to see bond form and break.",
        "vis_channels": ["ψ real", "ψ imag", "Potential V", "Probability |Ψ|²"],
        "vis_default": 3,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "clamped",
    },
    "quantum_antibonding": {
        "label": "Antibonding Orbital",
        "shader": "schrodinger_molecule_3d",
        "default_size": 192,
        "params": {"ħ/2m": 2.5, "Nuclear charge": 1.0, "Separation": 8.0, "Softening": 1.5},
        "param_ranges": {"ħ/2m": (0.1, 10.0), "Nuclear charge": (0.5, 4.0),
                         "Separation": (2.0, 20.0), "Softening": (0.5, 4.0)},
        "dt": 0.02,
        "dt_range": (0.005, 0.1),
        "init": "quantum_antibonding",
        "init_variants": ["quantum_molecule"],
        "description": "Antibonding molecular orbital — probability density avoids the internuclear midpoint (node).",
        "vis_channels": ["ψ real", "ψ imag", "Potential V", "Probability |Ψ|²"],
        "vis_default": 3,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "clamped",
    },
    "quantum_selfinteract": {
        "label": "Self-Interacting QM",
        "shader": "schrodinger_poisson_3d",
        "default_size": 192,
        "params": {"ħ/2m": 2.5, "V strength": 1.0, "Coupling α": 5.0, "Relaxation ω": 0.8},
        "param_ranges": {"ħ/2m": (0.1, 10.0), "V strength": (0.0, 5.0),
                         "Coupling α": (0.1, 50.0), "Relaxation ω": (0.1, 1.5)},
        "dt": 0.02,
        "dt_range": (0.005, 0.1),
        "init": "quantum_selfinteract",
        "description": "Schrödinger-Poisson — wavefunction generates its own potential. Simplified Hartree mean-field model.",
        "vis_channels": ["ψ real", "ψ imag", "Potential V", "Probability |Ψ|²"],
        "vis_default": 3,
        "vis_abs": False,
        "render_mode": "volumetric",
        "boundary": "clamped",
    },
}


# ── Initialization patterns (all return 4-channel: size x size x size x 4) ──

# All init functions generate random fields at a fixed canonical resolution
# (CANONICAL_INIT_SIZE³), then upsample to the target size. This ensures the
# same seed produces the same spatial pattern regardless of grid resolution.
CANONICAL_INIT_SIZE = 64

def _canonical_noise(size, rng, low=0.0, high=1.0):
    """Generate a 3D random field at canonical res, upsample to target size.
    Always draws the same number of random values regardless of target size."""
    from scipy.ndimage import zoom
    cs = CANONICAL_INIT_SIZE
    small = rng.uniform(low, high, (cs, cs, cs)).astype(np.float32)
    if size == cs:
        return small
    factor = size / cs
    return zoom(small, factor, order=1).astype(np.float32)

def _canonical_randint(size, rng, choices):
    """Generate a 3D field of integer choices at canonical res, upsample (nearest)."""
    from scipy.ndimage import zoom
    cs = CANONICAL_INIT_SIZE
    small = rng.choice(choices, size=(cs, cs, cs)).astype(np.float32)
    if size == cs:
        return small
    factor = size / cs
    return np.round(zoom(small, factor, order=0)).astype(np.float32)

def init_random_very_sparse(size, rng):
    """Random with ~3% density — for crystal rules that grow."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    data[:, :, :, 0] = (_canonical_noise(size, rng) < 0.03).astype(np.float32)
    return data

def init_random_sparse(size, rng):
    """Random with ~10% density."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    data[:, :, :, 0] = (_canonical_noise(size, rng) < 0.10).astype(np.float32)
    return data

def init_random_dense(size, rng):
    """Random with ~40% density."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    data[:, :, :, 0] = (_canonical_noise(size, rng) < 0.40).astype(np.float32)
    return data

def init_random_smooth(size, rng):
    """Smooth random blobs."""
    # Generate + blur at canonical res, then upsample (inherently smooth)
    from scipy.ndimage import zoom
    cs = CANONICAL_INIT_SIZE
    field = rng.random((cs, cs, cs)).astype(np.float32) * 0.5
    for _ in range(3):
        padded = np.pad(field, 1, mode='wrap')
        field = (padded[:-2, 1:-1, 1:-1] + padded[2:, 1:-1, 1:-1] +
                 padded[1:-1, :-2, 1:-1] + padded[1:-1, 2:, 1:-1] +
                 padded[1:-1, 1:-1, :-2] + padded[1:-1, 1:-1, 2:] +
                 padded[1:-1, 1:-1, 1:-1]) / 7.0
    if size != cs:
        field = zoom(field, size / cs, order=1).astype(np.float32)
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    data[:, :, :, 0] = field
    return data

def init_center_blob(size, rng):
    """Gaussian blob at center."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    c = size / 2.0
    r = size / 8.0
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    dist = np.sqrt((x - c)**2 + (y - c)**2 + (z - c)**2)
    data[:, :, :, 0] = np.exp(-0.5 * (dist / (r * 0.3))**2).astype(np.float32)
    return data

def init_crystal_seed(size, rng):
    """Phase-field crystal seed: φ=1 solid seed(s), supersaturation field in liquid."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    # R = phase field: start liquid (φ≈0) with small solid seeds
    data[:, :, :, 0] = 0.0
    # G = supersaturation: uniform in liquid, zero in solid
    data[:, :, :, 1] = 0.3 + _canonical_noise(size, rng, -0.02, 0.02)
    # Place 1-3 solid seeds
    n_seeds = rng.randint(1, 4)
    for _ in range(n_seeds):
        fx = 0.5 + rng.uniform(-1.0/6, 1.0/6)
        fy = 0.5 + rng.uniform(-1.0/6, 1.0/6)
        fz = 0.5 + rng.uniform(-1.0/6, 1.0/6)
        fr = rng.uniform(1.0 / size, 2.0 / size)
        cx, cy, cz, r = fx * size, fy * size, fz * size, max(1, fr * size)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        mask = dist <= r
        data[:, :, :, 0][mask] = 1.0  # solid
        data[:, :, :, 1][mask] = 0.0  # supersaturation consumed
    return data

def init_wave_pulse(size, rng):
    """Wave equation: 1-3 Gaussian pulses at random positions, varying widths."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    n_pulses = rng.randint(1, 4)  # 1-3 pulses
    for _ in range(n_pulses):
        cx = rng.uniform(size * 0.2, size * 0.8)
        cy = rng.uniform(size * 0.2, size * 0.8)
        cz = rng.uniform(size * 0.2, size * 0.8)
        r = rng.uniform(size * 0.03, size * 0.12)
        amp = rng.uniform(0.5, 1.5)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        data[:, :, :, 0] += (amp * np.exp(-0.5 * (dist / r)**2)).astype(np.float32)
    # Optionally give some pulses initial velocity (not purely displacement)
    if rng.random() < 0.5:
        vr = rng.uniform(size * 0.05, size * 0.15)
        vcx = rng.uniform(size * 0.2, size * 0.8)
        vcy = rng.uniform(size * 0.2, size * 0.8)
        vcz = rng.uniform(size * 0.2, size * 0.8)
        vdist = np.sqrt((x - vcx)**2 + (y - vcy)**2 + (z - vcz)**2)
        data[:, :, :, 1] = (rng.uniform(-1.0, 1.0) * np.exp(-0.5 * (vdist / vr)**2)).astype(np.float32)
    return data

def init_gray_scott(size, rng):
    """Gray-Scott: U=1 everywhere, V seeded in irregular noise-thresholded clusters.
    Irregular splotches instead of spherical patches — the Turing instability
    develops varied pattern orientations immediately rather than expanding as
    a boring ring that slowly breaks symmetry."""
    from scipy.ndimage import gaussian_filter
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    data[:, :, :, 0] = 1.0  # U = 1 everywhere

    # Generate smooth noise field — heavy blur creates large connected
    # clusters when thresholded, producing irregular organic splotches
    cs = CANONICAL_INIT_SIZE
    raw = rng.random((cs, cs, cs)).astype(np.float32)
    blurred = gaussian_filter(raw, sigma=3.0)
    from scipy.ndimage import zoom
    if size != cs:
        blurred = zoom(blurred, size / cs, order=1).astype(np.float32)

    # Threshold to create irregular splotches (~15% volume coverage)
    threshold = np.percentile(blurred, 85)
    mask = blurred > threshold

    # Concentration variation within clusters (breaks any residual symmetry)
    variation = _canonical_noise(size, rng, 0.15, 0.30)
    data[:, :, :, 0][mask] = 0.5
    data[:, :, :, 1][mask] = variation[mask]

    # Low-level noise everywhere for additional symmetry breaking
    noise = _canonical_noise(size, rng, 0.0, 0.01)
    data[:, :, :, 1] += noise

    return data


def init_predator_prey(size, rng):
    """Rosenzweig-MacArthur predator-prey: patchy prey and predator populations.
    Prey patches at carrying capacity with predator clusters trailing behind.
    Spatial heterogeneity triggers traveling pursuit waves and boom-bust cycles."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    mid = size / 2.0

    # Start with prey near carrying capacity everywhere + noise
    prey = 0.8 + _canonical_noise(size, rng, -0.2, 0.2)

    # Add 2-4 predator clusters (scattered hunting packs)
    n_packs = rng.randint(2, 5)
    predator = np.zeros((size, size, size), dtype=np.float32)
    for _ in range(n_packs):
        cx = rng.uniform(size * 0.15, size * 0.85)
        cy = rng.uniform(size * 0.15, size * 0.85)
        cz = rng.uniform(size * 0.15, size * 0.85)
        r = rng.uniform(size * 0.08, size * 0.2)
        amp = rng.uniform(0.3, 0.8)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        predator += (amp * np.exp(-0.5 * (dist / r)**2)).astype(np.float32)

    # Add a few prey-depleted zones (recently predated areas)
    n_gaps = rng.randint(1, 3)
    for _ in range(n_gaps):
        cx = rng.uniform(size * 0.2, size * 0.8)
        cy = rng.uniform(size * 0.2, size * 0.8)
        cz = rng.uniform(size * 0.2, size * 0.8)
        r = rng.uniform(size * 0.06, size * 0.15)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        prey -= (0.6 * np.exp(-0.5 * (dist / r)**2)).astype(np.float32)

    data[:, :, :, 0] = np.clip(prey, 0.05, 1.0).astype(np.float32)
    data[:, :, :, 1] = np.clip(predator, 0.0, 1.0).astype(np.float32)

    # Small noise for symmetry breaking
    data[:, :, :, 0] += _canonical_noise(size, rng, 0.0, 0.02)
    data[:, :, :, 1] += _canonical_noise(size, rng, 0.0, 0.01)
    data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0, 1)
    data[:, :, :, 1] = np.clip(data[:, :, :, 1], 0, 1)
    return data


def init_kuramoto(size, rng):
    """Kuramoto: random phases and spatially-varying natural frequencies."""
    from scipy.ndimage import zoom
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    # Random initial phase [0, 1) (represents 0 to 2*pi)
    data[:, :, :, 0] = _canonical_noise(size, rng)
    # Natural frequency: smooth spatial variation (creates interesting domain walls)
    cs = CANONICAL_INIT_SIZE
    freq = rng.random((cs, cs, cs)).astype(np.float32)
    for _ in range(3):
        padded = np.pad(freq, 1, mode='wrap')
        freq = (padded[:-2, 1:-1, 1:-1] + padded[2:, 1:-1, 1:-1] +
                 padded[1:-1, :-2, 1:-1] + padded[1:-1, 2:, 1:-1] +
                 padded[1:-1, 1:-1, :-2] + padded[1:-1, 1:-1, 2:] +
                 padded[1:-1, 1:-1, 1:-1]) / 7.0
    if size != cs:
        freq = zoom(freq, size / cs, order=1).astype(np.float32)
    data[:, :, :, 1] = freq
    return data

def init_bz_reaction(size, rng):
    """BZ/CGLE init: random complex field near the limit cycle |A|=1.
    Start with amplitude ~1 and random phase, with small perturbations.
    The Benjamin-Feir instability breaks this into spiral defect chaos."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)

    # Random phase everywhere [0, 2pi)
    phase = _canonical_noise(size, rng, 0.0, 2 * np.pi)

    # Amplitude near the limit cycle |A| = sqrt(mu) ≈ 1
    # with 10% perturbation to trigger instability
    amp = 1.0 + _canonical_noise(size, rng, -0.1, 0.1)

    data[:, :, :, 0] = (amp * np.cos(phase)).astype(np.float32)  # Re(A)
    data[:, :, :, 1] = (amp * np.sin(phase)).astype(np.float32)  # Im(A)
    data[:, :, :, 2] = (phase / (2 * np.pi)).astype(np.float32)  # Phase for vis
    return data


def init_barkley_excitable(size, rng):
    """Barkley excitable medium: resting state with a few excitation seeds.
    Seeds trigger propagating wavefronts; collisions and refractory zones
    create complex scroll wave dynamics."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    # Resting state: u=0, v=0 (excitable, ready to fire)
    # Place a few excited regions as seeds for wave nucleation
    n_seeds = rng.randint(3, 8)
    for _ in range(n_seeds):
        cx = rng.uniform(size * 0.1, size * 0.9)
        cy = rng.uniform(size * 0.1, size * 0.9)
        cz = rng.uniform(size * 0.1, size * 0.9)
        r = rng.uniform(size * 0.04, size * 0.1)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        excited = (dist < r).astype(np.float32)
        data[:, :, :, 0] = np.maximum(data[:, :, :, 0], excited)
    # Add a half-plane refractory region to create scroll wave seeds
    # (excitation + perpendicular refractory → scroll wave)
    plane_pos = rng.uniform(0.3, 0.7) * size
    plane_axis = rng.randint(0, 3)
    if plane_axis == 0:
        mask = x > plane_pos
    elif plane_axis == 1:
        mask = y > plane_pos
    else:
        mask = z > plane_pos
    data[:, :, :, 1] = np.where(mask & (data[:, :, :, 0] > 0.5), 0.5, data[:, :, :, 1])
    return data


def init_morphogen(size, rng):
    """Morphogenesis: uniform tissue with random activator perturbations.
    Start near the homogeneous steady state so Turing instability can grow."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    # Approximate GM steady state: a0 ≈ sigma_a/mu_a + correction ≈ 0.1
    # h0 ≈ a0^2/mu_h ≈ 0.01. But we keep it modest to let patterns emerge.
    a0, h0 = 0.2, 0.04  # near but not at equilibrium
    data[:, :, :, 0] = a0
    data[:, :, :, 1] = h0
    data[:, :, :, 2] = 0.5  # tissue density (moderate)
    # Random perturbations to activator to trigger Turing instability
    data[:, :, :, 0] += _canonical_noise(size, rng, -0.05, 0.05)
    data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0.01, None)
    return data


def init_flocking(size, rng):
    """Flocking: elongated density streaks with locally coherent velocities.
    Real flocks form streams and filaments, not spherical blobs. Each streak
    has a coherent bulk velocity so the alignment interaction is visible
    immediately — streaks merge, deflect, and form vortex mills."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    n_streaks = rng.randint(8, 14)
    for _ in range(n_streaks):
        # Random center
        cx = rng.uniform(0.05, 0.95) * size
        cy = rng.uniform(0.05, 0.95) * size
        cz = rng.uniform(0.05, 0.95) * size
        # Random elongation direction
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(-np.pi / 2, np.pi / 2)
        dx_dir = np.cos(phi) * np.cos(theta)
        dy_dir = np.cos(phi) * np.sin(theta)
        dz_dir = np.sin(phi)
        # Displacement from center
        rx = (x - cx).astype(np.float64)
        ry = (y - cy).astype(np.float64)
        rz = (z - cz).astype(np.float64)
        # Project along and perpendicular to streak direction
        along = rx * dx_dir + ry * dy_dir + rz * dz_dir
        perp_sq = np.maximum(rx**2 + ry**2 + rz**2 - along**2, 0.0)
        # Elongated Gaussian: long along direction, narrow perpendicular
        long_r = max(3, size // 6)
        short_r = max(2, size // 16)
        density = np.exp(-0.5 * (along / long_r)**2
                         - 0.5 * perp_sq / short_r**2).astype(np.float32)
        density *= rng.uniform(0.3, 0.6)
        data[:, :, :, 0] += density
        # Coherent velocity along streak direction (scaled by local density)
        speed = rng.uniform(0.1, 0.3)
        data[:, :, :, 1] += density * dx_dir * speed
        data[:, :, :, 2] += density * dy_dir * speed
        data[:, :, :, 3] += density * dz_dir * speed
    data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0, 1)
    return data


def init_element_mix(size, rng):
    """Random mix of common elements: C, N, O, Na, Fe, Cu, Au + vacuum."""
    # R=element_id, G=temperature, B=phase, A=velocity
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    common = [0, 0, 0, 0, 6, 7, 8, 11, 26, 29, 79]  # lots of vacuum + elements
    # Melting points for phase determination at 25°C
    mp_lookup = {6: 3550, 7: -210, 8: -219, 11: 98, 26: 1538, 29: 1085, 79: 1064}

    ids = _canonical_randint(size, rng, common)
    data[:, :, :, 0] = ids
    data[:, :, :, 1] = 25.0  # room temperature

    # Compute phase for each cell
    for z_id, mp in mp_lookup.items():
        mask = ids == z_id
        if mp > 25:
            data[:, :, :, 2][mask] = 0.0  # solid
        elif z_id in (7, 8):  # N, O: gas at room temp
            data[:, :, :, 2][mask] = 2.0  # gas
        else:
            data[:, :, :, 2][mask] = 1.0  # liquid

    return data


def init_sodium_water(size, rng):
    """Pool of water (H+O) at bottom, sodium chunk dropped from top."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    half = size // 2

    # Bottom half in Y (vertical axis): alternating H and O (simplified water)
    for y in range(half):
        for z in range(size):
            for x in range(size):
                data[z, y, x, 0] = 8.0 if ((x + y + z) % 2 == 0) else 1.0  # O and H
                data[z, y, x, 1] = 25.0  # room temp
                data[z, y, x, 2] = 1.0   # liquid

    # Top-center: sodium block (high Y = top, centered X/Z)
    c = size // 2
    r = max(2, size // 10)
    for y in range(half + 2, half + 2 + r):
        for z in range(c - r, c + r):
            for x in range(c - r, c + r):
                if 0 <= y < size and 0 <= z < size and 0 <= x < size:
                    data[z, y, x, 0] = 11.0  # Na
                    data[z, y, x, 1] = 25.0
                    data[z, y, x, 2] = 0.0  # solid
                    data[z, y, x, 3] = -1.0  # falling

    return data


def init_metal_layers(size, rng):
    """Layers of metals with different melting points: Sn(232°C), Pb(327°C), Cu(1085°C), Fe(1538°C)."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)

    # Wall floor at y=0 to catch melting metals
    data[:, 0, :, 0] = float(WALL_ID)
    data[:, 0, :, 1] = 25.0
    data[:, 0, :, 2] = 0.0

    metals = [50, 82, 29, 26]  # Sn, Pb, Cu, Fe from bottom to top
    n_layers = len(metals)
    # Layers start at y=1 (above wall floor), with gaps between
    layer_h = max(2, (size - 1) // (n_layers + 2))

    for i, metal_z in enumerate(metals):
        y_start = 1 + i * (layer_h + 1)  # +1 for gap row between layers
        y_end = min(y_start + layer_h, size)
        data[:, y_start:y_end, :, 0] = float(metal_z)
        data[:, y_start:y_end, :, 1] = 500.0  # pre-heated
        data[:, y_start:y_end, :, 2] = 0.0     # solid

    # Sn melts at 232°C so at 500°C it's liquid
    y0 = 1
    data[:, y0:y0+layer_h, :, 2] = 1.0  # Sn = liquid at 500°C
    # Pb melts at 327°C so at 500°C it's also liquid
    y1 = 1 + layer_h + 1
    data[:, y1:y1+layer_h, :, 2] = 1.0  # Pb = liquid at 500°C

    return data


def init_lenia_blobs(size, rng):
    """Lenia: ellipsoidal blobs with random aspect ratios and surface noise.
    Lenia creatures are defined by shape — spherical seeds converge to the
    same morphology from every direction. Elongated, noisy starts produce
    more varied creature types."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    n_blobs = rng.randint(2, 8)
    # Surface noise to roughen blob boundaries
    surface_noise = _canonical_noise(size, rng, 0.7, 1.3)
    for _ in range(n_blobs):
        fx = rng.uniform(1.0/6, 5.0/6)
        fy = rng.uniform(1.0/6, 5.0/6)
        fz = rng.uniform(1.0/6, 5.0/6)
        cx, cy, cz = fx * size, fy * size, fz * size
        fr = rng.uniform(1.0/16, 1.0/6)
        base_r = fr * size
        # Random aspect ratios: each axis independently scaled 0.5x-2.0x
        ax = base_r * rng.uniform(0.5, 2.0)
        ay = base_r * rng.uniform(0.5, 2.0)
        az = base_r * rng.uniform(0.5, 2.0)
        # Ellipsoidal distance
        edist = np.sqrt(((x - cx) / max(ax, 1))**2 +
                        ((y - cy) / max(ay, 1))**2 +
                        ((z - cz) / max(az, 1))**2)
        amp = rng.uniform(0.3, 0.9)
        blob = (amp * np.exp(-0.5 * (edist / 0.4)**2) * surface_noise).astype(np.float32)
        data[:, :, :, 0] += blob
    data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0.0, 1.0)
    return data


def init_lenia_multi(size, rng):
    """Multi-channel Lenia: 3 channels with overlapping but distinct blob patterns.
    Each channel starts with different blob placements so cross-channel
    interactions create asymmetric dynamics from the start."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    surface_noise = _canonical_noise(size, rng, 0.7, 1.3)
    for ch in range(3):
        n_blobs = rng.randint(2, 6)
        for _ in range(n_blobs):
            cx = rng.uniform(size * 0.15, size * 0.85)
            cy = rng.uniform(size * 0.15, size * 0.85)
            cz = rng.uniform(size * 0.15, size * 0.85)
            base_r = rng.uniform(size * 0.06, size * 0.15)
            ax = base_r * rng.uniform(0.6, 1.8)
            ay = base_r * rng.uniform(0.6, 1.8)
            az = base_r * rng.uniform(0.6, 1.8)
            edist = np.sqrt(((x - cx)/max(ax,1))**2 + ((y - cy)/max(ay,1))**2 + ((z - cz)/max(az,1))**2)
            amp = rng.uniform(0.3, 0.8)
            blob = (amp * np.exp(-0.5 * (edist / 0.4)**2) * surface_noise).astype(np.float32)
            data[:, :, :, ch] += blob
        data[:, :, :, ch] = np.clip(data[:, :, :, ch], 0.0, 1.0)
    return data


def init_phase_separation(size, rng):
    """Cahn-Hilliard: near-zero order parameter with small perturbations (spinodal regime)."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    # Small random perturbations around c=0 (unstable mixed state)
    data[:, :, :, 0] = _canonical_noise(size, rng, -0.05, 0.05)
    # G channel (chemical potential) starts at zero
    return data

def init_erosion_terrain(size, rng):
    """Erosion: solid terrain block with water source at top."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    # Solid terrain fills bottom 60% with some noise for variation
    terrain_height = int(size * 0.6)
    noise = _canonical_noise(size, rng, -0.1, 0.1)
    for y in range(size):
        base = 1.0 if y < terrain_height else 0.0
        # Rough surface near the terrain boundary
        if abs(y - terrain_height) < size // 8:
            data[:, y, :, 0] = np.clip(base + noise[:, y, :], 0.0, 1.0)
        else:
            data[:, y, :, 0] = base
    # Water source: thin layer at top
    data[:, -2:, :, 1] = 0.8
    return data

def init_mycelium(size, rng):
    """Mycelium: nutrient field with tiny linear seed filaments."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    # Channel 1 = nutrient: smooth random field 0.5-1.0
    data[:, :, :, 1] = _canonical_noise(size, rng, 0.5, 1.0)
    # Channel 0 = biomass: short filament seeds (3 voxels each)
    n_seeds = rng.randint(5, 15)
    for _ in range(n_seeds):
        sx = rng.randint(size // 4, 3 * size // 4)
        sy = rng.randint(size // 4, 3 * size // 4)
        sz = rng.randint(size // 4, 3 * size // 4)
        axis = rng.randint(0, 3)  # random direction
        for step in range(3):
            px, py, pz = sx, sy, sz
            if axis == 0: px += step
            elif axis == 1: py += step
            else: pz += step
            if 0 <= px < size and 0 <= py < size and 0 <= pz < size:
                if step < 2:
                    data[pz, py, px, 0] = 1.0   # established core
                else:
                    data[pz, py, px, 0] = 0.2   # growing tip
                    data[pz, py, px, 3] = 1.0   # tip marker
    return data

def init_em_wave(size, rng):
    """EM wave: Gaussian z-dipole with seed magnetic field.
    A true dipole has Ez∝y (changes sign across midplane), creating a
    proper radiation pattern. The seed B field from ∇×E gives the wave
    an immediate direction instead of ringing from a hard-edged blob."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    c = size / 2.0
    r = max(2, size // 10)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    dx = (x - c).astype(np.float64)
    dy = (y - c).astype(np.float64)
    dz = (z - c).astype(np.float64)
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    # Gaussian envelope × dipole moment (Ez changes sign across y-midplane)
    envelope = np.exp(-0.5 * (dist / r)**2)
    Ez = (envelope * dy / max(r, 1)).astype(np.float32)
    data[:, :, :, 0] = Ez
    # Seed B field from initial curl(E): Bx ∝ -dEz/dz, By ∝ dEz/dx
    dEz_dx = np.gradient(Ez, axis=2)
    dEz_dz = np.gradient(Ez, axis=0)
    data[:, :, :, 1] = (-dEz_dz * 0.1).astype(np.float32)  # Bx seed
    data[:, :, :, 2] = (dEz_dx * 0.1).astype(np.float32)    # By seed
    return data

def init_viscous_fingers(size, rng):
    """Viscous fingers: smooth injection port with perturbed interface.
    Real Hele-Shaw injection has a smooth circular port, but the interface
    is never perfectly smooth — micro-roughness seeds the fingering instability.
    Sigmoid edge prevents numerical ringing; interface noise creates immediate
    finger nucleation sites instead of waiting for floating-point perturbation."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    c = size // 2
    r = max(2, size // 16)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    dist = np.sqrt((x - c)**2 + (y - c)**2 + (z - c)**2).astype(np.float64)
    # Interface noise: perturb effective radius to seed fingering instability
    interface_noise = _canonical_noise(size, rng, -1.0, 1.0)
    perturbed_r = r + interface_noise * (r * 0.3)
    # Smooth sigmoid transition (~2-voxel width) instead of hard step
    sat = (1.0 / (1.0 + np.exp(np.clip((dist - perturbed_r) * 2.0, -20, 20)))).astype(np.float32)
    data[:, :, :, 0] = sat
    # Pressure: smooth from center, not hard-edged
    clean_transition = (1.0 / (1.0 + np.exp(np.clip((dist - r) * 2.0, -20, 20)))).astype(np.float32)
    data[:, :, :, 1] = clean_transition
    # Permeability: heterogeneous porous medium, higher where invader starts
    perm = 0.5 + 0.3 * _canonical_noise(size, rng, -1.0, 1.0)
    perm = np.clip(perm, 0.1, 1.0).astype(np.float32)
    # Invader region starts with full permeability (dissolved channel)
    perm = np.maximum(perm, sat * 0.9)
    data[:, :, :, 2] = perm
    return data

def init_fire(size, rng):
    """Fire: heterogeneous fuel terrain with clearings and ignition source.
    Uniform fuel produces a boring uniform wavefront. Real forests have
    clearings, dense stands, and undergrowth variation — the fire front
    breaks up, races through dense patches, stalls at clearings, and
    creates spotting where embers jump gaps."""
    from scipy.ndimage import gaussian_filter, zoom
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    fine_noise = _canonical_noise(size, rng, 0.0, 1.0)
    # Large-scale structure: clearings and dense patches
    cs = CANONICAL_INIT_SIZE
    coarse = rng.random((cs, cs, cs)).astype(np.float32)
    coarse = gaussian_filter(coarse, sigma=4.0)
    if size != cs:
        coarse = zoom(coarse, size / cs, order=1).astype(np.float32)
    coarse = (coarse - coarse.min()) / (coarse.max() - coarse.min() + 1e-8)
    for y_idx in range(size):
        frac = y_idx / float(size)
        # Vertical gradient: dense at bottom, sparse at top
        base = max(0.0, 1.0 - frac * 1.3)
        # Mix large-scale + fine-scale fuel structure
        fuel_layer = base * (coarse[:, y_idx, :] * 0.5 + fine_noise[:, y_idx, :] * 0.3 + 0.2)
        # Clearings: voids where coarse noise is low
        fuel_layer[coarse[:, y_idx, :] < 0.25] *= 0.1
        data[:, y_idx, :, 0] = np.clip(fuel_layer, 0.0, 1.0)
    data[:, :, :, 2] = 1.0  # oxygen everywhere
    # Ignition: small hot spot near bottom-center
    c = size // 2
    r = max(2, size // 12)
    z, yy, x = np.mgrid[0:size, 0:size, 0:size]
    iy = size // 6
    dist = np.sqrt((x - c)**2 + (yy - iy)**2 + (z - c)**2)
    data[:, :, :, 1] = np.where(dist < r, 0.8, 0.0).astype(np.float32)
    return data

def init_physarum(size, rng):
    """Physarum: agent halos around food sources with distant scouts.
    Agents scattered uniformly take forever to find food and build networks.
    Seeding agents in halos *near* (not on) food sources creates immediate
    exploration dynamics — nearby agents lay trail toward food, trail attracts
    distant scouts, competing fronts merge into optimal transport networks."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    # Food sources: concentrated blobs
    n_food = rng.randint(3, 7)
    food_centers = []
    for _ in range(n_food):
        fx = rng.uniform(0.15, 0.85) * size
        fy = rng.uniform(0.15, 0.85) * size
        fz = rng.uniform(0.15, 0.85) * size
        fr = rng.uniform(0.05, 0.12) * size
        food_centers.append((fx, fy, fz, fr))
        dist = np.sqrt((x - fx)**2 + (y - fy)**2 + (z - fz)**2)
        data[:, :, :, 2] += np.where(dist < fr, 1.0, 0.0).astype(np.float32)
    data[:, :, :, 2] = np.clip(data[:, :, :, 2], 0.0, 1.0)
    # Agent density: ring-shaped halos around each food source
    for fx, fy, fz, fr in food_centers:
        dist = np.sqrt((x - fx)**2 + (y - fy)**2 + (z - fz)**2)
        halo_r = fr * 1.5  # peak at 1.5× food radius
        halo_w = fr * 0.6
        ring = np.exp(-0.5 * ((dist - halo_r) / halo_w)**2).astype(np.float32)
        data[:, :, :, 1] += ring * 0.4
    # Distant scout clusters: sparse agents far from food for competing fronts
    scout_noise = _canonical_noise(size, rng, 0.0, 1.0)
    food_proximity = np.zeros((size, size, size), dtype=np.float32)
    for fx, fy, fz, fr in food_centers:
        dist = np.sqrt((x - fx)**2 + (y - fy)**2 + (z - fz)**2)
        food_proximity = np.maximum(food_proximity, np.exp(-0.5 * (dist / (fr * 3))**2))
    distant_mask = food_proximity < 0.3
    data[:, :, :, 1] += np.where(distant_mask & (scout_noise > 0.97), 0.5, 0.0).astype(np.float32)
    data[:, :, :, 1] = np.clip(data[:, :, :, 1], 0.0, 1.0)
    return data

def init_fracture(size, rng):
    """Fracture: compact tension specimen with grain structure and edge notch.
    A center-crack with random stress is just inverted crystal growth.
    Real fracture has: (1) grain boundaries — weak paths cracks prefer to
    follow, (2) edge notch like a CT specimen — stress concentrates at the
    tip via 1/√r singularity, (3) applied tensile load from boundaries,
    not random noise. This produces branching cracks along grain boundaries,
    distinctly different from crystal growth."""
    from scipy.ndimage import zoom
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    cs = CANONICAL_INIT_SIZE
    # Channel 2 = integrity: 1.0 with weakened grain boundaries
    # Voronoi grain structure — cracks preferentially follow boundaries
    n_grains = rng.randint(20, 40)
    grain_pts = rng.uniform(0, cs, (n_grains, 3))
    zz, yy, xx = np.mgrid[0:cs, 0:cs, 0:cs]
    min_dist1 = np.full((cs, cs, cs), 1e9)
    min_dist2 = np.full((cs, cs, cs), 1e9)
    for gp in grain_pts:
        d = np.sqrt((xx - gp[0])**2 + (yy - gp[1])**2 + (zz - gp[2])**2)
        new_min2 = np.where(d < min_dist1, min_dist1, np.minimum(min_dist2, d))
        min_dist1 = np.minimum(min_dist1, d)
        min_dist2 = new_min2
    # Sharp Voronoi boundaries: integrity dips where d1 ≈ d2
    boundary = np.exp(-((min_dist2 - min_dist1) * 2.0)**2).astype(np.float32)
    if size != cs:
        boundary = zoom(boundary, size / cs, order=1).astype(np.float32)
    data[:, :, :, 2] = 1.0 - boundary * 0.15  # grain boundaries at 85% integrity
    # Edge notch: V-shaped notch from -X face (compact tension specimen)
    c = size // 2
    notch_depth = max(3, size // 6)
    notch_w = max(1, size // 24)
    for xi in range(notch_depth):
        # Notch tapers to a point → stress concentrator at tip
        width = max(1, int(notch_w * (1.0 - xi / notch_depth)))
        y_lo, y_hi = max(0, c - width), min(size, c + width + 1)
        z_lo, z_hi = max(0, c - width), min(size, c + width + 1)
        data[z_lo:z_hi, y_lo:y_hi, xi, 2] = 0.0
    # Channel 1 = stress: tensile load (mode I) + crack-tip concentration
    z_g, y_g, x_g = np.mgrid[0:size, 0:size, 0:size]
    # Applied tension: pulls Y-faces apart (increases near top/bottom)
    y_norm = (y_g - c).astype(np.float64) / max(c, 1)
    tension = np.abs(y_norm) * 0.15
    # 1/√r stress singularity at notch tip (classic fracture mechanics, clamped)
    tip_dist = np.sqrt((x_g - notch_depth)**2 + (y_g - c)**2 + (z_g - c)**2).astype(np.float64)
    stress_conc = 0.5 / np.sqrt(np.maximum(tip_dist, 1.0) / size)
    data[:, :, :, 1] = np.clip(tension + stress_conc * 0.2, 0.0, 0.5).astype(np.float32)
    return data

def init_galaxy(size, rng):
    """Galaxy: near-uniform density with small perturbations (Jeans instability)."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    # Channel 0 = density: uniform + moderate noise to seed Jeans instability
    data[:, :, :, 0] = 0.1 + _canonical_noise(size, rng, -0.05, 0.05)
    data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0.001, None)
    # Channels 1-3 = velocity: small random initial velocities
    data[:, :, :, 1] = _canonical_noise(size, rng, -0.02, 0.02)
    data[:, :, :, 2] = _canonical_noise(size, rng, -0.02, 0.02)
    data[:, :, :, 3] = _canonical_noise(size, rng, -0.02, 0.02)
    return data

def init_lichen(size, rng):
    """Lichen: three competing species with scattered seeds on a resource field."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    # Channel 2 = resource: starts nearly full
    data[:, :, :, 2] = _canonical_noise(size, rng, 0.7, 1.0)
    # Channel 0 = species A (pioneer): scattered seeds
    noise_a = _canonical_noise(size, rng, 0.0, 1.0)
    data[:, :, :, 0] = np.where(noise_a > 0.94, 0.5, 0.0).astype(np.float32)
    # Channel 1 = species B (competitor): fewer, different locations
    noise_b = _canonical_noise(size, rng, 0.0, 1.0)
    data[:, :, :, 1] = np.where(noise_b > 0.96, 0.5, 0.0).astype(np.float32)
    # Channel 3 = species C (nomad): rarest
    noise_c = _canonical_noise(size, rng, 0.0, 1.0)
    data[:, :, :, 3] = np.where(noise_c > 0.97, 0.5, 0.0).astype(np.float32)
    return data


# ── Structured init variants ────────────────────────────────────────
# These create spatially organized initial conditions that seed specific
# dynamics: pursuit waves, Turing spots, spiral defects, grain boundaries,
# erosion channels, etc.  Used by search via init_variants.

def init_predator_prey_separated(size, rng):
    """Prey in one hemisphere, predators in the opposite → pursuit chase dynamics.
    Spatial separation forces migration → traveling waves, not static coexistence."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]

    # Random dividing plane through center
    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(-np.pi / 4, np.pi / 4)
    nx = np.cos(phi) * np.cos(theta)
    ny = np.cos(phi) * np.sin(theta)
    nz = np.sin(phi)
    mid = size / 2.0
    signed_dist = (x - mid) * nx + (y - mid) * ny + (z - mid) * nz

    # Prey concentrated on positive side, predators on negative
    prey = np.where(signed_dist > 0, 0.9, 0.05).astype(np.float32)
    predator = np.where(signed_dist < 0, 0.5, 0.01).astype(np.float32)

    # Smooth the boundary (3-cell transition zone)
    transition = np.exp(-0.5 * (signed_dist / max(size * 0.05, 1.5))**2).astype(np.float32)
    prey = prey * (1 - transition * 0.5) + transition * 0.4
    predator = predator * (1 - transition * 0.3) + transition * 0.2

    # Add noise for symmetry breaking
    prey += _canonical_noise(size, rng, -0.03, 0.03)
    predator += _canonical_noise(size, rng, -0.02, 0.02)

    data[:, :, :, 0] = np.clip(prey, 0.01, 1.0)
    data[:, :, :, 1] = np.clip(predator, 0.0, 1.0)
    return data


def init_morphogen_hotspots(size, rng):
    """High activator in a few hotspots, low everywhere else → Turing patterns
    grow outward from nucleation sites instead of competing everywhere at once."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]

    # Low baseline activator and inhibitor
    data[:, :, :, 0] = 0.05  # activator low
    data[:, :, :, 1] = 0.01  # inhibitor very low
    data[:, :, :, 2] = 0.5   # tissue density

    # Place 4-10 activator hotspots
    n_spots = rng.randint(4, 11)
    for _ in range(n_spots):
        cx = rng.uniform(0.1, 0.9) * size
        cy = rng.uniform(0.1, 0.9) * size
        cz = rng.uniform(0.1, 0.9) * size
        r = rng.uniform(max(1.5, size * 0.03), size * 0.08)
        amp = rng.uniform(0.3, 0.8)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        data[:, :, :, 0] += (amp * np.exp(-0.5 * (dist / r)**2)).astype(np.float32)

    data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0.01, 5.0)
    # Tiny noise for secondary instability
    data[:, :, :, 0] += _canonical_noise(size, rng, -0.005, 0.005)
    data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0.01, 5.0)
    return data


def init_bz_spiral_seed(size, rng):
    """Engineered phase singularity → spiral/scroll waves form immediately.
    Phase winds 0→2π around a central axis, creating the topological defect
    that seeds scroll wave dynamics in BZ/CGLE."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    mid = size / 2.0

    # Choose a random axis for the scroll wave
    axis = rng.randint(0, 3)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]

    if axis == 0:
        dx, dy = (x - mid).astype(np.float64), (y - mid).astype(np.float64)
    elif axis == 1:
        dx, dy = (x - mid).astype(np.float64), (z - mid).astype(np.float64)
    else:
        dx, dy = (y - mid).astype(np.float64), (z - mid).astype(np.float64)

    # Phase = angle around axis (creates 2π winding = topological defect)
    phase = np.arctan2(dy, dx)  # [-π, π]
    # Add a second singularity offset from center for richer dynamics
    off = rng.uniform(0.15, 0.35) * size
    angle = rng.uniform(0, 2 * np.pi)
    cx2 = mid + off * np.cos(angle)
    cy2 = mid + off * np.sin(angle)
    if axis == 0:
        dx2, dy2 = x - cx2, y - cy2
    elif axis == 1:
        dx2, dy2 = x - cx2, z - cy2
    else:
        dx2, dy2 = y - cx2, z - cy2
    phase2 = np.arctan2(dy2.astype(np.float64), dx2.astype(np.float64))
    # Combine: two counter-rotating spirals
    combined_phase = phase + phase2

    # Amplitude near limit cycle
    amp = 1.0 + _canonical_noise(size, rng, -0.05, 0.05)

    data[:, :, :, 0] = (amp * np.cos(combined_phase)).astype(np.float32)
    data[:, :, :, 1] = (amp * np.sin(combined_phase)).astype(np.float32)
    data[:, :, :, 2] = ((combined_phase / (2 * np.pi)) % 1.0).astype(np.float32)
    return data


def init_crystal_multi_seed(size, rng):
    """Multiple competing crystal nucleation sites → grain boundaries form
    where growing crystals collide. Much richer than single-seed growth."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]

    # Supersaturation field (uniform undercooled melt)
    data[:, :, :, 1] = 1.0

    # 3-7 seeds at random positions
    n_seeds = rng.randint(3, 8)
    for _ in range(n_seeds):
        cx = rng.uniform(0.1, 0.9) * size
        cy = rng.uniform(0.1, 0.9) * size
        cz = rng.uniform(0.1, 0.9) * size
        r = max(1.0, size * 0.02)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        seed_mask = dist < r
        data[:, :, :, 0] = np.where(seed_mask, 1.0, data[:, :, :, 0])
        data[:, :, :, 1] = np.where(seed_mask, 0.0, data[:, :, :, 1])

    return data


def init_erosion_ridges(size, rng):
    """Terrain with ridges, valleys, and pre-carved channels → water concentrates
    along existing weak points → feedback loops (deeper flow → faster erosion)."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z_grid, y_grid, x_grid = np.mgrid[0:size, 0:size, 0:size]

    # Height field with ridges (low-freq sine waves)
    n_ridges = rng.randint(2, 5)
    height = np.zeros((size, size), dtype=np.float32)
    for _ in range(n_ridges):
        angle = rng.uniform(0, np.pi)
        freq = rng.uniform(0.5, 2.0)
        amp = rng.uniform(0.1, 0.3)
        height += amp * np.sin(freq * 2 * np.pi *
            (x_grid[0, 0, :, np.newaxis] * np.cos(angle)
             + z_grid[0, :, 0, np.newaxis].T * np.sin(angle)) / size)

    # Normalize height to [0.3, 0.8] of grid
    height = 0.3 + 0.5 * (height - height.min()) / max(height.max() - height.min(), 0.01)
    terrain_top = (height * size).astype(int)

    # Fill solid below terrain surface
    for yi in range(size):
        for zi in range(size):
            h = min(terrain_top[zi, yi], size - 2)
            data[zi, :h, yi, 0] = 1.0

    # Cut 1-3 channels (pre-carved valleys)
    n_channels = rng.randint(1, 4)
    for _ in range(n_channels):
        ch_z = rng.randint(size // 4, 3 * size // 4)
        ch_width = rng.randint(max(1, size // 16), max(2, size // 8))
        ch_depth = rng.randint(max(1, size // 8), max(2, size // 4))
        z_lo = max(0, ch_z - ch_width)
        z_hi = min(size, ch_z + ch_width)
        for zi in range(z_lo, z_hi):
            for yi in range(size):
                h = min(terrain_top[zi, yi], size - 2)
                carve_top = max(0, h - ch_depth)
                for y_v in range(carve_top, h):
                    data[zi, y_v, yi, 0] = 0.0

    # Water source at top
    data[:, -2:, :, 1] = 0.8
    # Add some water in channels too
    data[:, :, :, 1] = np.where(
        (data[:, :, :, 0] < 0.5) & (y_grid < int(size * 0.5)),
        0.3, data[:, :, :, 1])
    return data


def init_flocking_vortex(size, rng):
    """Agents arranged in 2-4 initial vortex rings → immediate collective rotation
    instead of slow random alignment. Vortices interact, merge, or compete."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    mid = size / 2.0

    n_vortices = rng.randint(2, 5)
    for _ in range(n_vortices):
        # Vortex center
        cx = rng.uniform(0.2, 0.8) * size
        cy = rng.uniform(0.2, 0.8) * size
        cz = rng.uniform(0.2, 0.8) * size
        r_ring = rng.uniform(0.1, 0.25) * size
        r_tube = rng.uniform(0.04, 0.1) * size

        # Distance from ring center (in a random plane)
        dx = (x - cx).astype(np.float64)
        dy = (y - cy).astype(np.float64)
        dz = (z - cz).astype(np.float64)

        # Project onto ring plane (pick random axis)
        axis = rng.randint(0, 3)
        if axis == 0:
            r_from_axis = np.sqrt(dy**2 + dz**2)
            angle = np.arctan2(dz, dy)
            dist_from_ring = np.sqrt((r_from_axis - r_ring)**2 + dx**2)
        elif axis == 1:
            r_from_axis = np.sqrt(dx**2 + dz**2)
            angle = np.arctan2(dz, dx)
            dist_from_ring = np.sqrt((r_from_axis - r_ring)**2 + dy**2)
        else:
            r_from_axis = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)
            dist_from_ring = np.sqrt((r_from_axis - r_ring)**2 + dz**2)

        # Density concentrated around the ring
        density = 0.5 * np.exp(-0.5 * (dist_from_ring / r_tube)**2)
        data[:, :, :, 0] += density.astype(np.float32)

        # Tangential velocity (circular motion around the ring)
        speed = rng.uniform(0.1, 0.3)
        if axis == 0:
            data[:, :, :, 2] += (-density * np.sin(angle) * speed).astype(np.float32)
            data[:, :, :, 3] += ( density * np.cos(angle) * speed).astype(np.float32)
        elif axis == 1:
            data[:, :, :, 1] += (-density * np.sin(angle) * speed).astype(np.float32)
            data[:, :, :, 3] += ( density * np.cos(angle) * speed).astype(np.float32)
        else:
            data[:, :, :, 1] += (-density * np.sin(angle) * speed).astype(np.float32)
            data[:, :, :, 2] += ( density * np.cos(angle) * speed).astype(np.float32)

    data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0, 1)
    return data


# ── New init variants for improved CA visual quality ──

def init_game_of_life_centered(size, rng):
    """Clustered population in center third — grows outward into empty space."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    lo = size // 3
    hi = 2 * size // 3
    cs = CANONICAL_INIT_SIZE
    # Generate at canonical res, extract center
    field = rng.random((cs, cs, cs)).astype(np.float32)
    from scipy.ndimage import zoom
    if size != cs:
        field = zoom(field, size / cs, order=0).astype(np.float32)
    # Only populate center third (density ~35%)
    center = field[lo:hi, lo:hi, lo:hi]
    data[lo:hi, lo:hi, lo:hi, 0] = (center < 0.35).astype(np.float32)
    return data


def init_smoothlife_sparse(size, rng):
    """A few smooth blobs in empty space — watch creatures nucleate and interact."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    n_blobs = rng.randint(4, 9)
    for _ in range(n_blobs):
        cx = rng.uniform(0.15, 0.85) * size
        cy = rng.uniform(0.15, 0.85) * size
        cz = rng.uniform(0.15, 0.85) * size
        rx = rng.uniform(0.05, 0.12) * size
        ry = rng.uniform(0.05, 0.12) * size
        rz = rng.uniform(0.05, 0.12) * size
        z, y, x = np.mgrid[0:size, 0:size, 0:size]
        d2 = ((x - cx) / rx)**2 + ((y - cy) / ry)**2 + ((z - cz) / rz)**2
        val = rng.uniform(0.4, 0.8) * np.exp(-0.5 * d2)
        data[:, :, :, 0] += val.astype(np.float32)
    data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0.0, 1.0)
    return data


def init_gray_scott_worms_dense(size, rng):
    """Gray-Scott with dense V catalyst — worm patterns appear immediately."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    # U = 1 everywhere (substrate)
    data[:, :, :, 0] = 1.0
    # V = 0.25 baseline + noise — dense enough for immediate worm nucleation
    noise = _canonical_noise(size, rng, 0.0, 0.15)
    data[:, :, :, 1] = 0.25 + noise
    # Subtract catalyst from substrate
    data[:, :, :, 0] -= data[:, :, :, 1]
    return data


def init_lenia_multi_colocated(size, rng):
    """Multi-channel Lenia with co-located blobs — channels differentiate in place.
    Creates creatures with true internal structure (core/skin/metabolism)."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    n_blobs = rng.randint(3, 7)
    for _ in range(n_blobs):
        # Same center for all channels
        cx = rng.uniform(0.15, 0.85) * size
        cy = rng.uniform(0.15, 0.85) * size
        cz = rng.uniform(0.15, 0.85) * size
        z, y, x = np.mgrid[0:size, 0:size, 0:size]
        dx = x - cx
        dy = y - cy
        dz = z - cz

        for ch in range(3):
            # Each channel has slightly different shape
            rx = rng.uniform(0.06, 0.14) * size
            ry = rng.uniform(0.06, 0.14) * size
            rz = rng.uniform(0.06, 0.14) * size
            d2 = (dx / rx)**2 + (dy / ry)**2 + (dz / rz)**2
            amp = rng.uniform(0.3, 0.7)
            data[:, :, :, ch] += (amp * np.exp(-0.5 * d2)).astype(np.float32)

    data[:, :, :, :3] = np.clip(data[:, :, :, :3], 0.0, 1.0)
    return data


def init_kuramoto_clusters(size, rng):
    """Pre-formed phase-locked clusters — chimera states emerge as clusters compete."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    c = size / 2.0

    # 3-4 large phase-locked clusters at different spatial positions
    n_clusters = rng.randint(3, 5)
    phase_field = np.zeros((size, size, size), dtype=np.float64)
    weight_sum = np.zeros((size, size, size), dtype=np.float64) + 1e-10

    for i in range(n_clusters):
        cx = rng.uniform(0.2, 0.8) * size
        cy = rng.uniform(0.2, 0.8) * size
        cz = rng.uniform(0.2, 0.8) * size
        radius = rng.uniform(0.15, 0.3) * size
        target_phase = i / n_clusters  # evenly spaced phases [0,1)
        # Gaussian weight for this cluster
        d2 = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
        w = np.exp(-0.5 * d2 / radius**2)
        phase_field += w * target_phase
        weight_sum += w

    phase_field /= weight_sum
    # Add small noise so clusters aren't perfectly locked
    phase_field += rng.uniform(0, 0.05, (size, size, size))
    data[:, :, :, 0] = (phase_field % 1.0).astype(np.float32)

    # Natural frequency: spatially varying (channel G)
    freq = _canonical_noise(size, rng, -0.5, 0.5)
    data[:, :, :, 1] = freq
    # Coherence starts near 1 within clusters
    data[:, :, :, 2] = np.clip(1.0 - 2.0 * rng.random((size, size, size)).astype(np.float32) * 0.3, 0, 1).astype(np.float32)
    return data


def init_galaxy_filaments(size, rng):
    """Pre-seeded cosmic web — density perturbations at Jeans wavelength
    so gravitational collapse is visible within ~50 steps."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    c = size / 2.0
    z, y, x = np.mgrid[0:size, 0:size, 0:size]

    # Baseline density
    base_density = 0.1
    density = np.full((size, size, size), base_density, dtype=np.float64)

    # Seed 3-5 overdense filament-like regions (sine waves in random directions)
    for _ in range(rng.randint(3, 6)):
        kx = rng.uniform(-0.15, 0.15)
        ky = rng.uniform(-0.15, 0.15)
        kz = rng.uniform(-0.15, 0.15)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.02, 0.06)
        density += amp * np.cos(kx * x + ky * y + kz * z + phase)

    # Add 2-3 dense nodes (proto-cluster seeds)
    for _ in range(rng.randint(2, 4)):
        nx = rng.uniform(0.2, 0.8) * size
        ny = rng.uniform(0.2, 0.8) * size
        nz = rng.uniform(0.2, 0.8) * size
        r2 = (x - nx)**2 + (y - ny)**2 + (z - nz)**2
        sigma = rng.uniform(0.05, 0.1) * size
        density += 0.08 * np.exp(-0.5 * r2 / sigma**2)

    density = np.clip(density, 0.01, 0.5)
    data[:, :, :, 0] = density.astype(np.float32)
    # Small random velocities
    data[:, :, :, 1] = (rng.uniform(-0.005, 0.005, (size, size, size))).astype(np.float32)
    data[:, :, :, 2] = (rng.uniform(-0.005, 0.005, (size, size, size))).astype(np.float32)
    data[:, :, :, 3] = (rng.uniform(-0.005, 0.005, (size, size, size))).astype(np.float32)
    return data


def init_lichen_dense(size, rng):
    """Dense starting coverage — territorial competition begins immediately."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    c = size / 2.0

    # Place large patches of each species (fill ~20-30% each)
    for ch, n_patches in [(0, 5), (1, 4), (3, 3)]:  # A, B, C
        for _ in range(n_patches):
            cx = rng.uniform(0.1, 0.9) * size
            cy = rng.uniform(0.1, 0.9) * size
            cz = rng.uniform(0.1, 0.9) * size
            r = rng.uniform(0.08, 0.18) * size
            d2 = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
            data[:, :, :, ch] += (0.4 * np.exp(-0.5 * d2 / r**2)).astype(np.float32)

    # Resource (channel 2) — uniform
    data[:, :, :, 2] = 0.5
    data[:, :, :, 0] = np.clip(data[:, :, :, 0], 0, 1)
    data[:, :, :, 1] = np.clip(data[:, :, :, 1], 0, 1)
    data[:, :, :, 3] = np.clip(data[:, :, :, 3], 0, 1)
    return data


def init_mycelium_foraging(size, rng):
    """Nutrient hotspots with sparse initial tips — network grows to connect food sources."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    c = size / 2.0
    z, y, x = np.mgrid[0:size, 0:size, 0:size]

    # 3-5 concentrated nutrient hotspots
    nutrient = np.full((size, size, size), 0.05, dtype=np.float64)
    n_food = rng.randint(3, 6)
    food_positions = []
    for _ in range(n_food):
        fx = rng.uniform(0.15, 0.85) * size
        fy = rng.uniform(0.15, 0.85) * size
        fz = rng.uniform(0.15, 0.85) * size
        food_positions.append((fx, fy, fz))
        r2 = (x - fx)**2 + (y - fy)**2 + (z - fz)**2
        sigma = rng.uniform(0.04, 0.08) * size
        nutrient += 0.8 * np.exp(-0.5 * r2 / sigma**2)
    data[:, :, :, 1] = np.clip(nutrient, 0, 1).astype(np.float32)

    # Seed a few tips near center (colony start)
    cx, cy, cz = c, c, c
    for _ in range(5):
        tx = int(cx + rng.uniform(-3, 3))
        ty = int(cy + rng.uniform(-3, 3))
        tz = int(cz + rng.uniform(-3, 3))
        if 0 <= tx < size and 0 <= ty < size and 0 <= tz < size:
            data[tz, ty, tx, 0] = 1.0
    return data


def init_bz_turbulence_high_amp(size, rng):
    """High-amplitude BZ initial state — survives Benjamin-Feir instability longer."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    # Amplitude 2× limit cycle — will saturate to turbulent field
    amp = 2.0
    phase = _canonical_noise(size, rng, 0.0, 2.0 * np.pi)
    data[:, :, :, 0] = amp * np.cos(phase)  # Re(A)
    data[:, :, :, 1] = amp * np.sin(phase)  # Im(A)
    data[:, :, :, 2] = phase / (2.0 * np.pi)  # Phase [0,1]
    return data


def init_fire_sparse(size, rng):
    """Fuel islands separated by firebreaks — fire jumps between patches via embers."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]

    # Scattered fuel islands (Gaussian blobs)
    fuel = np.zeros((size, size, size), dtype=np.float64)
    n_patches = rng.randint(8, 16)
    for _ in range(n_patches):
        fx = rng.uniform(0.1, 0.9) * size
        fy = rng.uniform(0.05, 0.85) * size  # bias lower (gravity)
        fz = rng.uniform(0.1, 0.9) * size
        r = rng.uniform(0.06, 0.14) * size
        d2 = (x - fx)**2 + (y - fy)**2 + (z - fz)**2
        fuel += rng.uniform(0.5, 0.9) * np.exp(-0.5 * d2 / r**2)

    data[:, :, :, 0] = np.clip(fuel, 0, 1).astype(np.float32)

    # Ignition point (bottom center)
    ic = size // 2
    for dx in range(-2, 3):
        for dz in range(-2, 3):
            ix, iz = ic + dx, ic + dz
            if 0 <= ix < size and 0 <= iz < size:
                data[iz, 1, ix, 1] = 1.0  # temperature
    # Oxygen starts at 1
    data[:, :, :, 2] = 1.0
    return data


def init_phase_separation_quench(size, rng):
    """Deep quench — starts with pre-separated patches for fast coarsening dynamics."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]

    # Coarse random field using a few overlapping Gaussian blobs
    c_field = np.zeros((size, size, size), dtype=np.float64)
    n_blobs = rng.randint(8, 15)
    for _ in range(n_blobs):
        bx = rng.uniform(0, 1) * size
        by = rng.uniform(0, 1) * size
        bz = rng.uniform(0, 1) * size
        sigma = rng.uniform(0.08, 0.18) * size
        sign = rng.choice([-1, 1])
        d2 = (x - bx)**2 + (y - by)**2 + (z - bz)**2
        c_field += sign * 0.6 * np.exp(-0.5 * d2 / sigma**2)

    # Normalize to [-0.8, 0.8] — already near binodal
    c_max = np.max(np.abs(c_field))
    if c_max > 1e-6:
        c_field = 0.8 * c_field / c_max
    data[:, :, :, 0] = c_field.astype(np.float32)
    return data


def init_element_layered(size, rng):
    """Stratified element layers — reactions happen at interfaces between materials."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    # Element IDs: C=6, O=8, Na=11, Fe=26, Cu=29
    # Bottom third: iron. Middle third: carbon. Top third: oxygen gas.
    third = size // 3
    # Fe layer (solid, dense, bottom)
    data[:, :third, :, 0] = 26.0  # Fe
    data[:, :third, :, 1] = 25.0  # temp
    data[:, :third, :, 2] = 0.0   # solid
    # C layer (solid, middle)
    data[:, third:2*third, :, 0] = 6.0  # C
    data[:, third:2*third, :, 1] = 25.0
    data[:, third:2*third, :, 2] = 0.0
    # O layer (gas, top)
    data[:, 2*third:, :, 0] = 8.0  # O
    data[:, 2*third:, :, 1] = 25.0
    data[:, 2*third:, :, 2] = 1.0  # gas phase

    # Sprinkle some Na at the interfaces for reactivity
    interface_lo = third - 1
    interface_hi = 2 * third
    n_na = max(1, size // 8)
    for _ in range(n_na):
        nx = rng.randint(0, size)
        ny = rng.choice([interface_lo, interface_hi])
        nz = rng.randint(0, size)
        if 0 <= ny < size:
            data[nz, ny, nx, 0] = 11.0  # Na
            data[nz, ny, nx, 1] = 25.0
    return data


# ── Quantum mechanics initialization functions ──

def _coulomb_potential(size, Z, r_soft=1.5):
    """Softened Coulomb potential V(r) = -Z / sqrt(r² + r_soft²), centered in grid."""
    c = size / 2.0
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    r = np.sqrt((x - c + 0.5)**2 + (y - c + 0.5)**2 + (z - c + 0.5)**2).astype(np.float64)
    V = (-Z / np.sqrt(r**2 + r_soft**2)).astype(np.float32)
    return V


def _harmonic_potential(size, omega=0.02):
    """3D harmonic trap V(r) = ½ω²r²."""
    c = size / 2.0
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    r2 = ((x - c + 0.5)**2 + (y - c + 0.5)**2 + (z - c + 0.5)**2).astype(np.float64)
    V = (0.5 * omega**2 * r2).astype(np.float32)
    return V


def _double_well_potential(size, depth=0.5, sep=0.25):
    """Two Gaussian wells separated along x-axis — for tunneling demos."""
    c = size / 2.0
    d = sep * size * 0.5
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    r_width = size * 0.08
    r1 = np.sqrt((x - c + d + 0.5)**2 + (y - c + 0.5)**2 + (z - c + 0.5)**2)
    r2 = np.sqrt((x - c - d + 0.5)**2 + (y - c + 0.5)**2 + (z - c + 0.5)**2)
    V = -depth * (np.exp(-0.5 * (r1 / r_width)**2) + np.exp(-0.5 * (r2 / r_width)**2))
    return V.astype(np.float32)


def _box_potential(size, wall_thickness=2, wall_height=8.0):
    """Square well (particle in a box) — walls at grid edges.
    V=8 gives penetration depth ~0.35 cells (99.7% confined in 2 cells)."""
    V = np.zeros((size, size, size), dtype=np.float32)
    w = wall_thickness
    V[:w, :, :] = wall_height
    V[-w:, :, :] = wall_height
    V[:, :w, :] = wall_height
    V[:, -w:, :] = wall_height
    V[:, :, :w] = wall_height
    V[:, :, -w:] = wall_height
    return V


def _gaussian_wavepacket(size, cx, cy, cz, sigma, kx=0.0, ky=0.0, kz=0.0):
    """Gaussian wavepacket Ψ = exp(-r²/4σ²) * exp(i k·r), split into Re/Im."""
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    dx = (x - cx + 0.5).astype(np.float64)
    dy = (y - cy + 0.5).astype(np.float64)
    dz = (z - cz + 0.5).astype(np.float64)
    r2 = dx**2 + dy**2 + dz**2
    envelope = np.exp(-r2 / (4.0 * sigma**2))
    phase = kx * dx + ky * dy + kz * dz
    psi_r = (envelope * np.cos(phase)).astype(np.float32)
    psi_i = (envelope * np.sin(phase)).astype(np.float32)
    # Normalize so max|Ψ| = 1 (keeps peak probability visible in renderer)
    norm = np.sqrt(np.max(psi_r**2 + psi_i**2))
    if norm > 1e-10:
        psi_r /= norm
        psi_i /= norm
    return psi_r, psi_i


def _hydrogen_orbital(size, n, l, m, Z=1.0, scale=1.0):
    """Hydrogen-like orbital ψ_nlm(r,θ,φ) = R_nl(r) * Y_lm(θ,φ).

    Returns real/imaginary parts. Uses real spherical harmonics for m≠0
    (standard chemistry convention: orbitals like dxy, dxz etc.)
    """
    from scipy.special import assoc_laguerre, sph_harm_y
    c = size / 2.0
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    dx = (x - c + 0.5).astype(np.float64)
    dy = (y - c + 0.5).astype(np.float64)
    dz = (z - c + 0.5).astype(np.float64)
    r = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-10
    theta = np.arccos(np.clip(dz / r, -1, 1))
    phi = np.arctan2(dy, dx)

    # Radial part: R_nl(r) ∝ ρ^l * exp(-ρ/2) * L_{n-l-1}^{2l+1}(ρ)
    # a₀ = size/25 cells → K = a₀*Z/2 = size*Z/50 (auto-set by preset)
    # n=1 fills ~6% of grid, n=2 ~24%, n=3 ~54%
    a0 = size * scale / (25.0 * Z)
    rho = 2.0 * r / (n * a0)
    L = assoc_laguerre(rho, n - l - 1, 2 * l + 1)
    R = np.power(rho, l) * np.exp(-rho / 2.0) * L

    # Angular part: real spherical harmonic
    # sph_harm_y(l, m, theta, phi) — scipy ≥ 1.15 API
    if m == 0:
        Y = sph_harm_y(l, 0, theta, phi).real
    elif m > 0:
        # Real combination: (Y_lm + Y_l,-m) / sqrt(2) ∝ cos(mφ)
        Y = (sph_harm_y(l, m, theta, phi).real * np.sqrt(2))
    else:
        # Real combination: (Y_l,-|m| - Y_l,|m|) / (i√2) ∝ sin(|m|φ)
        Y = (sph_harm_y(l, abs(m), theta, phi).imag * np.sqrt(2))

    psi = (R * Y).astype(np.float64)
    # Normalize so max|Ψ| = 1 (keeps peak probability visible in renderer)
    norm = np.max(np.abs(psi))
    if norm > 1e-10:
        psi /= norm
    return psi.astype(np.float32), np.zeros((size, size, size), dtype=np.float32)


def init_quantum_hydrogen(size, rng):
    """Hydrogen atom 1s+2p superposition in Coulomb potential.
    Probability cloud oscillates between sphere and dumbbell shapes."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    # Superposition: 1s (sphere) + 2p_z (dumbbell)
    # Oscillation period ≈ 84 steps ≈ 1.4 sec at 60 fps
    psi_1s, _ = _hydrogen_orbital(size, n=1, l=0, m=0)
    psi_2p, _ = _hydrogen_orbital(size, n=2, l=1, m=0)
    psi_r = psi_1s + psi_2p
    psi_i = np.zeros_like(psi_r)
    # Re-normalize so max|Ψ| = 1
    norm = np.max(np.abs(psi_r))
    if norm > 1e-10:
        psi_r /= norm
    data[:, :, :, 0] = psi_r
    data[:, :, :, 1] = psi_i
    data[:, :, :, 2] = _coulomb_potential(size, Z=1.0, r_soft=1.5)
    data[:, :, :, 3] = psi_r**2 + psi_i**2
    return data


def init_quantum_orbital(size, rng):
    """Random hydrogen orbital (n=1-3) — watch standing wave patterns.
    Sometimes seeds a superposition of two orbitals for interference."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    # Pick random quantum numbers
    n = rng.randint(1, 4)  # n = 1, 2, or 3
    l = rng.randint(0, n)  # l = 0 to n-1
    m = rng.randint(-l, l + 1)  # m = -l to l

    psi_r, psi_i = _hydrogen_orbital(size, n, l, m)

    # 40% chance: superpose with a different orbital
    if rng.random() < 0.4:
        n2 = rng.randint(1, 4)
        l2 = rng.randint(0, n2)
        m2 = rng.randint(-l2, l2 + 1)
        if (n2, l2, m2) != (n, l, m):
            psi_r2, psi_i2 = _hydrogen_orbital(size, n2, l2, m2)
            weight = rng.uniform(0.3, 0.7)
            psi_r = psi_r * weight + psi_r2 * (1 - weight)
            psi_i = psi_i * weight + psi_i2 * (1 - weight)
            norm = np.sqrt(np.max(psi_r**2 + psi_i**2))
            if norm > 1e-10:
                psi_r /= norm
                psi_i /= norm

    data[:, :, :, 0] = psi_r
    data[:, :, :, 1] = psi_i
    data[:, :, :, 2] = _coulomb_potential(size, Z=1.0, r_soft=1.5)
    data[:, :, :, 3] = psi_r**2 + psi_i**2
    return data


# ── Individual orbital init variants (factory) ──

def _make_orbital_init(n, l, m, label):
    """Factory for orbital init functions with specific quantum numbers."""
    def init_fn(size, rng):
        data = np.zeros((size, size, size, 4), dtype=np.float32)
        psi_r, psi_i = _hydrogen_orbital(size, n, l, m)
        data[:, :, :, 0] = psi_r
        data[:, :, :, 1] = psi_i
        data[:, :, :, 2] = _coulomb_potential(size, Z=1.0, r_soft=1.5)
        data[:, :, :, 3] = psi_r**2 + psi_i**2
        return data
    init_fn.__name__ = f'init_orbital_{label}'
    init_fn.__doc__ = f'Hydrogen {label} orbital (n={n}, l={l}, m={m}).'
    return init_fn

# Standard orbital labels: spectroscopic notation + m subscript
_ORBITAL_DEFS = [
    # (n, l, m, label)
    (1, 0, 0, '1s'),
    (2, 0, 0, '2s'),
    (2, 1, 0, '2p0'),
    (2, 1, 1, '2p1'),
    (2, 1, -1, '2p-1'),
    (3, 0, 0, '3s'),
    (3, 1, 0, '3p0'),
    (3, 1, 1, '3p1'),
    (3, 2, 0, '3d0'),
    (3, 2, 1, '3d1'),
    (3, 2, 2, '3d2'),
    (3, 2, -1, '3d-1'),
    (3, 2, -2, '3d-2'),
    (4, 0, 0, '4s'),
    (4, 1, 0, '4p0'),
    (4, 2, 0, '4d0'),
    (4, 3, 0, '4f0'),
    (4, 3, 3, '4f3'),
]

ORBITAL_INITS = {}
for _n, _l, _m, _label in _ORBITAL_DEFS:
    _fname = f'orbital_{_label}'
    ORBITAL_INITS[_fname] = _make_orbital_init(_n, _l, _m, _label)


def init_quantum_wavepacket(size, rng):
    """Gaussian wavepacket with random momentum — watch it disperse,
    bounce off walls, and interfere with itself."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    c = size / 2.0
    sigma = rng.uniform(size * 0.06, size * 0.12)

    # Random offset from center
    cx = c + rng.uniform(-size * 0.15, size * 0.15)
    cy = c + rng.uniform(-size * 0.15, size * 0.15)
    cz = c + rng.uniform(-size * 0.15, size * 0.15)

    # Random momentum (determines direction of travel)
    k_mag = rng.uniform(0.2, 1.0)
    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(-np.pi / 2, np.pi / 2)
    kx = k_mag * np.cos(phi) * np.cos(theta)
    ky = k_mag * np.cos(phi) * np.sin(theta)
    kz = k_mag * np.sin(phi)

    psi_r, psi_i = _gaussian_wavepacket(size, cx, cy, cz, sigma, kx, ky, kz)
    data[:, :, :, 0] = psi_r
    data[:, :, :, 1] = psi_i
    # Box potential (particle in a box)
    data[:, :, :, 2] = _box_potential(size)
    data[:, :, :, 3] = psi_r**2 + psi_i**2
    return data


def init_quantum_harmonic(size, rng):
    """Wavepacket in harmonic trap — coherent state oscillation.
    Displaced Gaussian oscillates back and forth like a classical particle."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    c = size / 2.0
    sigma = size * 0.08

    # Displace from center (the restoring force will pull it back)
    displacement = rng.uniform(0.1, 0.25) * size
    angle = rng.uniform(0, 2 * np.pi)
    cx = c + displacement * np.cos(angle)
    cy = c + displacement * np.sin(angle)
    cz = c

    psi_r, psi_i = _gaussian_wavepacket(size, cx, cy, cz, sigma)
    data[:, :, :, 0] = psi_r
    data[:, :, :, 1] = psi_i
    data[:, :, :, 2] = _harmonic_potential(size, omega=0.02)
    data[:, :, :, 3] = psi_r**2 + psi_i**2
    return data


def init_quantum_tunneling(size, rng):
    """Wavepacket aimed at a potential barrier — watch part of it tunnel through.
    Classic quantum tunneling demonstration."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    c = size / 2.0
    sigma = size * 0.07

    # Wavepacket starts on the left, moving right
    cx = c - size * 0.2
    cy = c
    cz = c
    kx = rng.uniform(0.5, 1.2)  # rightward momentum

    psi_r, psi_i = _gaussian_wavepacket(size, cx, cy, cz, sigma, kx=kx)
    data[:, :, :, 0] = psi_r
    data[:, :, :, 1] = psi_i

    # Barrier: thin wall in the middle (Gaussian-shaped barrier along x)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    barrier_width = rng.uniform(1.5, 4.0)
    barrier_height = rng.uniform(0.3, 1.5)
    barrier = barrier_height * np.exp(-0.5 * ((x - c) / barrier_width)**2)
    data[:, :, :, 2] = barrier.astype(np.float32)
    data[:, :, :, 3] = psi_r**2 + psi_i**2
    return data


def init_quantum_double_slit(size, rng):
    """Wavepacket approaching a double-slit barrier — produces interference pattern.
    The barrier has two gaps that act as coherent sources."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    c = size / 2.0
    sigma = size * 0.07

    # Plane wave packet moving in +x direction
    cx = c - size * 0.25
    cy = c
    cz = c
    kx = rng.uniform(0.6, 1.0)

    psi_r, psi_i = _gaussian_wavepacket(size, cx, cy, cz, sigma, kx=kx)
    data[:, :, :, 0] = psi_r
    data[:, :, :, 1] = psi_i

    # Double slit barrier at x = center
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    barrier = np.zeros((size, size, size), dtype=np.float32)
    wall_x = int(c)
    wall_thick = 2
    # Wall (V=8 — good confinement without CFL issues)
    barrier[:, :, wall_x - wall_thick:wall_x + wall_thick] = 8.0
    # Cut two slits (along y-axis, centered in z)
    slit_sep = max(3, int(size * 0.15))
    slit_width = max(2, int(size * 0.04))
    for slit_off in [-slit_sep // 2, slit_sep // 2]:
        y_start = max(0, int(c + slit_off - slit_width // 2))
        y_end = min(size, int(c + slit_off + slit_width // 2))
        barrier[:, y_start:y_end, wall_x - wall_thick:wall_x + wall_thick] = 0.0

    data[:, :, :, 2] = barrier
    data[:, :, :, 3] = psi_r**2 + psi_i**2
    return data


def init_quantum_molecule(size, rng):
    """Two-center molecular orbital — electron in potential of two nuclei.
    Seeds with a 1s-like wavefunction centered between nuclei (bonding orbital)."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    c = size / 2.0
    sigma = size * 0.1

    # Wavepacket centered between the two nuclei
    psi_r, psi_i = _gaussian_wavepacket(size, c, c, c, sigma)
    data[:, :, :, 0] = psi_r
    data[:, :, :, 1] = psi_i

    # Two-center potential is computed dynamically in the shader,
    # but we pre-compute it here for the initial |Ψ|² rendering
    sep = size * 0.25  # default separation
    r_soft = 1.5
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    d1 = np.sqrt((x - c + sep/2 + 0.5)**2 + (y - c + 0.5)**2 + (z - c + 0.5)**2)
    d2 = np.sqrt((x - c - sep/2 + 0.5)**2 + (y - c + 0.5)**2 + (z - c + 0.5)**2)
    V = -1.0 / np.sqrt(d1**2 + r_soft**2) - 1.0 / np.sqrt(d2**2 + r_soft**2)
    data[:, :, :, 2] = V.astype(np.float32)
    data[:, :, :, 3] = psi_r**2 + psi_i**2
    return data


def init_quantum_antibonding(size, rng):
    """Antibonding molecular orbital — electron probability avoids the midpoint.
    Seeds with two out-of-phase Gaussians centered on each nucleus."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    c = size / 2.0
    sep = size * 0.25
    sigma = size * 0.07

    # Two Gaussians with opposite phase (antibonding = ψ_A - ψ_B)
    psi_r1, psi_i1 = _gaussian_wavepacket(size, c - sep/2, c, c, sigma)
    psi_r2, psi_i2 = _gaussian_wavepacket(size, c + sep/2, c, c, sigma)
    psi_r = psi_r1 - psi_r2  # destructive at midpoint
    psi_i = psi_i1 - psi_i2
    norm = np.sqrt(np.max(psi_r**2 + psi_i**2))
    if norm > 1e-10:
        psi_r /= norm
        psi_i /= norm

    data[:, :, :, 0] = psi_r
    data[:, :, :, 1] = psi_i

    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    d1 = np.sqrt((x - c + sep/2 + 0.5)**2 + (y - c + 0.5)**2 + (z - c + 0.5)**2)
    d2 = np.sqrt((x - c - sep/2 + 0.5)**2 + (y - c + 0.5)**2 + (z - c + 0.5)**2)
    V = -1.0 / np.sqrt(d1**2 + 1.5**2) - 1.0 / np.sqrt(d2**2 + 1.5**2)
    data[:, :, :, 2] = V.astype(np.float32)
    data[:, :, :, 3] = psi_r**2 + psi_i**2
    return data


def init_quantum_selfinteract(size, rng):
    """Wavepacket with self-interaction (Hartree mean-field).
    For the Schrödinger-Poisson shader — starts with Gaussian, potential=0."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    c = size / 2.0
    sigma = rng.uniform(size * 0.06, size * 0.12)

    # Offset slightly from center for asymmetry
    cx = c + rng.uniform(-size * 0.1, size * 0.1)
    cy = c + rng.uniform(-size * 0.1, size * 0.1)
    cz = c + rng.uniform(-size * 0.1, size * 0.1)

    psi_r, psi_i = _gaussian_wavepacket(size, cx, cy, cz, sigma)
    data[:, :, :, 0] = psi_r
    data[:, :, :, 1] = psi_i
    # V starts at 0 — the Poisson solver builds it from |Ψ|²
    data[:, :, :, 3] = psi_r**2 + psi_i**2
    return data


INIT_FUNCS = {
    'random_very_sparse': init_random_very_sparse,
    'random_sparse': init_random_sparse,
    'random_dense': init_random_dense,
    'random_smooth': init_random_smooth,
    'center_blob': init_center_blob,
    'crystal_seed': init_crystal_seed,
    'wave_pulse': init_wave_pulse,
    'gray_scott': init_gray_scott,
    'predator_prey': init_predator_prey,
    'kuramoto': init_kuramoto,
    'bz_reaction': init_bz_reaction,
    'barkley_excitable': init_barkley_excitable,
    'morphogen': init_morphogen,
    'flocking': init_flocking,
    'lenia_blobs': init_lenia_blobs,
    'lenia_multi': init_lenia_multi,
    'element_mix': init_element_mix,
    'sodium_water': init_sodium_water,
    'metal_layers': init_metal_layers,
    'phase_separation': init_phase_separation,
    'erosion_terrain': init_erosion_terrain,
    'mycelium': init_mycelium,
    'em_wave': init_em_wave,
    'viscous_fingers': init_viscous_fingers,
    'fire': init_fire,
    'physarum': init_physarum,
    'fracture': init_fracture,
    'galaxy': init_galaxy,
    'lichen': init_lichen,
    'predator_prey_separated': init_predator_prey_separated,
    'morphogen_hotspots': init_morphogen_hotspots,
    'bz_spiral_seed': init_bz_spiral_seed,
    'crystal_multi_seed': init_crystal_multi_seed,
    'erosion_ridges': init_erosion_ridges,
    'flocking_vortex': init_flocking_vortex,
    'game_of_life_centered': init_game_of_life_centered,
    'smoothlife_sparse': init_smoothlife_sparse,
    'gray_scott_worms_dense': init_gray_scott_worms_dense,
    'lenia_multi_colocated': init_lenia_multi_colocated,
    'kuramoto_clusters': init_kuramoto_clusters,
    'galaxy_filaments': init_galaxy_filaments,
    'lichen_dense': init_lichen_dense,
    'mycelium_foraging': init_mycelium_foraging,
    'bz_turbulence_high_amp': init_bz_turbulence_high_amp,
    'fire_sparse': init_fire_sparse,
    'phase_separation_quench': init_phase_separation_quench,
    'element_layered': init_element_layered,
    'quantum_hydrogen': init_quantum_hydrogen,
    'quantum_orbital': init_quantum_orbital,
    'quantum_wavepacket': init_quantum_wavepacket,
    'quantum_harmonic': init_quantum_harmonic,
    'quantum_tunneling': init_quantum_tunneling,
    'quantum_double_slit': init_quantum_double_slit,
    'quantum_molecule': init_quantum_molecule,
    'quantum_antibonding': init_quantum_antibonding,
    'quantum_selfinteract': init_quantum_selfinteract,
}

# Register individual orbital init variants
INIT_FUNCS.update(ORBITAL_INITS)


def init_sandbox_empty(size, rng):
    """Empty grid with a wall floor for sandbox mode."""
    data = np.zeros((size, size, size, 4), dtype=np.float32)
    # Wall floor at y=0 (shader Y axis = vertical/gravity)
    data[:, 0, :, 0] = float(WALL_ID)
    data[:, 0, :, 1] = 25.0
    data[:, 0, :, 2] = 0.0  # solid
    return data

INIT_FUNCS['sandbox_empty'] = init_sandbox_empty


# ── Simulator class ──────────────────────────────────────────────────

class Simulator:
    def __init__(self, size=64, rule='game_of_life_3d'):
        self.size = size
        self.rule_name = rule
        self.preset = RULE_PRESETS[rule]
        self.paused = True
        self.step_count = 0
        self.sim_speed = 1  # steps per batch
        self.target_sps = 0  # target steps/sec (0 = unlimited, run every frame)
        self._last_step_time = 0.0
        self.seed = 42
        # Idle-frame skip: when scene state (sim, camera, render knobs) is
        # unchanged from the previous frame, we blit a cached copy of the
        # last render instead of re-marching rays. Huge win when paused or
        # stepping slower than the render rate, which is the common case
        # at 384³/512³ where ray-march is 5–30 ms/frame.
        self._last_scene_hash = None
        self._scene_cache_fbo = None
        self._scene_cache_tex = None
        self._scene_cache_size = None

        # Camera
        self.cam_theta = 0.5      # horizontal angle
        self.cam_phi = 0.4        # vertical angle
        self.cam_dist = 2.5       # distance from center
        self.cam_target = np.array([0.5, 0.5, 0.5])

        # Rendering
        self.density_scale = 15.0
        self.brightness = 1.5
        self.render_mode = 0      # 0=volume, 1=iso, 2=MIP
        self.iso_threshold = 0.5
        self.colormap = 0
        self.slice_pos = -1.0     # -1 = disabled
        self.slice_axis = 2

        # Rendering mode: 'volumetric' or 'voxel'
        self.renderer_mode = self.preset.get('render_mode', 'volumetric')
        self.boundary_mode = _BOUNDARY_NAME_TO_MODE.get(
            self.preset.get('boundary', 'toroidal'), 0)
        self.is_element_ca = self.preset.get('is_element_ca', False)

        # Voxel rendering settings
        self.voxel_gap = 0.1       # gap between cubes (0=touching, 0.2=20% gap)
        self.voxel_threshold = 0.5  # visibility threshold for non-element CAs
        self.voxel_alpha = 1.0      # transparency
        self._cull_valid = False     # True when SSBO has valid cull data for current state

        # Visualization channel (for multi-field rules)
        self.vis_channels = self.preset.get('vis_channels', ['Value'])
        self.vis_channel = self.preset.get('vis_default', 0)
        self.vis_abs = self.preset.get('vis_abs', False)

        # Parameters (copy from preset)
        self.params = dict(self.preset['params'])
        self.dt = self.preset['dt']

        # Mouse state for orbit
        self.mouse_pressed = False
        self.mouse_right_pressed = False
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

        # Sandbox mode
        self.sandbox_mode = (rule == 'sandbox')
        self.paint_mode = False         # universal paint (for any CA)
        self.brush_tool = 0             # 0=element/place, 1=temperature, 2=eraser
        self.brush_size = 1             # radius in voxels
        self.brush_element = 26         # current element (Fe by default)
        self.brush_temp = 1000.0        # temperature brush value
        self.mouse_middle_pressed = False

        # Discovery / live scoring
        self.discoveries = []
        self.discovery_file = 'discoveries.json'
        self.discovery_index = -1  # -1 = not browsing
        self._load_discoveries()
        self._score = 0.0
        self._score_metrics = {}  # latest metrics dict
        self._score_frame = 0     # frames since last score update
        self._score_interval = max(20, (size // 32) * 15)  # scale with grid size
        self._prev_grid = None    # for activity measurement
        self._metric_history = []  # rolling window for scoring

        # Performance: cached CPU grid for brush/raycast (invalidated on sim step)
        self._cpu_grid = None
        self._cpu_grid_dirty = True

        # Video recording
        self._recording = False
        self._rec_process = None     # ffmpeg subprocess
        self._rec_start_time = 0.0
        self._rec_frame_count = 0
        self._rec_fps = 60
        self._rec_width = 2560       # output resolution (1440p)
        self._rec_height = 1440
        self._rec_filename = ''
        self._rec_msg = ''
        self._rec_msg_time = 0.0
        self._rec_fbo = None         # offscreen framebuffer
        self._rec_rbo_color = None
        self._rec_rbo_depth = None
        self._rec_write_thread = None
        self._rec_write_queue = None

        # Quick-access element palette
        self.palette_elements = [
            0, 1, 6, 7, 8, 11, 13, 14, 26, 29, 79, 74, 80, 50, 82, WALL_ID
        ]  # vacuum, H, C, N, O, Na, Al, Si, Fe, Cu, Au, W, Hg, Sn, Pb, Wall

        # Window
        self.width = 1280
        self.height = 800

        self._init_window()
        self._init_gl()
        self._init_volume()

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self.window = glfw.create_window(self.width, self.height,
                                          "3D Cellular Automata Simulator", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # vsync

        # imgui setup — GlfwRenderer handles context, fonts, input, and rendering
        imgui.create_context()
        self.imgui_renderer = GlfwRenderer(self.window)

        # Install additional callbacks AFTER GlfwRenderer (they chain)
        self._prev_mouse_button_cb = glfw.set_mouse_button_callback(self.window, self._mouse_button_cb)
        self._prev_cursor_pos_cb = glfw.set_cursor_pos_callback(self.window, self._cursor_pos_cb)
        self._prev_scroll_cb = glfw.set_scroll_callback(self.window, self._scroll_cb)
        self._prev_key_cb = glfw.set_key_callback(self.window, self._key_cb)

    def _init_gl(self):
        self.ctx = moderngl.create_context()

        # Detect nouveau — shared memory in compute shaders crashes on it
        renderer = self.ctx.info.get('GL_RENDERER', '').lower()
        vendor = self.ctx.info.get('GL_VENDOR', '').lower()
        self._use_shared_mem = 'nouveau' not in renderer and 'nouveau' not in vendor

        # Probe the actual per-texture allocation budget for this driver and
        # raise _TEX_ALLOC_LIMIT if we have a proper VRAM query. Keeps the
        # rgba32f → rgba16f fallback honest without over-restricting NVIDIA
        # cards that can comfortably hold 512³ rgba32f textures.
        global _TEX_ALLOC_LIMIT
        _TEX_ALLOC_LIMIT = _probe_tex_alloc_limit(self.ctx)

        # Determine texture format (rgba32f or rgba16f depending on grid size)
        self._tex_dtype, self._tex_np_dtype, self._tex_bpt, self._tex_glsl_fmt = \
            _tex_format_for_size(self.size)

        # Full-screen quad (for volumetric mode)
        vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)

        # Compile volumetric render shader
        self.render_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )
        # Cache volumetric program uniform references
        rp = self.render_prog
        # Many of these are None-safe because the GLSL compiler strips any
        # uniform not referenced in the active control flow (e.g. u_volume
        # is no longer referenced by the fragment shader since sample_vol
        # reads from u_view_tex).
        self._rp_u_volume = rp.get('u_volume', None)
        self._rp_u_size = rp.get('u_size', None)
        self._rp_u_camera_pos = rp['u_camera_pos']
        self._rp_u_camera_rot = rp['u_camera_rot']
        self._rp_u_fov = rp['u_fov']
        self._rp_u_aspect = rp['u_aspect']
        self._rp_u_density_scale = rp['u_density_scale']
        self._rp_u_brightness = rp['u_brightness']
        self._rp_u_render_mode = rp['u_render_mode']
        self._rp_u_iso_threshold = rp['u_iso_threshold']
        self._rp_u_slice_pos = rp['u_slice_pos']
        self._rp_u_slice_axis = rp['u_slice_axis']
        self._rp_u_colormap = rp['u_colormap']
        self._rp_u_vis_channel = rp.get('u_vis_channel', None)
        self._rp_u_vis_abs = rp.get('u_vis_abs', None)
        # New acceleration uniforms (use .get() — GLSL compiler may strip unused ones)
        self._rp_u_frame_id = rp.get('u_frame_id', None)
        self._rp_u_occupancy = rp.get('u_occupancy', None)
        self._rp_u_occ_size = rp.get('u_occ_size', None)
        self._rp_u_use_occupancy = rp.get('u_use_occupancy', None)
        self._rp_u_minmax_mip = rp.get('u_minmax_mip', None)
        self._rp_u_minmax_size = rp.get('u_minmax_size', None)
        self._rp_u_use_minmax = rp.get('u_use_minmax', None)
        self._rp_u_view_tex = rp.get('u_view_tex', None)
        # Bind acceleration textures to fixed units
        if self._rp_u_occupancy is not None:
            self._rp_u_occupancy.value = 1   # texture unit 1
        if self._rp_u_minmax_mip is not None:
            self._rp_u_minmax_mip.value = 2  # texture unit 2
        if self._rp_u_view_tex is not None:
            self._rp_u_view_tex.value = 3    # texture unit 3
        self.vao = self.ctx.simple_vertex_array(self.render_prog, self.vbo, 'in_pos')

        # Compile upsample shader (for half-res rendering)
        self._upsample_prog = self.ctx.program(
            vertex_shader=UPSAMPLE_VERTEX_SHADER,
            fragment_shader=UPSAMPLE_FRAGMENT_SHADER,
        )
        self._up_u_half_res = self._upsample_prog['u_half_res']
        self._up_u_texel_size = self._upsample_prog['u_texel_size']
        self._up_u_half_res.value = 0
        self._upsample_vao = self.ctx.simple_vertex_array(self._upsample_prog, self.vbo, 'in_pos')

        # Compile compute raymarcher
        self._compute_ray_prog = self.ctx.compute_shader(COMPUTE_RAYMARCH_SHADER)
        self._init_compute_ray_uniforms()

        # Compile fused occupancy + minmax build shader.
        # The two used to be separate dispatches that each did a full pass
        # over the source texture; they're now a single shader that writes
        # both outputs from one read pass.
        self._accel_prog = self.ctx.compute_shader(ACCEL_FUSED_SHADER)

        # Compile the view-texture baker (full-res R16F scalar).
        self._view_build_prog = self.ctx.compute_shader(VIEW_TEX_BUILD_SHADER)

        # Acceleration texture state
        self._occ_block_size = 4  # default; recomputed per-size in _alloc_accel_textures
        self._accel_textures_valid = False
        self._frame_counter = 0
        self._half_res_fbo = None
        self._half_res_tex = None
        self._use_half_res = False
        self._use_compute_ray = False

        # Compile voxel render shader
        self.voxel_prog = self.ctx.program(
            vertex_shader=VOXEL_VERTEX_SHADER,
            fragment_shader=VOXEL_FRAGMENT_SHADER,
        )
        # Cache voxel program uniform references
        vp = self.voxel_prog
        self._vp_u_size = vp['u_size']
        self._vp_u_view_proj = vp['u_view_proj']
        self._vp_u_voxel_gap = vp['u_voxel_gap']
        self._vp_u_camera_pos = vp['u_camera_pos']
        self._vp_u_brightness = vp['u_brightness']
        self._vp_u_colormap = vp['u_colormap']
        self._vp_u_is_element_ca = vp['u_is_element_ca']
        self._vp_u_alpha = vp['u_alpha']
        # New uniforms for texture-sampled color in vertex shader
        self._vp_u_channel = vp['u_channel']
        self._vp_u_use_abs = vp['u_use_abs']
        self._vp_u_threshold = vp['u_threshold']
        self._vp_u_volume_tex = vp['u_volume_tex']
        self._vp_u_volume_tex.value = 0  # texture unit 0
        # Empty VAO for instanced rendering (vertices come from SSBO)
        self.voxel_vao = self.ctx.vertex_array(self.voxel_prog, [])

        # Compile voxel cull compute shader
        self._compile_cull()

        # Compile GPU metrics reduction shader + tiny SSBO (4 uints = 16 bytes)
        self._compile_metrics()
        self._metrics_ssbo = self.ctx.buffer(reserve=16)

        # Voxel SSBOs — size dynamically based on grid
        self._alloc_voxel_buffer(self.size)

        # Element property SSBO
        self.element_ssbo = self.ctx.buffer(data=ELEMENT_GPU_DATA.tobytes())

        # Compile compute shader for current rule
        self._compile_compute()
        self._cache_compute_uniforms()

    def _alloc_voxel_buffer(self, size):
        """Allocate voxel SSBO (1 uint = 4 bytes per voxel) with multi-pass chunking.
        Small grids (≤256³): single pass, full allocation, zero clipping.
        Large grids: split into spatial chunks, each chunk processed in its own
        cull+draw pass reusing the same SSBO. No voxels are ever dropped."""
        total_voxels = size * size * size
        bytes_per_voxel = 4  # 1 uint (position-only packed format)
        ssbo_max = 128 * 1024 * 1024  # 128 MB typical minimum
        try:
            ssbo_max = self.ctx.info.get("GL_MAX_SHADER_STORAGE_BLOCK_SIZE", ssbo_max)
        except Exception:
            pass
        max_budget = ssbo_max // bytes_per_voxel  # 128MB / 4 = 32M voxels

        if total_voxels <= max_budget:
            # Full grid fits in one SSBO — single pass, no chunking
            self._voxel_chunks_per_dim = 1
            max_voxels = total_voxels
        else:
            # Multi-pass: find smallest chunks_per_dim where each chunk fits
            for cpd in (2, 3, 4, 5, 6):
                chunk_edge = (size + cpd - 1) // cpd
                chunk_vol = chunk_edge ** 3
                if chunk_vol <= max_budget:
                    break
            self._voxel_chunks_per_dim = cpd
            # SSBO sized for one chunk (the largest possible chunk)
            chunk_edge = (size + cpd - 1) // cpd
            max_voxels = chunk_edge ** 3

        self.voxel_budget_capped = False  # multi-pass means no clipping
        self.max_voxels = max_voxels
        if hasattr(self, 'voxel_buffer'):
            self.voxel_buffer.release()
            self.voxel_indirect_buffer.release()
        if hasattr(self, '_voxel_total_counter'):
            self._voxel_total_counter.release()
        self.voxel_buffer = self.ctx.buffer(reserve=max_voxels * bytes_per_voxel)
        # Indirect draw command: [vertexCount=36, instanceCount=0, first=0, baseInstance=0]
        self._indirect_cmd = np.array([36, 0, 0, 0], dtype=np.uint32)
        self.voxel_indirect_buffer = self.ctx.buffer(data=self._indirect_cmd.tobytes())
        # Pre-allocate reset bytes (avoids .tobytes() every frame)
        reset = np.array([36, 0, 0, 0], dtype=np.uint32)
        self._indirect_reset_bytes = reset.tobytes()
        # Total visible counter (1 uint, accumulates across all chunks per frame)
        self._voxel_total_counter = self.ctx.buffer(data=np.array([0], dtype=np.uint32).tobytes())

    def _compile_cull(self):
        """Compile voxel cull shader (uses sampler3D, no format dependency)."""
        if hasattr(self, 'voxel_cull_prog'):
            self.voxel_cull_prog.release()
        self.voxel_cull_prog = self.ctx.compute_shader(VOXEL_CULL_SHADER)
        # Cache cull shader uniform references
        self._cull_u_volume_tex = self.voxel_cull_prog['u_volume_tex']
        self._cull_u_volume_tex.value = 0  # texture unit 0
        self._cull_u_size = self.voxel_cull_prog['u_size']
        self._cull_u_threshold = self.voxel_cull_prog['u_threshold']
        self._cull_u_is_element_ca = self.voxel_cull_prog['u_is_element_ca']
        self._cull_u_max_voxels = self.voxel_cull_prog['u_max_voxels']
        self._cull_u_channel = self.voxel_cull_prog['u_channel']
        self._cull_u_use_abs = self.voxel_cull_prog['u_use_abs']
        self._cull_u_chunk_min = self.voxel_cull_prog['u_chunk_min']
        self._cull_u_chunk_max = self.voxel_cull_prog['u_chunk_max']
        # Compile indirect reset shader (one-time, shared across recompiles)
        if not hasattr(self, '_indirect_reset_prog'):
            self._indirect_reset_prog = self.ctx.compute_shader(INDIRECT_RESET_SHADER)
            self._indirect_reset_u_reset_total = self._indirect_reset_prog['u_reset_total']

    def _init_compute_ray_uniforms(self):
        """Cache uniform references for the compute raymarcher."""
        cr = self._compute_ray_prog
        # Use .get() for all uniforms — GLSL compiler may strip unused ones
        self._cr_u_output = cr.get('u_output', None)
        self._cr_u_volume = cr.get('u_volume', None)
        self._cr_u_size = cr.get('u_size', None)
        self._cr_u_camera_pos = cr.get('u_camera_pos', None)
        self._cr_u_camera_rot = cr.get('u_camera_rot', None)
        self._cr_u_fov = cr.get('u_fov', None)
        self._cr_u_density_scale = cr.get('u_density_scale', None)
        self._cr_u_brightness = cr.get('u_brightness', None)
        self._cr_u_render_mode = cr.get('u_render_mode', None)
        self._cr_u_iso_threshold = cr.get('u_iso_threshold', None)
        self._cr_u_colormap = cr.get('u_colormap', None)
        self._cr_u_vis_channel = cr.get('u_vis_channel', None)
        self._cr_u_vis_abs = cr.get('u_vis_abs', None)
        self._cr_u_aspect = cr.get('u_aspect', None)
        self._cr_u_frame_id = cr.get('u_frame_id', None)
        self._cr_u_resolution = cr.get('u_resolution', None)
        self._cr_u_occupancy = cr.get('u_occupancy', None)
        self._cr_u_occ_size = cr.get('u_occ_size', None)
        self._cr_u_use_occupancy = cr.get('u_use_occupancy', None)
        self._cr_u_minmax_mip = cr.get('u_minmax_mip', None)
        self._cr_u_minmax_size = cr.get('u_minmax_size', None)
        self._cr_u_use_minmax = cr.get('u_use_minmax', None)
        self._cr_u_view_tex = cr.get('u_view_tex', None)
        # Bind to fixed texture units
        if self._cr_u_volume is not None:
            self._cr_u_volume.value = 0
        if self._cr_u_occupancy is not None:
            self._cr_u_occupancy.value = 1
        if self._cr_u_minmax_mip is not None:
            self._cr_u_minmax_mip.value = 2
        if self._cr_u_view_tex is not None:
            self._cr_u_view_tex.value = 3

    def _alloc_accel_textures(self, size):
        """Allocate or reallocate occupancy bitmap, min/max mipmap, and
        reduced-precision view texture."""
        # Adaptive brick size: coarser bricks at larger grids halve the DDA
        # traversal cost. At 512³ with bs=8 the occupancy grid is 64³ (512k
        # cells) instead of 128³ (2M cells) — ~2× faster empty-space skip,
        # with negligible over-march from the larger bricks (a brick with a
        # single active voxel is 512 voxels wide vs 64, which the per-voxel
        # marching loop handles cheaply thanks to the R16F view tex).
        self._occ_block_size = 8 if size >= 192 else 4
        bs = self._occ_block_size
        occ_dim = (size + bs - 1) // bs

        # Release old textures
        if hasattr(self, '_occ_tex') and self._occ_tex is not None:
            self._occ_tex.release()
        if hasattr(self, '_mm_tex') and self._mm_tex is not None:
            self._mm_tex.release()
        if hasattr(self, '_view_tex') and self._view_tex is not None:
            self._view_tex.release()

        # Occupancy: R8UI (1 byte per block)
        occ_data = bytes(occ_dim ** 3)
        self._occ_tex = self.ctx.texture3d(
            (occ_dim, occ_dim, occ_dim), 1, occ_data, dtype='u1')
        self._occ_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._occ_dim = occ_dim

        # Min/max mipmap: RG16F (4 bytes per block: min + max)
        mm_data = bytes(occ_dim ** 3 * 4)
        self._mm_tex = self.ctx.texture3d(
            (occ_dim, occ_dim, occ_dim), 2, mm_data, dtype='f2')
        self._mm_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._mm_dim = occ_dim

        # Reduced-precision view tex: single-channel R16F, full voxel res.
        # Trilinear filter so ray-march sampling interpolates smoothly.
        view_data = bytes(size ** 3 * 2)  # 2 bytes per voxel
        self._view_tex = self.ctx.texture3d(
            (size, size, size), 1, view_data, dtype='f2')
        self._view_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._view_tex.repeat_x = False
        self._view_tex.repeat_y = False
        self._view_tex.repeat_z = False

        self._accel_textures_valid = False

    def _build_accel_textures(self):
        """Dispatch occupancy and min/max compute shaders to build acceleration structures.
        Called once per sim step (cheap — typically (size/4)³ threads)."""
        src_tex = self.tex_a if self.ping == 0 else self.tex_b
        size = self.size
        bs = self._occ_block_size
        occ_dim = self._occ_dim

        # Determine channel / abs mode from current vis settings
        shader = self.preset['shader']
        is_elem = self.preset.get('is_element_ca', False)
        if is_elem:
            channel, use_abs = 0, 0
            threshold = 0.5
        elif shader in ('wave_3d', 'em_wave_3d'):
            channel, use_abs = self.vis_channel, 1
            threshold = 0.005
        else:
            channel, use_abs = self.vis_channel, 0
            threshold = 0.01

        src_tex.use(location=0)

        # ── Stage 1: Reduced-precision view tex ──
        # Voxel-wise pass that reads the heavy RGBA32F source ONCE,
        # writes a single R16F scalar per voxel. Downstream consumers
        # (ray-march + occupancy build) read the cheaper R16F.
        self._view_tex.bind_to_image(0, read=False, write=True)
        self._view_build_prog['u_volume'].value = 0
        self._view_build_prog['u_size'].value = size
        self._view_build_prog['u_channel'].value = channel
        self._view_build_prog['u_use_abs'].value = use_abs
        vgroups = (size + 3) // 4
        self._view_build_prog.run(vgroups, vgroups, vgroups)

        # Ensure view-tex writes are visible to the next texture sample
        GL.glMemoryBarrier(GL.GL_TEXTURE_FETCH_BARRIER_BIT |
                           GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # ── Stage 2: Fused occupancy + min/max (reads from view tex) ──
        # Reads 1/8 the bytes vs. the previous RGBA32F direct-read variant
        # since channel-select + abs are already baked into the view tex.
        self._view_tex.use(location=0)
        self._occ_tex.bind_to_image(0, read=False, write=True)
        self._mm_tex.bind_to_image(1, read=False, write=True)
        self._accel_prog['u_view'].value = 0
        self._accel_prog['u_size'].value = size
        self._accel_prog['u_block_size'].value = bs
        self._accel_prog['u_threshold'].value = threshold
        groups = (occ_dim + 3) // 4
        self._accel_prog.run(groups, groups, groups)

        GL.glMemoryBarrier(GL.GL_TEXTURE_FETCH_BARRIER_BIT |
                           GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        self._accel_textures_valid = True

    def _alloc_half_res_fbo(self, width, height):
        """Create or recreate the half-resolution FBO for volume rendering."""
        hw, hh = max(width // 2, 1), max(height // 2, 1)
        if hasattr(self, '_half_res_tex') and self._half_res_tex is not None:
            self._half_res_tex.release()
        if hasattr(self, '_half_res_fbo') and self._half_res_fbo is not None:
            self._half_res_fbo.release()
        self._half_res_tex = self.ctx.texture((hw, hh), 4, dtype='f1')
        self._half_res_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._half_res_fbo = self.ctx.framebuffer(color_attachments=[self._half_res_tex])
        self._half_res_size = (hw, hh)

    def _alloc_compute_ray_output(self, width, height):
        """Create or recreate the 2D output texture for compute raymarcher."""
        if hasattr(self, '_cr_output_tex') and self._cr_output_tex is not None:
            self._cr_output_tex.release()
        if hasattr(self, '_cr_fbo') and self._cr_fbo is not None:
            self._cr_fbo.release()
        self._cr_output_tex = self.ctx.texture((width, height), 4, dtype='f1')
        self._cr_output_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._cr_fbo = self.ctx.framebuffer(color_attachments=[self._cr_output_tex])
        self._cr_output_size = (width, height)

    def _compile_metrics(self):
        """Compile GPU metrics reduction shader with the current texture format."""
        if hasattr(self, '_metrics_prog'):
            self._metrics_prog.release()
        source = METRICS_REDUCE_SHADER.replace('rgba32f', self._tex_glsl_fmt)
        self._metrics_prog = self.ctx.compute_shader(source)

    def _compile_compute(self):
        shader_key = self.preset['shader']
        if shader_key == 'element_ca':
            source = ELEMENT_COMPUTE_HEADER + ELEMENT_CA_RULE
        else:
            source = COMPUTE_HEADER + CA_RULES[shader_key]
        # Inject shared memory toggle (disabled on nouveau to avoid driver crashes)
        smem = '1' if self._use_shared_mem else '0'
        source = source.replace('#version 430', '#version 430\n#define USE_SHARED_MEM ' + smem)
        # Match shader image format to texture format
        if self._tex_glsl_fmt != 'rgba32f':
            source = source.replace('rgba32f', self._tex_glsl_fmt)
        if hasattr(self, 'compute_prog'):
            self.compute_prog.release()
        self.compute_prog = self.ctx.compute_shader(source)

    def _init_volume(self):
        rng = np.random.RandomState(self.seed)
        init_name = getattr(self, '_current_init', self.preset['init'])
        init_func = INIT_FUNCS.get(init_name, init_random_sparse)
        data = init_func(self.size, rng)

        # Convert to the active texture dtype (float32 for small grids, float16 for large)
        data_bytes = np.ascontiguousarray(data.astype(self._tex_np_dtype)).tobytes()
        self.tex_a = self.ctx.texture3d((self.size, self.size, self.size), 4,
                                         data_bytes, dtype=self._tex_dtype)
        del data, data_bytes  # free before allocating second texture
        zeros = bytes(self.size ** 3 * self._tex_bpt)
        self.tex_b = self.ctx.texture3d((self.size, self.size, self.size), 4,
                                         zeros, dtype=self._tex_dtype)
        del zeros
        self.tex_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.ping = 0
        # Allocate acceleration structures for new volume
        self._alloc_accel_textures(self.size)

    def _get_camera_pos(self):
        x = self.cam_dist * math.cos(self.cam_phi) * math.sin(self.cam_theta)
        y = self.cam_dist * math.sin(self.cam_phi)
        z = self.cam_dist * math.cos(self.cam_phi) * math.cos(self.cam_theta)
        return self.cam_target + np.array([x, y, z])

    def _get_camera_rot(self):
        pos = self._get_camera_pos()
        forward = self.cam_target - pos
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        norm = np.linalg.norm(right)
        if norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / norm
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        return np.column_stack([right, up, -forward])

    # ── Raycasting & Brush ────────────────────────────────────────────

    def _screen_to_ray(self, mx, my):
        """Convert screen mouse coords to a world-space ray (origin, direction)."""
        # NDC coords ([-1, 1])
        ndc_x = (2.0 * mx / max(self.width, 1)) - 1.0
        ndc_y = 1.0 - (2.0 * my / max(self.height, 1))

        # Camera ray in world space
        cam_pos = self._get_camera_pos()
        cam_rot = self._get_camera_rot()
        right = cam_rot[:, 0]
        up = cam_rot[:, 1]
        forward = -cam_rot[:, 2]  # forward = -z column

        aspect = self.width / max(self.height, 1)
        fov_y = 1.0
        half_h = math.tan(fov_y * 0.5)
        half_w = half_h * aspect

        direction = forward + right * (ndc_x * half_w) + up * (ndc_y * half_h)
        direction = direction / np.linalg.norm(direction)
        return cam_pos, direction

    def _raycast_voxel(self, mx, my):
        """Cast ray from mouse into voxel grid, return (hit, grid_pos) using DDA.
        The voxel grid occupies world-space [0,1]^3."""
        origin, direction = self._screen_to_ray(mx, my)
        size = self.size

        # Ray-box intersection with [0, 1]^3
        inv_dir = np.where(np.abs(direction) > 1e-10,
                           1.0 / direction,
                           np.sign(direction) * 1e10)
        t_min_v = (0.0 - origin) * inv_dir
        t_max_v = (1.0 - origin) * inv_dir

        t1 = np.minimum(t_min_v, t_max_v)
        t2 = np.maximum(t_min_v, t_max_v)

        t_near = np.max(t1)
        t_far = np.min(t2)

        if t_near > t_far or t_far < 0:
            return False, None

        # Entry point in world space
        t_start = max(t_near, 0.0) + 1e-5
        entry = origin + direction * t_start

        # Convert to grid coordinates [0, size)
        gx, gy, gz = entry[0] * size, entry[1] * size, entry[2] * size

        # DDA traversal
        ix, iy, iz = int(gx), int(gy), int(gz)
        # Clamp to grid
        ix = max(0, min(size - 1, ix))
        iy = max(0, min(size - 1, iy))
        iz = max(0, min(size - 1, iz))

        step_x = 1 if direction[0] >= 0 else -1
        step_y = 1 if direction[1] >= 0 else -1
        step_z = 1 if direction[2] >= 0 else -1

        # tMax: t at which ray crosses next voxel boundary
        # tDelta: t to traverse one voxel in each axis
        dx = direction[0] * size
        dy = direction[1] * size
        dz = direction[2] * size

        t_delta_x = abs(1.0 / dx) if abs(dx) > 1e-10 else 1e10
        t_delta_y = abs(1.0 / dy) if abs(dy) > 1e-10 else 1e10
        t_delta_z = abs(1.0 / dz) if abs(dz) > 1e-10 else 1e10

        t_max_x = ((ix + (1 if step_x > 0 else 0)) - gx) / dx if abs(dx) > 1e-10 else 1e10
        t_max_y = ((iy + (1 if step_y > 0 else 0)) - gy) / dy if abs(dy) > 1e-10 else 1e10
        t_max_z = ((iz + (1 if step_z > 0 else 0)) - gz) / dz if abs(dz) > 1e-10 else 1e10

        # Read current grid state to check for occupied cells (use CPU cache)
        if self._cpu_grid_dirty or self._cpu_grid is None:
            src_tex = self.tex_a if self.ping == 0 else self.tex_b
            self._cpu_grid = np.frombuffer(src_tex.read(), dtype=self._tex_np_dtype).reshape(size, size, size, 4).copy()
            self._cpu_grid_dirty = False
        grid = self._cpu_grid

        # March through grid
        prev_empty = None
        max_steps = size * 3
        for _ in range(max_steps):
            if 0 <= ix < size and 0 <= iy < size and 0 <= iz < size:
                cell_id = grid[iz, iy, ix, 0]
                if self.is_element_ca:
                    is_occupied = int(round(cell_id)) > 0
                else:
                    is_occupied = cell_id > 0.5

                if self.brush_tool == 0:
                    # Element brush: place in the empty cell in front of a surface
                    if is_occupied:
                        target = prev_empty if prev_empty else (ix, iy, iz)
                        return True, target
                    prev_empty = (ix, iy, iz)
                elif self.brush_tool == 1:
                    # Temperature brush: target occupied cells
                    if is_occupied:
                        return True, (ix, iy, iz)
                elif self.brush_tool == 2:
                    # Eraser: target occupied cells
                    if is_occupied:
                        return True, (ix, iy, iz)

            # Step to next voxel
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    ix += step_x
                    t_max_x += t_delta_x
                else:
                    iz += step_z
                    t_max_z += t_delta_z
            else:
                if t_max_y < t_max_z:
                    iy += step_y
                    t_max_y += t_delta_y
                else:
                    iz += step_z
                    t_max_z += t_delta_z

            if ix < 0 or ix >= size or iy < 0 or iy >= size or iz < 0 or iz >= size:
                break

        return False, None

    def _apply_brush(self, gx, gy, gz):
        """Apply the current brush tool at grid position (gx, gy, gz) with brush_size radius.

        Performance: only the touched sub-region is written back to the GPU
        texture (via texture3d.write(viewport=...)) instead of the whole
        size³ × 16-byte buffer. At size=256 that's 64 KiB written per stroke
        instead of 256 MiB — roughly a 4000× reduction in PCIe traffic for
        a brush_size=2 stroke and removes the GPU pipeline stall that made
        rapid painting visibly stutter.
        """
        size = self.size
        src_tex = self.tex_a if self.ping == 0 else self.tex_b

        # Use CPU grid cache; fetch only if stale
        if self._cpu_grid_dirty or self._cpu_grid is None:
            self._cpu_grid = np.frombuffer(src_tex.read(), dtype=self._tex_np_dtype).reshape(size, size, size, 4).copy()
            self._cpu_grid_dirty = False
        grid = self._cpu_grid

        r = self.brush_size
        # Bounding box of the brush, clamped to the grid. We'll write back
        # only this sub-region rather than the whole texture.
        x0 = max(0, gx - r); x1 = min(size, gx + r + 1)
        y0 = max(0, gy - r); y1 = min(size, gy + r + 1)
        z0 = max(0, gz - r); z1 = min(size, gz + r + 1)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            return  # brush entirely outside grid

        for dz in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx*dx + dy*dy + dz*dz > r*r:
                        continue
                    x, y, z = gx + dx, gy + dy, gz + dz
                    if 0 <= x < size and 0 <= y < size and 0 <= z < size:
                        if self.is_element_ca:
                            self._apply_element_brush(grid, x, y, z)
                        else:
                            self._apply_generic_brush(grid, x, y, z)

        # Upload only the touched sub-region. The grid is stored as
        # [z, y, x, channel]; texture3d.write expects a contiguous block
        # in (x, y, z) order matching the GL viewport (x_off, y_off, z_off,
        # width, height, depth). np.ascontiguousarray ensures the slice
        # is packed for the upload without a hidden full-grid copy.
        sub = np.ascontiguousarray(grid[z0:z1, y0:y1, x0:x1, :])
        src_tex.write(
            sub.tobytes(),
            viewport=(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0),
        )
        self._cull_valid = False

    def _apply_element_brush(self, grid, x, y, z):
        """Apply brush for element CA at grid[z,y,x]."""
        if self.brush_tool == 0:
            elem_id = self.brush_element
            grid[z, y, x, 0] = float(elem_id)
            if elem_id == 0:
                grid[z, y, x, 1:] = 0.0
            elif elem_id == WALL_ID:
                grid[z, y, x, 1] = 25.0
                grid[z, y, x, 2] = 0.0
                grid[z, y, x, 3] = 0.0
            else:
                ambient = self.params.get('Temperature', 25.0)
                mp = ELEMENT_GPU_DATA[elem_id, 4]
                bp = ELEMENT_GPU_DATA[elem_id, 5]
                grid[z, y, x, 1] = ambient
                if ambient < mp:
                    grid[z, y, x, 2] = 0.0
                elif ambient < bp:
                    grid[z, y, x, 2] = 1.0
                else:
                    grid[z, y, x, 2] = 2.0
                grid[z, y, x, 3] = 0.0
        elif self.brush_tool == 1:
            cell_id = int(round(grid[z, y, x, 0]))
            if cell_id > 0 and cell_id != WALL_ID:
                grid[z, y, x, 1] = self.brush_temp
                mp = ELEMENT_GPU_DATA[cell_id, 4]
                bp = ELEMENT_GPU_DATA[cell_id, 5]
                if self.brush_temp < mp:
                    grid[z, y, x, 2] = 0.0
                elif self.brush_temp < bp:
                    grid[z, y, x, 2] = 1.0
                else:
                    grid[z, y, x, 2] = 2.0
        elif self.brush_tool == 2:
            cell_id = int(round(grid[z, y, x, 0]))
            if cell_id > 0:
                grid[z, y, x] = 0.0

    def _apply_generic_brush(self, grid, x, y, z):
        """Apply brush for generic (non-element) CAs at grid[z,y,x]."""
        if self.brush_tool == 0:
            # Place: set channel 0 to 1.0
            grid[z, y, x, 0] = 1.0
        elif self.brush_tool == 2:
            # Erase: zero all channels
            grid[z, y, x] = 0.0

    def _save_state(self, filepath):
        """Save current grid state to a .npz file."""
        src_tex = self.tex_a if self.ping == 0 else self.tex_b
        grid = np.frombuffer(src_tex.read(), dtype=self._tex_np_dtype).reshape(
            self.size, self.size, self.size, 4).astype(np.float32).copy()
        # Save params as JSON string to preserve key names (np.savez can't store dicts)
        params_json = json.dumps({k: float(v) for k, v in self.params.items()})
        np.savez_compressed(filepath,
                            grid=grid,
                            rule=self.rule_name,
                            size=self.size,
                            step=self.step_count,
                            params=np.array(list(self.params.values()), dtype=np.float32),
                            params_json=np.array(params_json),
                            dt=self.dt)

    def _load_state(self, filepath):
        """Load grid state from a .npz file."""
        self._cull_valid = False
        data = np.load(filepath, allow_pickle=False)
        grid = data['grid']
        loaded_size = int(data['size'])

        # Validate grid shape
        expected_shape = (loaded_size, loaded_size, loaded_size, 4)
        if grid.shape != expected_shape:
            print(f"[load_state] ERROR: grid shape {grid.shape} != expected {expected_shape}")
            return

        # Restore rule if saved and different from current
        loaded_rule = str(data['rule']) if 'rule' in data else None
        if loaded_rule and loaded_rule in RULE_PRESETS and loaded_rule != self.rule_name:
            self._change_rule(loaded_rule)

        # If size doesn't match, fully resize all GPU resources
        if loaded_size != self.size:
            self.tex_a.release()
            self.tex_b.release()
            self.size = loaded_size
            self._tex_dtype, self._tex_np_dtype, self._tex_bpt, self._tex_glsl_fmt = \
                _tex_format_for_size(loaded_size)
            self._compile_compute()
            self._cache_compute_uniforms()
            self._compile_cull()
            self._alloc_voxel_buffer(loaded_size)
            self._compile_metrics()
            self._init_volume()
            self._score_interval = max(20, (loaded_size // 32) * 15)

        # Restore params from JSON (preferred) or fall back to positional array
        if 'params_json' in data:
            saved_params = json.loads(str(data['params_json']))
            for k, v in saved_params.items():
                if k in self.params:
                    self.params[k] = v
        elif 'params' in data:
            # Legacy: positional values only — apply in order to matching keys
            saved_vals = data['params']
            for i, k in enumerate(self.params):
                if i < len(saved_vals):
                    self.params[k] = float(saved_vals[i])

        # Restore dt
        if 'dt' in data:
            self.dt = float(data['dt'])

        # Write loaded data into the current source texture
        self.ping = 0
        src_tex = self.tex_a
        src_tex.write(grid.astype(self._tex_np_dtype).tobytes())
        self.step_count = int(data['step'])

        # Reset scoring state for the new grid
        self._score_frame = 0
        self._metric_history = []
        self._prev_grid = None
        self._cpu_grid_dirty = True
        self._metrics_in_flight = False

    # ── Input callbacks ───────────────────────────────────────────────

    def _mouse_button_cb(self, window, button, action, mods):
        # Let imgui process first
        if self._prev_mouse_button_cb:
            self._prev_mouse_button_cb(window, button, action, mods)

        if imgui.get_io().want_capture_mouse:
            return

        can_paint = self.sandbox_mode or self.paint_mode

        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_pressed = (action == glfw.PRESS)
            if can_paint and action == glfw.PRESS:
                x, y = glfw.get_cursor_pos(window)
                self._sandbox_paint(x, y)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse_right_pressed = (action == glfw.PRESS)
            if can_paint and action == glfw.PRESS:
                # Right-click erases
                x, y = glfw.get_cursor_pos(window)
                old_tool = self.brush_tool
                self.brush_tool = 2
                self._sandbox_paint(x, y)
                self.brush_tool = old_tool
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.mouse_middle_pressed = (action == glfw.PRESS)
        x, y = glfw.get_cursor_pos(window)
        self.last_mouse_x = x
        self.last_mouse_y = y

    def _sandbox_paint(self, mx, my):
        """Attempt to paint at the mouse position."""
        hit, pos = self._raycast_voxel(mx, my)
        if hit and pos:
            self._apply_brush(*pos)

    def _cursor_pos_cb(self, window, x, y):
        if self._prev_cursor_pos_cb:
            self._prev_cursor_pos_cb(window, x, y)

        if imgui.get_io().want_capture_mouse:
            return

        dx = x - self.last_mouse_x
        dy = y - self.last_mouse_y
        self.last_mouse_x = x
        self.last_mouse_y = y

        can_paint = self.sandbox_mode or self.paint_mode

        if can_paint:
            # Paint mode: left-drag paints, middle-drag orbits
            if self.mouse_pressed:
                self._sandbox_paint(x, y)
                return
            if self.mouse_middle_pressed:
                self.cam_theta -= dx * 0.005
                self.cam_phi += dy * 0.005
                self.cam_phi = max(-1.5, min(1.5, self.cam_phi))
                return
            if self.mouse_right_pressed:
                old_tool = self.brush_tool
                self.brush_tool = 2
                self._sandbox_paint(x, y)
                self.brush_tool = old_tool
                return
        else:
            # Orbit mode: left-drag orbits, right-drag pans
            if self.mouse_pressed:
                self.cam_theta -= dx * 0.005
                self.cam_phi += dy * 0.005
                self.cam_phi = max(-1.5, min(1.5, self.cam_phi))

            if self.mouse_right_pressed:
                rot = self._get_camera_rot()
                right = rot[:, 0]
                up = rot[:, 1]
                self.cam_target -= right * dx * 0.002 * self.cam_dist
                self.cam_target += up * dy * 0.002 * self.cam_dist

    def _scroll_cb(self, window, xoffset, yoffset):
        if self._prev_scroll_cb:
            self._prev_scroll_cb(window, xoffset, yoffset)

        if imgui.get_io().want_capture_mouse:
            return

        self.cam_dist *= 0.9 if yoffset > 0 else 1.1
        self.cam_dist = max(0.5, min(10.0, self.cam_dist))

    def _key_cb(self, window, key, scancode, action, mods):
        if self._prev_key_cb:
            self._prev_key_cb(window, key, scancode, action, mods)

        if imgui.get_io().want_capture_keyboard:
            return

        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                self.paused = not self.paused
            elif key == glfw.KEY_R:
                self._reset()
            elif key == glfw.KEY_RIGHT and self.paused:
                self._step_sim()
            elif key == glfw.KEY_V:
                self.renderer_mode = 'voxel' if self.renderer_mode == 'volumetric' else 'volumetric'
            elif key == glfw.KEY_B:
                self.sandbox_mode = not self.sandbox_mode
            elif key == glfw.KEY_P:
                self.paint_mode = not self.paint_mode
            elif key == glfw.KEY_1:
                self.brush_tool = 0  # element
            elif key == glfw.KEY_2:
                self.brush_tool = 1  # temperature
            elif key == glfw.KEY_3:
                self.brush_tool = 2  # eraser
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_F5:
                if self._recording:
                    self._stop_recording()
                else:
                    self._start_recording()

    # ── Simulation ────────────────────────────────────────────────────

    def _cache_compute_uniforms(self):
        """Cache uniform member references to avoid dict lookups every step."""
        prog = self.compute_prog
        self._cu_size = prog['u_size'] if 'u_size' in prog else None
        self._cu_dt = prog['u_dt'] if 'u_dt' in prog else None
        self._cu_boundary = prog['u_boundary'] if 'u_boundary' in prog else None
        self._cu_frame = prog['u_frame'] if 'u_frame' in prog else None
        self._cu_params = []
        for i in range(4):
            name = f'u_param{i}'
            self._cu_params.append(prog[name] if name in prog else None)
        self._cu_groups = (self.size + 7) // 8

    def _step_sim(self):
        src = self.tex_a if self.ping == 0 else self.tex_b
        dst = self.tex_b if self.ping == 0 else self.tex_a

        src.bind_to_image(0, read=True, write=False)
        dst.bind_to_image(1, read=False, write=True)

        if self.is_element_ca:
            self.element_ssbo.bind_to_storage_buffer(2)

        # Use cached uniform references (built in _cache_compute_uniforms)
        if self._cu_size is not None:
            self._cu_size.value = self.size
        if self._cu_dt is not None:
            self._cu_dt.value = self.dt
        if self._cu_boundary is not None:
            self._cu_boundary.value = self.boundary_mode
        if self._cu_frame is not None:
            self._cu_frame.value = self.step_count

        param_values = list(self.params.values())
        for i, cu in enumerate(self._cu_params):
            if cu is not None:
                cu.value = float(param_values[i]) if i < len(param_values) else 0.0

        self.compute_prog.run(self._cu_groups, self._cu_groups, self._cu_groups)
        GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                           GL.GL_TEXTURE_FETCH_BARRIER_BIT)

        self.ping = 1 - self.ping
        self.step_count += 1
        self._cpu_grid_dirty = True  # invalidate CPU cache
        self._cull_valid = False
        self._accel_textures_valid = False

    def _reset(self):
        self._cull_valid = False
        self._accel_textures_valid = False
        self.step_count = 0
        rng = np.random.RandomState(self.seed)
        init_name = getattr(self, '_current_init', self.preset['init'])
        init_func = INIT_FUNCS.get(init_name, init_random_sparse)
        data = init_func(self.size, rng)

        self.tex_a.write(np.ascontiguousarray(data.astype(self._tex_np_dtype)).tobytes())
        del data
        self.tex_b.write(bytes(self.size ** 3 * self._tex_bpt))
        self.ping = 0
        self._prev_grid = None
        self._metrics_in_flight = False
        self._metric_history = []
        self._score = 0.0
        self._score_metrics = {}
        self._cpu_grid = None
        self._cpu_grid_dirty = True

    def _sync_quantum_K(self):
        """Auto-set ħ/2m = size/50 for quantum shaders so Bohr radius a₀ = size/25 matches init."""
        shader = self.preset.get('shader', '')
        if shader.startswith('schrodinger') and 'ħ/2m' in self.params:
            self.params['ħ/2m'] = self.size / 50.0

    def _change_rule(self, rule_name):
        self.rule_name = rule_name
        self.preset = RULE_PRESETS[rule_name]
        self.params = dict(self.preset['params'])
        self.dt = self.preset['dt']
        self._current_init = self.preset['init']
        self.vis_channels = self.preset.get('vis_channels', ['Value'])
        self.vis_channel = self.preset.get('vis_default', 0)
        self.vis_abs = self.preset.get('vis_abs', False)
        self.renderer_mode = self.preset.get('render_mode', 'volumetric')
        self.boundary_mode = _BOUNDARY_NAME_TO_MODE.get(
            self.preset.get('boundary', 'toroidal'), 0)
        self.is_element_ca = self.preset.get('is_element_ca', False)
        # Apply preset's preferred grid size if specified and current size is smaller
        pref_size = self.preset.get('default_size')
        if pref_size and self.size < pref_size:
            self._change_size(pref_size)
        if rule_name == 'sandbox':
            self.sandbox_mode = True
        else:
            self.sandbox_mode = False
        # Auto-set voxel visibility threshold to match rule's alive criterion
        shader = self.preset['shader']
        if self.is_element_ca:
            self.voxel_threshold = 0.5
        elif shader == 'reaction_diffusion_3d':
            self.voxel_threshold = 0.01
        elif shader == 'wave_3d':
            self.voxel_threshold = 0.005
        elif shader in ('smoothlife_3d', 'lenia_3d'):
            self.voxel_threshold = 0.01
        elif shader == 'cahn_hilliard':
            self.voxel_threshold = 0.1
        elif shader == 'erosion_3d':
            self.voxel_threshold = 0.3
        elif shader in ('mycelium_3d', 'fracture_3d'):
            self.voxel_threshold = 0.3
        elif shader == 'fire_3d':
            self.voxel_threshold = 0.1
        elif shader in ('em_wave_3d', 'viscous_fingers_3d', 'physarum_3d', 'galaxy_3d', 'lichen_3d'):
            self.voxel_threshold = 0.05
        elif shader.startswith('schrodinger'):
            self.voxel_threshold = 0.01
        else:
            self.voxel_threshold = 0.5
        # Auto-set ħ/2m = size/50 for quantum shaders so a₀ matches init
        self._sync_quantum_K()
        self._compile_compute()
        self._cache_compute_uniforms()
        self._reset()

    def _change_size(self, new_size):
        if new_size == self.size:
            return
        self._cull_valid = False
        self.tex_a.release()
        self.tex_b.release()
        old_fmt = self._tex_glsl_fmt
        self.size = new_size
        self._tex_dtype, self._tex_np_dtype, self._tex_bpt, self._tex_glsl_fmt = \
            _tex_format_for_size(new_size)
        self._alloc_voxel_buffer(new_size)
        # Recompile shaders if texture format changed (e.g. rgba32f ↔ rgba16f)
        if self._tex_glsl_fmt != old_fmt:
            self._compile_compute()
            self._cache_compute_uniforms()
            self._compile_cull()
            self._compile_metrics()
        # Always update dispatch group count (even without recompile)
        self._cu_groups = (new_size + 7) // 8
        # Auto-sync ħ/2m for quantum shaders when grid size changes
        self._sync_quantum_K()
        self._init_volume()
        self.step_count = 0
        self._prev_grid = None
        self._metrics_in_flight = False
        self._metric_history = []
        self._score = 0.0
        self._score_metrics = {}
        self._cpu_grid = None
        self._cpu_grid_dirty = True
        # Scale score interval with grid volume to avoid GPU readback stutter
        # 32³→20, 64³→30, 128³→60, 256³→120
        self._score_interval = max(20, (new_size // 32) * 15)

    def _colormap_semantic_labels(self):
        """Return (low_label, high_label, description) based on current rule."""
        shader = self.preset.get('shader', '')
        vis_abs = self.preset.get('vis_abs', False)
        if shader == 'game_of_life_3d':
            return ("exposed", "buried", "surface neighbor count")
        elif shader == 'reaction_diffusion_3d':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'V'
            if 'U' in ch:
                return ("depleted", "saturated", "substrate concentration")
            return ("sparse", "dense", "catalyst concentration")
        elif shader == 'smoothlife_3d':
            return ("empty", "alive", "cell field density")
        elif shader == 'lenia_3d':
            return ("empty", "alive", "organism density")
        elif shader == 'wave_3d':
            if vis_abs:
                return ("calm", "peak", "wave amplitude")
            return ("trough", "peak", "wave displacement")
        elif shader == 'crystal_growth':
            return ("dilute", "saturated", "nutrient concentration")
        elif shader == 'predator_prey_3d':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'Prey'
            if 'Prey' in ch: return ("none", "dense", "prey population")
            if 'Predator' in ch: return ("none", "dense", "predator population")
            return ("barren", "lush", "grass coverage")
        elif shader == 'kuramoto_3d':
            return ("0", "2\u03c0", "oscillator phase")
        elif shader == 'bz_3d':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'u'
            if 'u' in ch.lower() or 'Activator' in ch: return ("resting", "excited", "activator (HBrO\u2082)")
            return ("recovered", "oxidized", "catalyst recovery")
        elif shader == 'morphogen_3d':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'Activator'
            if 'Activator' in ch: return ("absent", "peak", "morphogen activator")
            if 'Inhibitor' in ch: return ("low", "saturated", "morphogen inhibitor")
            return ("sparse", "dense", "tissue density")
        elif shader == 'flocking_3d':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'Density'
            if 'Density' in ch: return ("empty", "crowded", "agent density")
            return ("-max", "+max", "velocity component")
        elif shader == 'cahn_hilliard':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'Order param'
            if 'Order' in ch: return ("phase A", "phase B", "order parameter")
            return ("low μ", "high μ", "chemical potential")
        elif shader == 'erosion_3d':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'Solid'
            if 'Solid' in ch: return ("empty", "rock", "solid density")
            if 'Fluid' in ch: return ("dry", "flooded", "water amount")
            return ("clear", "laden", "sediment load")
        elif shader == 'mycelium_3d':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'Biomass'
            if 'Biomass' in ch: return ("empty", "dense", "hyphal biomass")
            if 'Nutrient' in ch: return ("depleted", "rich", "nutrient concentration")
            return ("quiet", "active", "chemical signal")
        elif shader == 'em_wave_3d':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'Ez'
            if 'Ez' in ch: return ("-E", "+E", "electric field (Ez)")
            if 'Bx' in ch: return ("-B", "+B", "magnetic field (Bx)")
            return ("-B", "+B", "magnetic field (By)")
        elif shader == 'viscous_fingers_3d':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'Saturation'
            if 'Saturation' in ch: return ("defender", "invader", "fluid saturation")
            if 'Pressure' in ch: return ("low P", "high P", "fluid pressure")
            return ("bulk", "front", "interface marker")
        elif shader == 'fire_3d':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'Temperature'
            if 'Fuel' in ch: return ("burnt", "unburnt", "fuel density")
            if 'Temperature' in ch: return ("cold", "inferno", "temperature")
            if 'Oxygen' in ch: return ("vacuum", "rich", "oxygen level")
            return ("none", "bright", "ember density")
        elif shader == 'physarum_3d':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'Trail'
            if 'Trail' in ch: return ("fresh", "marked", "pheromone trail")
            if 'Agent' in ch: return ("empty", "crowded", "agent density")
            return ("consumed", "abundant", "food source")
        elif shader == 'fracture_3d':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'Integrity'
            if 'Displacement' in ch: return ("-u", "+u", "elastic displacement")
            if 'Stress' in ch: return ("relaxed", "strained", "stress magnitude")
            if 'Integrity' in ch: return ("broken", "intact", "material integrity")
            return ("zero", "high", "strain rate")
        elif shader == 'galaxy_3d':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'Density'
            if 'Density' in ch: return ("void", "cluster", "matter density")
            return ("-v", "+v", "velocity component")
        elif shader == 'lichen_3d':
            ch = self.vis_channels[self.vis_channel] if hasattr(self, 'vis_channel') else 'Species A'
            if 'Species A' in ch: return ("absent", "dominant", "pioneer species")
            if 'Species B' in ch: return ("absent", "dominant", "competitor species")
            if 'Resource' in ch: return ("depleted", "abundant", "substrate resource")
            return ("absent", "dominant", "nomad species")
        return ("low", "high", "cell value")

    def _draw_colormap_legend(self):
        """Draw a colormap legend with gradient bar, labels, and description."""
        _c2u = imgui.color_convert_float4_to_u32
        colormap_names = ["Fire", "Cool", "Grayscale", "Neon", "Discrete", "Spectral"]
        low_label, high_label, desc = self._colormap_semantic_labels()
        white = _c2u(imgui.ImVec4(1, 1, 1, 1))
        gray = _c2u(imgui.ImVec4(0.7, 0.7, 0.7, 1))
        dim = _c2u(imgui.ImVec4(0.5, 0.5, 0.5, 1))
        draw_list = imgui.get_window_draw_list()
        cursor = imgui.get_cursor_screen_pos()
        bar_w = 220.0
        bar_h = 14.0
        x0, y0 = cursor.x, cursor.y

        # Description line
        draw_list.add_text(imgui.ImVec2(x0, y0), dim, desc)
        y0 += 16

        if self.colormap == 4:  # Discrete — draw individual swatches
            n_bands = 16
            swatch_w = bar_w / n_bands
            for i in range(n_bands):
                t = (i + 0.5) / n_bands
                c = self._colormap_eval(t)
                col = _c2u(imgui.ImVec4(c[0], c[1], c[2], 1.0))
                sx = x0 + i * swatch_w
                draw_list.add_rect_filled(
                    imgui.ImVec2(sx + 1, y0),
                    imgui.ImVec2(sx + swatch_w - 1, y0 + bar_h),
                    col,
                )
            # Border
            draw_list.add_rect(
                imgui.ImVec2(x0, y0),
                imgui.ImVec2(x0 + bar_w, y0 + bar_h),
                dim,
            )
            # Semantic labels
            draw_list.add_text(imgui.ImVec2(x0, y0 + bar_h + 2), gray, low_label)
            lbl_w = len(high_label) * 7
            draw_list.add_text(imgui.ImVec2(x0 + bar_w - lbl_w, y0 + bar_h + 2), gray, high_label)
        else:  # Continuous gradient
            n_segments = 64
            seg_w = bar_w / n_segments
            for i in range(n_segments):
                t = i / n_segments
                t2 = (i + 1) / n_segments
                c = self._colormap_eval(t)
                c2 = self._colormap_eval(t2)
                col_l = _c2u(imgui.ImVec4(c[0], c[1], c[2], 1.0))
                col_r = _c2u(imgui.ImVec4(c2[0], c2[1], c2[2], 1.0))
                draw_list.add_rect_filled_multi_color(
                    imgui.ImVec2(x0 + i * seg_w, y0),
                    imgui.ImVec2(x0 + (i + 1) * seg_w, y0 + bar_h),
                    col_l, col_r, col_r, col_l,
                )
            # Border
            draw_list.add_rect(
                imgui.ImVec2(x0, y0),
                imgui.ImVec2(x0 + bar_w, y0 + bar_h),
                dim,
            )
            # Tick marks at 0.25, 0.5, 0.75
            for frac in (0.25, 0.5, 0.75):
                tx = x0 + bar_w * frac
                draw_list.add_line(
                    imgui.ImVec2(tx, y0 + bar_h - 3),
                    imgui.ImVec2(tx, y0 + bar_h + 2),
                    dim,
                )
            # Semantic labels
            draw_list.add_text(imgui.ImVec2(x0, y0 + bar_h + 2), gray, low_label)
            lbl_w = len(high_label) * 7
            draw_list.add_text(imgui.ImVec2(x0 + bar_w - lbl_w, y0 + bar_h + 2), gray, high_label)

        # Advance cursor past the legend
        imgui.dummy(imgui.ImVec2(bar_w, bar_h + 34))

    def _colormap_eval(self, t):
        """Evaluate current colormap at t in [0,1], returns (r,g,b)."""
        t = max(0.0, min(1.0, t))
        if self.colormap == 0:  # Fire
            return (min(t * 3.0, 1.0), max(min(t * 3.0 - 1.0, 1.0), 0.0), max(min(t * 3.0 - 2.0, 1.0), 0.0))
        elif self.colormap == 1:  # Cool
            return (math.sin(t * 1.5708) * 0.3, t * 0.8, 0.5 + t * 0.5)
        elif self.colormap == 3:  # Neon
            h = t * 4.0
            r = max(min(abs(h - 2.0) - 1.0, 1.0), 0.0)
            g = max(min(2.0 - abs(h - 1.5), 1.0), 0.0)
            b = max(min(2.0 - abs(h - 3.0), 1.0), 0.0)
            s = 0.5 + t * 0.5
            return (r * s, g * s, b * s)
        elif self.colormap == 4:  # Discrete
            idx = min(int(t * 16), 15)
            hue = (idx * 0.618033988) % 1.0
            s, v = 0.75, 0.95
            c = v * s
            h = hue * 6.0
            x = c * (1.0 - abs(h % 2.0 - 1.0))
            if h < 1: rgb = (c, x, 0)
            elif h < 2: rgb = (x, c, 0)
            elif h < 3: rgb = (0, c, x)
            elif h < 4: rgb = (0, x, c)
            elif h < 5: rgb = (x, 0, c)
            else: rgb = (c, 0, x)
            m = v - c
            return (rgb[0] + m, rgb[1] + m, rgb[2] + m)
        else:  # Grayscale
            return (t, t, t)

    def _randomize_params(self):
        """Randomize all parameters within their defined ranges, including dt if applicable."""
        ranges = self.preset['param_ranges']
        for name, (lo, hi) in ranges.items():
            if name.startswith("unused"):
                continue
            # Normalize swapped or degenerate ranges so randint/uniform never throw.
            if hi < lo:
                lo, hi = hi, lo
            if isinstance(lo, int) and isinstance(hi, int):
                self.params[name] = lo if hi == lo else random.randint(lo, hi)
            else:
                self.params[name] = float(lo) if hi == lo else random.uniform(lo, hi)
        # Also randomize dt if the preset defines a safe range
        dt_range = self.preset.get('dt_range')
        if dt_range:
            dlo, dhi = float(min(dt_range)), float(max(dt_range))
            self.dt = dlo if dhi == dlo else random.uniform(dlo, dhi)
        self.seed = random.randint(0, 99999)
        self._reset()

    def _mutate_params(self, strength=0.2):
        """Small random perturbation of current params, including dt."""
        ranges = self.preset['param_ranges']
        for name, (lo, hi) in ranges.items():
            if name.startswith("unused"):
                continue
            if hi < lo:
                lo, hi = hi, lo
            val = self.params.get(name, lo)
            span = hi - lo if hi != lo else 1.0
            delta = random.gauss(0, strength * span)
            if isinstance(lo, int) and isinstance(hi, int):
                self.params[name] = max(lo, min(hi, int(round(val + delta))))
            else:
                self.params[name] = max(float(lo), min(float(hi), val + delta))
        dt_range = self.preset.get('dt_range')
        if dt_range:
            dlo, dhi = float(min(dt_range)), float(max(dt_range))
            dt_span = dhi - dlo
            self.dt = max(dlo, min(dhi, self.dt + random.gauss(0, strength * dt_span)))
        self._reset()

    def _load_discoveries(self):
        """Load discoveries from JSON file."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.discovery_file)
        if os.path.exists(path):
            with open(path) as f:
                self.discoveries = json.load(f)
        else:
            self.discoveries = []

    def _save_current_to_discoveries(self):
        """Save current params as a new discovery."""
        entry = {
            'rule': self.rule_name,
            'params': dict(self.params),
            'dt': self.dt,
            'score': self._score,
            'seed': self.seed,
            'final_alive': self._score_metrics.get('alive_ratio', 0),
            'final_activity': self._score_metrics.get('activity', 0),
            'final_surface': self._score_metrics.get('surface_ratio', 0),
        }
        self.discoveries.append(entry)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.discovery_file)
        with open(path, 'w') as f:
            json.dump(self.discoveries, f, indent=2)
        self.discovery_index = len(self.discoveries) - 1

    def _load_discovery(self, index):
        """Load a discovery by index into the simulator.

        Surfaces a warning when the saved entry references parameters that
        the current rule no longer defines (e.g. a rule was renamed or its
        param schema changed since the discovery was saved). Without this
        the discovery would silently load with default values for the
        missing params and behave nothing like the recorded run.
        """
        if not self.discoveries or index < 0 or index >= len(self.discoveries):
            return
        disc = self.discoveries[index]
        # Preserve user's renderer choice across discovery switches
        user_renderer = self.renderer_mode
        # Always apply full rule settings (render mode, vis channel, threshold)
        self._change_rule(disc['rule'])
        # Restore user's renderer preference
        self.renderer_mode = user_renderer
        unknown = []
        for k, v in disc['params'].items():
            if k in self.params:
                self.params[k] = v
            else:
                unknown.append(k)
        if unknown:
            sys.stderr.write(
                f"[discovery] WARNING: rule '{disc.get('rule', '?')}' no longer "
                f"defines params: {', '.join(unknown)} — these were dropped.\n")
        if 'dt' in disc:
            self.dt = disc['dt']
        if 'seed' in disc:
            self.seed = disc['seed']
        self.discovery_index = index
        self._reset()

    def _update_score(self):
        """Compute live interestingness score entirely on the GPU.

        Uses a pipelined approach: read the *previous* interval's 16-byte
        SSBO result (shader finished many frames ago → instant), then dispatch
        a new reduction shader + GPU texture copy with no CPU sync.
        """
        from test_harness import score_interestingness

        src = self.tex_a if self.ping == 0 else self.tex_b
        total = self.size ** 3

        # ── 1. Harvest previous results (if any) ─────────────────────
        if getattr(self, '_metrics_in_flight', False):
            # Barrier ensures compute writes are visible to CPU read
            GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT)
            raw = np.frombuffer(self._metrics_ssbo.read(), dtype=np.uint32)
            alive_count = int(raw[0])
            change_count = int(raw[1])
            surface_count = int(raw[2])
            nan_count = int(raw[3])

            alive_ratio = alive_count / total
            activity = change_count / total
            surface_ratio = surface_count / max(alive_count, 1)
            has_bad = nan_count > 0

            m = {
                'alive_count': alive_count,
                'alive_ratio': alive_ratio,
                'activity': activity,
                'surface_ratio': surface_ratio,
                'has_nan': has_bad,
                'has_inf': has_bad,
            }
            self._score_metrics = m
            self._metric_history.append(m)
            if len(self._metric_history) > 40:
                self._metric_history.pop(0)
            self._score = score_interestingness(self._metric_history)

        # ── 2. Determine rule-specific metric config ──────────────────
        preset = self.preset
        shader = preset['shader']
        is_elem = preset.get('is_element_ca', False)
        if is_elem:
            channel, mode, threshold, change_thr = 0, 3, 0.5, 0.5
        elif shader == 'reaction_diffusion_3d':
            channel, mode, threshold, change_thr = 1, 1, 0.01, 0.001
        elif shader == 'wave_3d':
            channel, mode, threshold, change_thr = 0, 2, 0.005, 0.001
        elif shader in ('smoothlife_3d', 'lenia_3d'):
            channel, mode, threshold, change_thr = 0, 1, 0.01, 0.001
        else:
            channel, mode, threshold, change_thr = 0, 0, 0.5, 0.01

        # ── 3. Zero SSBO, dispatch reduction shader (no CPU sync) ─────
        self._metrics_ssbo.write(b'\x00' * 16)

        # Bind current grid and the *previous step* texture (the other
        # ping-pong buffer) for activity detection.  This catches period-1
        # and period-2 oscillations that the long-interval snapshot misses.
        prev = self.tex_b if self.ping == 0 else self.tex_a
        src.bind_to_image(0, read=True, write=False)
        prev.bind_to_image(1, read=True, write=False)
        self._metrics_ssbo.bind_to_storage_buffer(5)

        prog = self._metrics_prog
        prog['u_size'].value = self.size
        prog['u_threshold'].value = threshold
        prog['u_channel'].value = channel
        prog['u_mode'].value = mode
        prog['u_change_thr'].value = change_thr
        prog['u_has_prev'].value = 1 if self.step_count > 0 else 0
        prog['u_boundary'].value = self.boundary_mode

        gx = (self.size + 7) // 8
        prog.run(gx, gx, gx)

        # Barrier ensures compute SSBO writes complete before next CPU read
        GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT)
        self._metrics_in_flight = True

    # ── Rendering ─────────────────────────────────────────────────────

    def _render_volume(self):
        # Build acceleration structures if stale
        if not self._accel_textures_valid:
            self._build_accel_textures()

        src_tex = self.tex_a if self.ping == 0 else self.tex_b

        cam_pos = self._get_camera_pos()
        cam_rot = self._get_camera_rot()
        fov = 1.0
        aspect = self.width / max(self.height, 1)

        # Determine if rendering to half-res FBO
        use_half = self._use_half_res and self.size >= 96
        if use_half:
            hw, hh = max(self.width // 2, 1), max(self.height // 2, 1)
            if (self._half_res_fbo is None or
                    getattr(self, '_half_res_size', None) != (hw, hh)):
                self._alloc_half_res_fbo(self.width, self.height)
            self._half_res_fbo.use()
            self.ctx.viewport = (0, 0, hw, hh)
        else:
            self.ctx.screen.use()

        self.ctx.clear(0.02, 0.02, 0.04)

        # Bind textures
        src_tex.use(location=0)
        if hasattr(self, '_occ_tex') and self._occ_tex is not None:
            self._occ_tex.use(location=1)
        if hasattr(self, '_mm_tex') and self._mm_tex is not None:
            self._mm_tex.use(location=2)
        if hasattr(self, '_view_tex') and self._view_tex is not None:
            self._view_tex.use(location=3)

        # Set uniforms (None-safe — unused uniforms may be stripped by GLSL)
        if self._rp_u_volume is not None:
            self._rp_u_volume.value = 0
        if self._rp_u_size is not None:
            self._rp_u_size.value = self.size
        self._rp_u_camera_pos.value = tuple(cam_pos)
        self._rp_u_camera_rot.value = tuple(cam_rot.T.flatten())
        self._rp_u_fov.value = fov
        self._rp_u_aspect.value = aspect
        self._rp_u_density_scale.value = self.density_scale
        self._rp_u_brightness.value = self.brightness
        self._rp_u_render_mode.value = self.render_mode
        self._rp_u_iso_threshold.value = self.iso_threshold
        self._rp_u_slice_pos.value = self.slice_pos
        self._rp_u_slice_axis.value = self.slice_axis
        self._rp_u_colormap.value = self.colormap
        if self._rp_u_vis_channel is not None:
            self._rp_u_vis_channel.value = self.vis_channel
        if self._rp_u_vis_abs is not None:
            self._rp_u_vis_abs.value = 1 if self.vis_abs else 0

        # Acceleration uniforms (None-safe — GLSL compiler may strip unused)
        if self._rp_u_frame_id is not None:
            self._rp_u_frame_id.value = self._frame_counter
        occ_dim = getattr(self, '_occ_dim', 1)
        mm_dim = getattr(self, '_mm_dim', 1)
        if self._rp_u_occ_size is not None:
            self._rp_u_occ_size.value = occ_dim
        if self._rp_u_use_occupancy is not None:
            self._rp_u_use_occupancy.value = 1
        if self._rp_u_minmax_size is not None:
            self._rp_u_minmax_size.value = mm_dim
        if self._rp_u_use_minmax is not None:
            self._rp_u_use_minmax.value = 1

        self.vao.render(moderngl.TRIANGLE_STRIP)
        self._frame_counter += 1

        # Upsample half-res to screen
        if use_half:
            self.ctx.screen.use()
            self.ctx.viewport = (0, 0, self.width, self.height)
            self._half_res_tex.use(location=0)
            self._up_u_half_res.value = 0
            self._up_u_texel_size.value = (1.0 / hw, 1.0 / hh)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self._upsample_vao.render(moderngl.TRIANGLE_STRIP)

    def _get_view_proj(self):
        """Build a view-projection matrix for voxel rendering."""
        cam_pos = self._get_camera_pos()
        cam_rot = self._get_camera_rot()

        # View matrix (looking at cam_target from cam_pos)
        forward = self.cam_target - cam_pos
        forward = forward / np.linalg.norm(forward)
        right = cam_rot[:, 0]
        up = cam_rot[:, 1]

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[0, 3] = -np.dot(right, cam_pos)
        view[1, 3] = -np.dot(up, cam_pos)
        view[2, 3] = np.dot(forward, cam_pos)

        # Perspective projection
        aspect = self.width / max(self.height, 1)
        fov_y = 1.0
        near, far = 0.01, 100.0
        f = 1.0 / math.tan(fov_y * 0.5)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0

        # Transpose: numpy is row-major, GLSL mat4 expects column-major
        vp_row_major = proj @ view
        return vp_row_major.T.flatten(), vp_row_major

    @staticmethod
    def _extract_frustum_planes(vp):
        """Extract 6 frustum planes from a row-major 4x4 VP matrix.
        Each plane is (a, b, c, d) where ax+by+cz+d >= 0 is inside.
        Uses the Gribb-Hartmann method."""
        planes = np.zeros((6, 4), dtype=np.float32)
        # Left:   row3 + row0
        planes[0] = vp[3] + vp[0]
        # Right:  row3 - row0
        planes[1] = vp[3] - vp[0]
        # Bottom: row3 + row1
        planes[2] = vp[3] + vp[1]
        # Top:    row3 - row1
        planes[3] = vp[3] - vp[1]
        # Near:   row3 + row2
        planes[4] = vp[3] + vp[2]
        # Far:    row3 - row2
        planes[5] = vp[3] - vp[2]
        # Normalize
        for i in range(6):
            n = np.linalg.norm(planes[i, :3])
            if n > 1e-8:
                planes[i] /= n
        return planes

    @staticmethod
    def _aabb_in_frustum(planes, bmin, bmax):
        """Test if AABB (bmin, bmax) is at least partially inside the frustum.
        Returns False only if the box is fully outside any plane."""
        for i in range(6):
            nx, ny, nz, d = planes[i]
            # P-vertex: the corner most in the direction of the plane normal
            px = bmax[0] if nx >= 0 else bmin[0]
            py = bmax[1] if ny >= 0 else bmin[1]
            pz = bmax[2] if nz >= 0 else bmin[2]
            if nx * px + ny * py + nz * pz + d < 0:
                return False
        return True

    def _render_voxels(self):
        """Render visible cells as instanced cubes with Phong lighting.
        Uses multi-pass spatial chunking for large grids: the grid is split
        into chunks, each cull+draw pass reuses the same SSBO, so no voxels
        are ever dropped regardless of grid size or fill density."""
        self.ctx.clear(0.02, 0.02, 0.04)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        if self.voxel_alpha < 0.99:
            self.ctx.enable(moderngl.BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        src_tex = self.tex_a if self.ping == 0 else self.tex_b

        # Determine rule-specific cull parameters (channel, mode, threshold)
        shader = self.preset['shader']
        is_elem = self.preset.get('is_element_ca', False)
        if is_elem:
            cull_channel, cull_use_abs = 0, 0
        elif shader in ('wave_3d', 'em_wave_3d'):
            cull_channel, cull_use_abs = self.vis_channel, 1
        else:
            cull_channel, cull_use_abs = self.vis_channel, 0

        # Set up render uniforms (constant across all chunks)
        src_tex.use(location=0)
        self.element_ssbo.bind_to_storage_buffer(2)
        self.voxel_buffer.bind_to_storage_buffer(3)

        cam_pos = self._get_camera_pos()
        view_proj_flat, vp_row_major = self._get_view_proj()

        self._vp_u_size.value = self.size
        self._vp_u_view_proj.value = tuple(view_proj_flat)
        self._vp_u_voxel_gap.value = self.voxel_gap
        self._vp_u_camera_pos.value = tuple(cam_pos)
        self._vp_u_brightness.value = self.brightness
        self._vp_u_colormap.value = self.colormap
        self._vp_u_is_element_ca.value = 1 if is_elem else 0
        self._vp_u_alpha.value = self.voxel_alpha
        self._vp_u_channel.value = cull_channel
        self._vp_u_use_abs.value = cull_use_abs
        self._vp_u_threshold.value = self.voxel_threshold

        cpd = self._voxel_chunks_per_dim

        # ── Fast path: reuse cached cull result (single-pass only) ────
        # When the grid data and cull parameters haven't changed, the
        # voxel SSBO and indirect buffer are still valid from the
        # previous frame.  Just re-draw with updated camera/uniforms.
        # This eliminates the non-deterministic atomicAdd ordering that
        # causes flicker on NVK/Zink when the cull is re-dispatched.
        if self._cull_valid and cpd == 1:
            self.voxel_vao.render_indirect(self.voxel_indirect_buffer,
                                           moderngl.TRIANGLES, count=1)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.BLEND)
            return

        # ── Full cull path ────────────────────────────────────────────
        # Barrier: ensure previous frame's draw-call SSBO reads complete
        # before this frame's cull compute overwrites the buffers.
        GL.glMemoryBarrier(GL.GL_ALL_BARRIER_BITS)

        # Set up cull uniforms (constant across all chunks)
        self.voxel_indirect_buffer.bind_to_storage_buffer(4)
        self._cull_u_size.value = self.size
        self._cull_u_threshold.value = self.voxel_threshold
        self._cull_u_is_element_ca.value = 1 if is_elem else 0
        self._cull_u_max_voxels.value = self.max_voxels
        self._cull_u_channel.value = cull_channel
        self._cull_u_use_abs.value = cull_use_abs

        # Extract frustum planes for chunk culling (only useful when cpd > 1)
        frustum_planes = self._extract_frustum_planes(vp_row_major) if cpd > 1 else None

        chunk_edge = (self.size + cpd - 1) // cpd

        # Total counter is now reset GPU-side in the first chunk's indirect reset
        self._voxel_total_counter.bind_to_storage_buffer(6)
        is_first_chunk = True

        for cz in range(cpd):
            for cy in range(cpd):
                for cx in range(cpd):
                    # Compute chunk bounds
                    cmin = (cx * chunk_edge, cy * chunk_edge, cz * chunk_edge)
                    cmax = (min((cx + 1) * chunk_edge, self.size),
                            min((cy + 1) * chunk_edge, self.size),
                            min((cz + 1) * chunk_edge, self.size))

                    # Frustum cull: skip chunks entirely outside the view
                    if frustum_planes is not None:
                        inv_size = 1.0 / self.size
                        bmin = (cmin[0] * inv_size, cmin[1] * inv_size, cmin[2] * inv_size)
                        bmax = (cmax[0] * inv_size, cmax[1] * inv_size, cmax[2] * inv_size)
                        if not self._aabb_in_frustum(frustum_planes, bmin, bmax):
                            continue

                    # Reset indirect buffer for this chunk (GPU-side, no CPU->GPU stall)
                    # First chunk also resets the frame-level total counter
                    self.voxel_indirect_buffer.bind_to_storage_buffer(4)
                    self._indirect_reset_u_reset_total.value = 1 if is_first_chunk else 0
                    self._indirect_reset_prog.run(1, 1, 1)
                    GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
                    is_first_chunk = False

                    # Bind SSBOs for cull
                    src_tex.use(location=0)
                    self.voxel_buffer.bind_to_storage_buffer(3)
                    self.voxel_indirect_buffer.bind_to_storage_buffer(4)
                    self._voxel_total_counter.bind_to_storage_buffer(6)

                    # Set chunk bounds
                    self._cull_u_chunk_min.value = cmin
                    self._cull_u_chunk_max.value = cmax

                    # Dispatch cull for this chunk only
                    gx = (cmax[0] - cmin[0] + 7) // 8
                    gy = (cmax[1] - cmin[1] + 7) // 8
                    gz = (cmax[2] - cmin[2] + 7) // 8
                    self.voxel_cull_prog.run(gx, gy, gz)
                    GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_COMMAND_BARRIER_BIT)

                    # Draw this chunk's voxels
                    self.voxel_buffer.bind_to_storage_buffer(3)
                    self.element_ssbo.bind_to_storage_buffer(2)
                    src_tex.use(location=0)
                    self.voxel_vao.render_indirect(self.voxel_indirect_buffer, moderngl.TRIANGLES, count=1)

        # Mark cull as valid for single-pass reuse on subsequent frames
        if cpd == 1:
            self._cull_valid = True

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        # Read back total voxel count via async copy of the accumulator.
        self._voxel_count_frame = getattr(self, '_voxel_count_frame', 0) + 1
        if self._voxel_count_frame >= 15:
            self._voxel_count_frame = 0
            if not hasattr(self, '_total_staging'):
                self._total_staging = self.ctx.buffer(reserve=4)
                self._total_staging_ready = False
            if self._total_staging_ready:
                raw_count = int(np.frombuffer(self._total_staging.read(), dtype=np.uint32)[0])
                self._last_voxel_count = raw_count
                self._voxels_clipped = False
            # Issue copy for next readback (returns immediately)
            self.ctx.copy_buffer(self._total_staging, self._voxel_total_counter)
            self._total_staging_ready = True

    def _render(self):
        """Dispatch to the appropriate renderer."""
        if self.renderer_mode == 'voxel':
            self._render_voxels()
        elif self._use_compute_ray:
            self._render_volume_compute()
        else:
            self._render_volume()

    def _scene_hash(self):
        """Tuple of every state bit that affects what a rendered frame shows.
        If this is unchanged between frames we can skip the render and blit
        the cached image instead. A miss just means one-frame staleness —
        never corruption — so we keep the hash conservative but cheap.
        """
        return (
            self.step_count,
            self.size, self.rule_name, self.renderer_mode,
            self.width, self.height,
            # Camera (rounded to suppress sub-pixel jitter from trackpads)
            round(self.cam_theta, 4), round(self.cam_phi, 4),
            round(self.cam_dist, 4),
            round(float(self.cam_target[0]), 4),
            round(float(self.cam_target[1]), 4),
            round(float(self.cam_target[2]), 4),
            # Render knobs
            round(self.density_scale, 4), round(self.brightness, 4),
            self.render_mode, round(self.iso_threshold, 4), self.colormap,
            round(self.slice_pos, 4), self.slice_axis,
            self.vis_channel, bool(self.vis_abs),
            # Recording disables caching (we need a fresh render each frame)
            bool(getattr(self, '_recording', False)),
        )

    def _ensure_scene_cache(self):
        """Allocate (or resize) the screen-sized scene cache FBO."""
        need = (self.width, self.height)
        if self._scene_cache_size == need and self._scene_cache_fbo is not None:
            return
        # Free previous
        if self._scene_cache_fbo is not None:
            try: self._scene_cache_fbo.release()
            except Exception: pass
        if self._scene_cache_tex is not None:
            try: self._scene_cache_tex.release()
            except Exception: pass
        self._scene_cache_tex = self.ctx.texture(need, 4, dtype='f1')
        self._scene_cache_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._scene_cache_fbo = self.ctx.framebuffer(
            color_attachments=[self._scene_cache_tex])
        self._scene_cache_size = need

    def _snapshot_scene(self):
        """Copy current backbuffer contents into the scene cache."""
        self._ensure_scene_cache()
        # moderngl's copy_framebuffer wraps glBlitFramebuffer — fast GPU copy.
        self.ctx.copy_framebuffer(self._scene_cache_fbo, self.ctx.screen)

    def _restore_scene_snapshot(self):
        """Blit the cached scene onto the backbuffer (before UI overlay)."""
        if self._scene_cache_fbo is None:
            return False
        self.ctx.copy_framebuffer(self.ctx.screen, self._scene_cache_fbo)
        return True

    def _render_volume_compute(self):
        """Render volume using compute shader raymarcher."""
        if not self._accel_textures_valid:
            self._build_accel_textures()

        w, h = self.width, self.height
        if (not hasattr(self, '_cr_output_tex') or self._cr_output_tex is None or
                getattr(self, '_cr_output_size', None) != (w, h)):
            self._alloc_compute_ray_output(w, h)

        src_tex = self.tex_a if self.ping == 0 else self.tex_b
        cam_pos = self._get_camera_pos()
        cam_rot = self._get_camera_rot()

        # Bind output image
        self._cr_output_tex.bind_to_image(0, read=False, write=True)
        # Bind volume + accel as textures
        src_tex.use(location=0)
        if hasattr(self, '_occ_tex') and self._occ_tex is not None:
            self._occ_tex.use(location=1)
        if hasattr(self, '_mm_tex') and self._mm_tex is not None:
            self._mm_tex.use(location=2)
        if hasattr(self, '_view_tex') and self._view_tex is not None:
            self._view_tex.use(location=3)

        # Set uniforms (None-safe — GLSL compiler may strip unused)
        def _su(u, v):
            if u is not None:
                u.value = v
        _su(self._cr_u_size, self.size)
        _su(self._cr_u_camera_pos, tuple(cam_pos))
        _su(self._cr_u_camera_rot, tuple(cam_rot.T.flatten()))
        _su(self._cr_u_fov, 1.0)
        _su(self._cr_u_aspect, w / max(h, 1))
        _su(self._cr_u_density_scale, self.density_scale)
        _su(self._cr_u_brightness, self.brightness)
        _su(self._cr_u_render_mode, self.render_mode)
        _su(self._cr_u_iso_threshold, self.iso_threshold)
        _su(self._cr_u_colormap, self.colormap)
        _su(self._cr_u_vis_channel, self.vis_channel)
        _su(self._cr_u_vis_abs, 1 if self.vis_abs else 0)
        _su(self._cr_u_frame_id, self._frame_counter)
        _su(self._cr_u_resolution, (w, h))
        occ_dim = getattr(self, '_occ_dim', 1)
        mm_dim = getattr(self, '_mm_dim', 1)
        _su(self._cr_u_occ_size, occ_dim)
        _su(self._cr_u_use_occupancy, 1)
        _su(self._cr_u_minmax_size, mm_dim)
        _su(self._cr_u_use_minmax, 1)

        # Dispatch: 8x8 workgroups
        gx = (w + 7) // 8
        gy = (h + 7) // 8
        self._compute_ray_prog.run(gx, gy, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                           GL.GL_TEXTURE_FETCH_BARRIER_BIT)
        self._frame_counter += 1

        # Blit result to screen
        self.ctx.screen.use()
        self.ctx.clear(0.02, 0.02, 0.04)
        self._cr_output_tex.use(location=0)
        self._up_u_half_res.value = 0
        self._up_u_texel_size.value = (1.0 / w, 1.0 / h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._upsample_vao.render(moderngl.TRIANGLE_STRIP)

    # ── ImGui UI ──────────────────────────────────────────────────────

    def _draw_ui(self):
        imgui.set_next_window_pos(imgui.ImVec2(10, 10), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(320, 0), imgui.Cond_.first_use_ever)

        imgui.begin("Controls", flags=imgui.WindowFlags_.always_auto_resize)

        # Status
        status = "PAUSED" if self.paused else "RUNNING"
        imgui.text(f"Step: {self.step_count}  [{status}]")
        imgui.separator()

        # Rule selection
        imgui.text("Rule:")
        if imgui.begin_combo("##rule", RULE_PRESETS[self.rule_name]['label']):
            for name in RULE_PRESETS:
                preset = RULE_PRESETS[name]
                is_selected = (name == self.rule_name)
                if imgui.selectable(preset['label'], is_selected)[0]:
                    self._change_rule(name)
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        imgui.text_wrapped(self.preset['description'])
        imgui.separator()

        # Grid size
        sizes = [32, 48, 64, 96, 128, 192, 256, 384, 512]
        imgui.text("Grid size:")
        if imgui.begin_combo("##size", str(self.size)):
            for s in sizes:
                is_sel = (s == self.size)
                if imgui.selectable(f"{s}x{s}x{s}", is_sel)[0]:
                    self._change_size(s)
                if is_sel:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        imgui.separator()

        # Parameters
        imgui.text("Parameters:")
        param_names = list(self.params.keys())
        param_ranges = self.preset['param_ranges']
        for name in param_names:
            if name.startswith("unused"):
                continue
            lo, hi = param_ranges[name]
            # Defensive: presets are user-editable; a swapped range would crash
            # the slider. Normalize silently here.
            if hi < lo:
                lo, hi = hi, lo
            val = self.params[name]
            if isinstance(lo, int) and isinstance(hi, int) and lo != hi:
                changed, new_val = imgui.slider_int(name, int(val), lo, hi)
                if changed:
                    self.params[name] = new_val
            else:
                changed, new_val = imgui.slider_float(name, float(val), float(lo), float(hi))
                if changed:
                    self.params[name] = new_val

        # Time step — respect the preset's safe range when one is declared,
        # otherwise fall back to the historical wide [0.001, 2.0] window.
        dt_range = self.preset.get('dt_range', (0.001, 2.0))
        dt_lo = float(min(dt_range))
        dt_hi = float(max(dt_range))
        # Allow a 2× headroom above the preset's recommended max for exploration,
        # but never below the lower bound.
        dt_hi = max(dt_hi * 2.0, dt_lo * 1.001)
        changed, new_dt = imgui.slider_float(
            "Time step", float(self.dt), dt_lo, dt_hi, "%.4f",
            flags=imgui.SliderFlags_.logarithmic)
        if changed:
            self.dt = new_dt

        changed, new_speed = imgui.slider_int("Steps/batch", self.sim_speed, 1, 20)
        if changed:
            self.sim_speed = new_speed

        sps_labels = ["Max", "1", "2", "5", "10", "20", "30", "60"]
        sps_values = [0, 1, 2, 5, 10, 20, 30, 60]
        cur_sps_label = str(self.target_sps) if self.target_sps > 0 else "Max"
        if cur_sps_label not in sps_labels:
            cur_sps_label = "Max"
        if imgui.begin_combo("Steps/sec", cur_sps_label):
            for label, val in zip(sps_labels, sps_values):
                is_sel = (val == self.target_sps)
                if imgui.selectable(label, is_sel)[0]:
                    self.target_sps = val
                    self._last_step_time = 0.0
                if is_sel:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        imgui.separator()

        # Playback controls
        if imgui.button("Play/Pause [Space]"):
            self.paused = not self.paused
        imgui.same_line()
        if imgui.button("Step [->]"):
            self._step_sim()
        imgui.same_line()
        if imgui.button("Reset [R]"):
            self._reset()
        imgui.same_line()
        if imgui.button("Randomize"):
            self._randomize_params()
        imgui.same_line()
        if imgui.button("Mutate"):
            self._mutate_params()

        changed, new_seed = imgui.input_int("Seed", self.seed)
        if changed:
            # NumPy RandomState requires uint32; clamp to a safe positive range
            # so user-entered negatives or huge values don't crash _reset().
            self.seed = int(max(0, min(0xFFFFFFFF, new_seed)))

        # Init pattern selector (if variants exist)
        variants = self.preset.get('init_variants', [])
        if variants:
            current = getattr(self, '_current_init', self.preset['init'])
            all_inits = [self.preset['init']] + [v for v in variants if v != self.preset['init']]
            if current not in all_inits:
                all_inits.append(current)
            imgui.text("Init pattern:")
            if imgui.begin_combo("##init", current):
                for name in all_inits:
                    is_sel = (name == current)
                    if imgui.selectable(name, is_sel)[0]:
                        self._current_init = name
                        self._reset()
                    if is_sel:
                        imgui.set_item_default_focus()
                imgui.end_combo()

        # ── Video recording ───────────────────────────────────────────
        if self._recording:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.8, 0.1, 0.1, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(1.0, 0.2, 0.2, 1.0))
            elapsed = time.time() - self._rec_start_time
            if imgui.button(f"  Stop Recording ({elapsed:.0f}s)  "):
                self._stop_recording()
            imgui.pop_style_color(2)
        else:
            if imgui.button("Record Video [F5]"):
                self._start_recording()
        if self._rec_msg and time.time() - self._rec_msg_time < 5.0:
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(0.5, 1.0, 0.5, 1.0), self._rec_msg)

        imgui.separator()

        # ── Live score & discovery ────────────────────────────────────
        # Score display
        score_color = imgui.ImVec4(0.2, 1.0, 0.2, 1.0) if self._score >= 0.7 else \
                      imgui.ImVec4(1.0, 1.0, 0.2, 1.0) if self._score >= 0.4 else \
                      imgui.ImVec4(1.0, 0.4, 0.2, 1.0)
        imgui.text_colored(score_color, f"Score: {self._score:.2f}")
        if self._score_metrics:
            m = self._score_metrics
            imgui.same_line()
            imgui.text(f"alive={m.get('alive_ratio', 0):.2f} act={m.get('activity', 0):.3f}")

        # Save / Browse
        if imgui.button("Save Discovery"):
            self._save_current_to_discoveries()
        imgui.same_line()
        if imgui.button("Reload##disc_reload"):
            self._load_discoveries()
        imgui.same_line()
        imgui.text(f"({len(self.discoveries)} total)")

        # Discovery browser — all rules
        if self.discoveries:
            # Group by rule (cached; invalidated on save/load)
            if getattr(self, '_disc_by_rule', None) is None or \
               getattr(self, '_disc_cache_len', -1) != len(self.discoveries):
                from collections import OrderedDict
                by_rule = OrderedDict()
                for i, d in enumerate(self.discoveries):
                    by_rule.setdefault(d.get('rule', 'unknown'), []).append(i)
                self._disc_by_rule = by_rule
                self._disc_cache_len = len(self.discoveries)
            by_rule = self._disc_by_rule

            # Show "All rules" or filter toggle
            if not hasattr(self, '_disc_show_all'):
                self._disc_show_all = True
            if not hasattr(self, '_disc_sort_mode'):
                self._disc_sort_mode = 1  # 0=score, 1=newest
            _, self._disc_show_all = imgui.checkbox("All rules##disc_all", self._disc_show_all)
            imgui.same_line()
            sort_labels = ["By score", "Newest first"]
            if imgui.begin_combo("##disc_sort", sort_labels[self._disc_sort_mode]):
                for i, label in enumerate(sort_labels):
                    is_sel = (i == self._disc_sort_mode)
                    if imgui.selectable(label, is_sel)[0]:
                        self._disc_sort_mode = i
                    if is_sel:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            imgui.same_line()

            # Build sorted index list matching the displayed order
            def _sorted_disc_indices():
                if self._disc_show_all:
                    indices = []
                    for rn in by_rule:
                        ri = by_rule[rn]
                        if self._disc_sort_mode == 0:
                            indices.extend(sorted(ri, key=lambda i: self.discoveries[i].get('score', 0), reverse=True))
                        else:
                            indices.extend(sorted(ri, reverse=True))
                    return indices
                else:
                    ri = [i for i, d in enumerate(self.discoveries) if d.get('rule') == self.rule_name]
                    if self._disc_sort_mode == 0:
                        return sorted(ri, key=lambda i: self.discoveries[i].get('score', 0), reverse=True)
                    else:
                        return sorted(ri, reverse=True)

            if imgui.button("<##disc_prev"):
                all_idx = _sorted_disc_indices()
                if all_idx:
                    if self.discovery_index in all_idx:
                        pos = all_idx.index(self.discovery_index)
                        new_pos = (pos - 1) % len(all_idx)
                    else:
                        new_pos = len(all_idx) - 1
                    self._load_discovery(all_idx[new_pos])
            imgui.same_line()
            if imgui.button(">##disc_next"):
                all_idx = _sorted_disc_indices()
                if all_idx:
                    if self.discovery_index in all_idx:
                        pos = all_idx.index(self.discovery_index)
                        new_pos = (pos + 1) % len(all_idx)
                    else:
                        new_pos = 0
                    self._load_discovery(all_idx[new_pos])
            imgui.same_line()
            if 0 <= self.discovery_index < len(self.discoveries):
                d = self.discoveries[self.discovery_index]
                imgui.text(f"#{self.discovery_index} {d['rule']} S={d.get('score',0):.2f}")
            else:
                imgui.text("(unsaved)")

            # Scrollable discovery list with rule headers
            if imgui.begin_child("##disc_list", imgui.ImVec2(0, 200), imgui.ChildFlags_.borders):
                show_rules = by_rule.keys() if self._disc_show_all else \
                    [r for r in by_rule if r == self.rule_name]
                for rule_name in show_rules:
                    indices = by_rule[rule_name]
                    # Sort by selected mode
                    if self._disc_sort_mode == 0:
                        indices_sorted = sorted(indices, key=lambda i: self.discoveries[i].get('score', 0), reverse=True)
                    else:
                        indices_sorted = sorted(indices, reverse=True)  # newest first (highest index)
                    header_open = imgui.tree_node(f"{rule_name} ({len(indices)})##rh_{rule_name}")
                    if header_open:
                        for idx in indices_sorted:
                            d = self.discoveries[idx]
                            sc = d.get('score', 0)
                            gc = d.get('gol_coherence', 0)
                            pc = d.get('projection_complexity', 0)
                            mi = d.get('slice_mi', 0)
                            is_sel = (idx == self.discovery_index)
                            label = f"#{idx:3d} S={sc:.2f} G={gc:.2f} P={pc:.2f} M={mi:.2f}##d{idx}"
                            if imgui.selectable(label, is_sel)[0]:
                                self._load_discovery(idx)
                        imgui.tree_pop()
                imgui.end_child()
            else:
                imgui.end_child()

        imgui.separator()

        # Renderer mode toggle
        imgui.text("Renderer:")
        renderer_modes = ["Volumetric", "Voxel"]
        cur_idx = 1 if self.renderer_mode == 'voxel' else 0
        if imgui.begin_combo("##renderer_mode", renderer_modes[cur_idx]):
            for i, label in enumerate(renderer_modes):
                is_sel = (i == cur_idx)
                if imgui.selectable(label, is_sel)[0]:
                    self.renderer_mode = 'voxel' if i == 1 else 'volumetric'
                if is_sel:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        imgui.text("Boundary:")
        boundary_modes = ["Toroidal (wrap)", "Clamped (Dirichlet)", "Mirror (Neumann)"]
        cur_b = self.boundary_mode
        if imgui.begin_combo("##boundary_mode", boundary_modes[cur_b]):
            for i, label in enumerate(boundary_modes):
                is_sel = (i == cur_b)
                if imgui.selectable(label, is_sel)[0]:
                    self.boundary_mode = i
                if is_sel:
                    imgui.set_item_default_focus()
            imgui.end_combo()

        if self.renderer_mode == 'voxel':
            # Voxel-specific settings
            changed, new_val = imgui.slider_float("Cube gap", self.voxel_gap, 0.0, 0.5)
            if changed:
                self.voxel_gap = new_val

            if not self.is_element_ca:
                changed, new_val = imgui.slider_float("Threshold", self.voxel_threshold, 0.01, 0.99)
                if changed:
                    self.voxel_threshold = new_val
                    self._cull_valid = False

            changed, new_val = imgui.slider_float("Opacity", self.voxel_alpha, 0.1, 1.0)
            if changed:
                self.voxel_alpha = new_val

            changed, new_val = imgui.slider_float("Brightness", self.brightness, 0.1, 5.0)
            if changed:
                self.brightness = new_val

            if not self.is_element_ca:
                colormap_names = ["Fire", "Cool", "Grayscale", "Neon", "Discrete", "Spectral"]
                if imgui.begin_combo("##colormap", colormap_names[min(self.colormap, len(colormap_names)-1)]):
                    for i, label in enumerate(colormap_names):
                        is_sel = (i == self.colormap)
                        if imgui.selectable(label, is_sel)[0]:
                            self.colormap = i
                        if is_sel:
                            imgui.set_item_default_focus()
                    imgui.end_combo()
                self._draw_colormap_legend()

            if hasattr(self, '_last_voxel_count'):
                imgui.text(f"Visible voxels: {self._last_voxel_count:,}")
                cpd = getattr(self, '_voxel_chunks_per_dim', 1)
                if cpd > 1:
                    imgui.text(f"Render passes: {cpd**3} ({cpd}x{cpd}x{cpd})")
                if getattr(self, '_voxels_clipped', False):
                    imgui.text_colored(imgui.ImVec4(1.0, 0.7, 0.0, 1.0),
                                       f"Voxel buffer full ({self.max_voxels:,} max)")
                    imgui.text_colored(imgui.ImVec4(1.0, 0.7, 0.0, 1.0),
                                       "Switch to Volumetric for full view")

        else:
            # Volumetric render settings
            render_modes = ["Volume", "Iso-surface", "Max Intensity"]
            if imgui.begin_combo("##render_mode", render_modes[self.render_mode]):
                for i, label in enumerate(render_modes):
                    is_sel = (i == self.render_mode)
                    if imgui.selectable(label, is_sel)[0]:
                        self.render_mode = i
                    if is_sel:
                        imgui.set_item_default_focus()
                imgui.end_combo()

            changed, new_val = imgui.slider_float("Density", self.density_scale, 0.1, 50.0,
                                                   flags=imgui.SliderFlags_.logarithmic)
            if changed:
                self.density_scale = new_val

            changed, new_val = imgui.slider_float("Brightness", self.brightness, 0.1, 5.0)
            if changed:
                self.brightness = new_val

            if self.render_mode == 1:
                changed, new_val = imgui.slider_float("Iso threshold", self.iso_threshold, 0.01, 0.99)
                if changed:
                    self.iso_threshold = new_val

            colormap_names = ["Fire", "Cool", "Grayscale", "Neon", "Discrete", "Spectral"]
            if imgui.begin_combo("##colormap", colormap_names[min(self.colormap, len(colormap_names)-1)]):
                for i, label in enumerate(colormap_names):
                    is_sel = (i == self.colormap)
                    if imgui.selectable(label, is_sel)[0]:
                        self.colormap = i
                    if is_sel:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            self._draw_colormap_legend()

            # Performance options
            imgui.separator()
            imgui.text("Performance:")
            changed, val = imgui.checkbox("Half-res volume", self._use_half_res)
            if changed:
                self._use_half_res = val
            if imgui.is_item_hovered():
                imgui.set_tooltip("Render at half resolution then upsample (2-4x faster)")
            changed, val = imgui.checkbox("Compute raymarcher", self._use_compute_ray)
            if changed:
                self._use_compute_ray = val
            if imgui.is_item_hovered():
                imgui.set_tooltip("Use compute shader raymarcher with shared memory cache")

        # Visualization channel selector (for multi-field rules)
        if len(self.vis_channels) > 1 and self.renderer_mode != 'voxel':
            imgui.separator()
            imgui.text("Channel:")
            if imgui.begin_combo("##vis_channel", self.vis_channels[self.vis_channel]):
                for i, name in enumerate(self.vis_channels):
                    is_sel = (i == self.vis_channel)
                    if imgui.selectable(name, is_sel)[0]:
                        if self.vis_channel != i:
                            self.vis_channel = i
                            self._accel_textures_valid = False
                    if is_sel:
                        imgui.set_item_default_focus()
                imgui.end_combo()

            changed, self.vis_abs = imgui.checkbox("Absolute value", self.vis_abs)
            if changed:
                self._accel_textures_valid = False

        imgui.separator()

        # Slice mode (volumetric only)
        if self.renderer_mode != 'voxel':
            changed, enabled = imgui.checkbox("Slice view", self.slice_pos >= 0.0)
            if changed:
                self.slice_pos = 0.5 if enabled else -1.0

            if self.slice_pos >= 0.0:
                axis_labels = ["X", "Y", "Z"]
                if imgui.begin_combo("##slice_axis", axis_labels[self.slice_axis]):
                    for i, label in enumerate(axis_labels):
                        is_sel = (i == self.slice_axis)
                        if imgui.selectable(label, is_sel)[0]:
                            self.slice_axis = i
                        if is_sel:
                            imgui.set_item_default_focus()
                    imgui.end_combo()

                changed, new_val = imgui.slider_float("Position", self.slice_pos, 0.0, 1.0)
                if changed:
                    self.slice_pos = new_val

        imgui.separator()
        if self.sandbox_mode or self.paint_mode:
            imgui.text("PAINT: L=place, R=erase, Mid=orbit, scroll=zoom")
        else:
            imgui.text("Mouse: L-drag=orbit, R-drag=pan, scroll=zoom")

        if not self.sandbox_mode:
            changed, val = imgui.checkbox("Paint mode [P]", self.paint_mode)
            if changed:
                self.paint_mode = val
            if self.paint_mode:
                changed, new_r = imgui.slider_int("Brush radius", self.brush_size, 1, 5)
                if changed:
                    self.brush_size = new_r

        imgui.text("Keys: Space=play, R=reset, V=renderer, P=paint")

        imgui.end()

        # ── Sandbox tools panel ───────────────────────────────────────
        if self.sandbox_mode:
            imgui.set_next_window_pos(imgui.ImVec2(10, self.height - 220.0),
                                       imgui.Cond_.first_use_ever)
            imgui.set_next_window_size(imgui.ImVec2(320, 0), imgui.Cond_.first_use_ever)
            imgui.begin("Sandbox Tools", flags=imgui.WindowFlags_.always_auto_resize)

            # Tool selection
            tool_names = ["Element [1]", "Temperature [2]", "Eraser [3]"]
            for i, name in enumerate(tool_names):
                if i > 0:
                    imgui.same_line()
                selected = (self.brush_tool == i)
                if selected:
                    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.3, 0.6, 1.0, 1.0))
                if imgui.button(name):
                    self.brush_tool = i
                if selected:
                    imgui.pop_style_color()

            # Brush size
            changed, new_val = imgui.slider_int("Brush size", self.brush_size, 1, 8)
            if changed:
                self.brush_size = new_val

            imgui.separator()

            if self.brush_tool == 0:
                # Element palette
                imgui.text("Element:")
                cols = 4
                for idx, elem_id in enumerate(self.palette_elements):
                    if idx % cols != 0:
                        imgui.same_line()
                    sym = SYMBOLS[elem_id] if elem_id < len(SYMBOLS) else "?"
                    name = sym if elem_id == 0 else f"{sym} ({elem_id})"
                    if elem_id == 0:
                        name = "Vacuum"
                    elif elem_id == WALL_ID:
                        name = "Wall"

                    selected = (self.brush_element == elem_id)
                    if selected:
                        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.3, 0.6, 1.0, 1.0))

                    # Color preview button
                    if elem_id > 0 and elem_id < len(ELEMENT_GPU_DATA):
                        r_c = ELEMENT_GPU_DATA[elem_id, 8]
                        g_c = ELEMENT_GPU_DATA[elem_id, 9]
                        b_c = ELEMENT_GPU_DATA[elem_id, 10]
                        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(r_c, g_c, b_c, 1.0))

                    if imgui.button(f"{sym}##{idx}", imgui.ImVec2(60, 28)):
                        self.brush_element = elem_id

                    if elem_id > 0 and elem_id < len(ELEMENT_GPU_DATA):
                        imgui.pop_style_color()
                    if selected:
                        imgui.pop_style_color()

                    if imgui.is_item_hovered():
                        full_name = NAMES[elem_id] if elem_id < len(NAMES) else "Unknown"
                        imgui.set_tooltip(full_name)

                # Current selection display
                sel_sym = SYMBOLS[self.brush_element] if self.brush_element < len(SYMBOLS) else "?"
                sel_name = NAMES[self.brush_element] if self.brush_element < len(NAMES) else "Unknown"
                imgui.text(f"Selected: {sel_sym} - {sel_name}")

            elif self.brush_tool == 1:
                # Temperature brush
                changed, new_val = imgui.slider_float("Temperature °C", self.brush_temp,
                                                       -200.0, 5000.0, "%.0f")
                if changed:
                    self.brush_temp = new_val

            imgui.separator()

            # Save / Load
            if imgui.button("Save State"):
                import os
                save_dir = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(save_dir, "sandbox_save.npz")
                self._save_state(path)
                self._save_msg = f"Saved to {os.path.basename(path)}"
                self._save_msg_time = time.time()

            imgui.same_line()
            if imgui.button("Load State"):
                import os
                save_dir = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(save_dir, "sandbox_save.npz")
                if os.path.exists(path):
                    self._load_state(path)
                    self._save_msg = "Loaded!"
                    self._save_msg_time = time.time()
                else:
                    self._save_msg = "No save file found"
                    self._save_msg_time = time.time()

            if hasattr(self, '_save_msg') and time.time() - self._save_msg_time < 3.0:
                imgui.same_line()
                imgui.text(self._save_msg)

            imgui.end()

        # Recording indicator (top-right corner, always visible during recording)
        if self._recording:
            elapsed = time.time() - self._rec_start_time
            blink = int(elapsed * 2) % 2 == 0
            rec_text = f"  REC {elapsed:.0f}s  "
            text_size = imgui.calc_text_size(rec_text)
            pad = 10
            imgui.set_next_window_pos(
                imgui.ImVec2(self.width - text_size.x - pad * 3, pad))
            imgui.set_next_window_size(imgui.ImVec2(0, 0))
            imgui.push_style_var(imgui.StyleVar_.window_rounding, 8.0)
            imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(0.15, 0.0, 0.0, 0.85))
            imgui.begin("##rec_indicator",
                        flags=(imgui.WindowFlags_.no_title_bar |
                               imgui.WindowFlags_.no_resize |
                               imgui.WindowFlags_.no_move |
                               imgui.WindowFlags_.always_auto_resize |
                               imgui.WindowFlags_.no_mouse_inputs |
                               imgui.WindowFlags_.no_focus_on_appearing |
                               imgui.WindowFlags_.no_nav))
            if blink:
                imgui.text_colored(imgui.ImVec4(1.0, 0.15, 0.15, 1.0), rec_text)
            else:
                imgui.text_colored(imgui.ImVec4(0.6, 0.1, 0.1, 1.0), rec_text)
            imgui.end()
            imgui.pop_style_color()
            imgui.pop_style_var()

    # ── Video recording ─────────────────────────────────────────────────

    def _init_rec_fbo(self):
        """Create (or recreate) the offscreen FBO for recording."""
        w, h = self._rec_width, self._rec_height
        # Cleanup old
        for attr in ('_rec_fbo', '_rec_rbo_color', '_rec_rbo_depth'):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()

        self._rec_rbo_color = self.ctx.renderbuffer((w, h))
        self._rec_rbo_depth = self.ctx.depth_renderbuffer((w, h))
        self._rec_fbo = self.ctx.framebuffer(
            color_attachments=[self._rec_rbo_color],
            depth_attachment=self._rec_rbo_depth)

    @staticmethod
    def _rec_writer_loop(queue, proc):
        """Background thread: pull frame data from queue, write to ffmpeg stdin."""
        while True:
            data = queue.get()
            if data is None:
                break
            if proc.poll() is not None:
                break  # ffmpeg has exited
            try:
                proc.stdin.write(data)
            except (BrokenPipeError, OSError):
                break

    def _start_recording(self):
        """Start piping rendered frames to ffmpeg at 1440p60.

        Recording is an opt-in feature gated behind the CA_RECORDING_ENABLED
        environment variable so that clones of the repo don't expose the
        feature to anonymous users by default.  Set it in your shell profile:

            export CA_RECORDING_ENABLED=1
        """
        if self._recording:
            return
        if not os.environ.get('CA_RECORDING_ENABLED'):
            self._rec_msg = "Recording disabled — set CA_RECORDING_ENABLED=1 to enable"
            self._rec_msg_time = time.time()
            return
        if not shutil.which('ffmpeg'):
            self._rec_msg = "ffmpeg not found on PATH"
            self._rec_msg_time = time.time()
            return

        os.makedirs('recordings', exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        # Sanitise: rule labels may contain '/' (e.g. '4/4/5 Crystal' GOL
        # notation), parens, or other characters that confuse the path or
        # later ffmpeg/json parsers.  Keep alnum, '.', '_', '-' only.
        raw_label = RULE_PRESETS[self.rule_name]['label']
        rule_label = re.sub(r'[^A-Za-z0-9._-]+', '_', raw_label.replace(' ', '_')).strip('_') or 'rule'
        self._rec_filename = f'recordings/{timestamp}_{rule_label}.mp4'

        w, h = self._rec_width, self._rec_height
        self._init_rec_fbo()

        # --- Build text overlay via ffmpeg drawtext filters ---
        font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        fontb = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
        if not os.path.exists(fontb):
            fontb = font

        preset = RULE_PRESETS[self.rule_name]
        title = preset['label']
        desc = preset.get('description', '')
        # Build param string from current values
        param_parts = []
        for k, v in self.params.items():
            param_parts.append(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}')
        param_str = '  '.join(param_parts)

        # Escape special chars for ffmpeg drawtext (: ' \ need escaping)
        def _esc(s):
            return s.replace('\\', '\\\\').replace("'", "\u2019").replace(':', '\\:').replace('%', '%%')

        vf_parts = ['vflip']
        base_y = 20
        # Title
        vf_parts.append(
            f"drawtext=fontfile='{fontb}':text='{_esc(title)}':"
            f"fontcolor=white:fontsize=42:x=24:y={base_y}:"
            f"borderw=2:bordercolor=black@0.8")
        # Description (under title)
        if desc:
            vf_parts.append(
                f"drawtext=fontfile='{font}':text='{_esc(desc)}':"
                f"fontcolor=white:fontsize=24:x=24:y={base_y + 52}:"
                f"borderw=2:bordercolor=black@0.8")
        # Parameters (bottom-left)
        if param_str:
            vf_parts.append(
                f"drawtext=fontfile='{font}':text='{_esc(param_str)}':"
                f"fontcolor=white:fontsize=20:x=24:y=h-42:"
                f"borderw=2:bordercolor=black@0.8")
        # Seed + grid size (bottom-right)
        info_str = f'Seed {self.seed}  Grid {self.size}x{self.size}x{self.size}'
        vf_parts.append(
            f"drawtext=fontfile='{font}':text='{_esc(info_str)}':"
            f"fontcolor=white:fontsize=20:x=w-tw-24:y=h-42:"
            f"borderw=2:bordercolor=black@0.8")
        # Discovery score (top-right, if viewing a discovery)
        if self.discovery_index >= 0 and self.discovery_index < len(self.discoveries):
            score = self.discoveries[self.discovery_index].get('score', 0)
            score_str = f'Score\\: {score:.2f}'
            vf_parts.append(
                f"drawtext=fontfile='{fontb}':text='{score_str}':"
                f"fontcolor=yellow:fontsize=28:x=w-tw-24:y=20:"
                f"borderw=2:bordercolor=black@0.8")

        vf_chain = ','.join(vf_parts)

        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            # Input: raw RGB from pipe
            '-f', 'rawvideo',
            '-pixel_format', 'rgb24',
            '-video_size', f'{w}x{h}',
            '-framerate', str(self._rec_fps),
            '-i', 'pipe:0',
            # Flip + text overlay + encode
            '-vf', vf_chain,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-threads', str(min(os.cpu_count() or 4, 8)),
            self._rec_filename,
        ]
        self._rec_process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, bufsize=w * h * 3 * 2)

        # Verify ffmpeg started successfully
        time.sleep(0.05)
        if self._rec_process.poll() is not None:
            stderr = self._rec_process.stderr.read().decode(errors='replace').strip()
            self._rec_msg = f"ffmpeg failed: {stderr[:120]}" if stderr else "ffmpeg failed to start"
            self._rec_msg_time = time.time()
            self._rec_process = None
            return

        self._recording = True
        self._rec_start_time = time.time()
        self._rec_frame_count = 0

        # Background writer thread to avoid blocking main loop on pipe writes
        self._rec_write_queue = _queue.Queue(maxsize=30)
        self._rec_write_thread = threading.Thread(
            target=self._rec_writer_loop,
            args=(self._rec_write_queue, self._rec_process),
            daemon=True)
        self._rec_write_thread.start()

        self._rec_msg = f"Recording {w}x{h}@{self._rec_fps}fps"
        self._rec_msg_time = time.time()

    def _stop_recording(self):
        """Finish recording and close ffmpeg."""
        if not self._recording:
            return
        self._recording = False

        # Signal writer thread to stop, then wait for it to drain.
        # Bounded waits so a hung writer cannot block window close.
        if self._rec_write_queue is not None:
            try:
                self._rec_write_queue.put(None, timeout=1.0)  # sentinel
            except Exception:
                pass
        if self._rec_write_thread is not None:
            self._rec_write_thread.join(timeout=3.0)
            if self._rec_write_thread.is_alive():
                # Daemon thread will be killed at interpreter exit; warn the user.
                print("[record] WARNING: writer thread did not exit cleanly", flush=True)
            self._rec_write_thread = None
        self._rec_write_queue = None

        # Close ffmpeg stdin after writer thread has finished (avoids BrokenPipeError).
        # Bounded wait + escalating SIGTERM/SIGKILL so a hung ffmpeg cannot
        # stall the GL shutdown sequence (which is what crashes NVK).
        if self._rec_process:
            proc = self._rec_process
            self._rec_process = None
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=1.0)
                    except Exception:
                        pass
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        duration = time.time() - self._rec_start_time
        self._rec_msg = f"Saved {self._rec_filename} ({self._rec_frame_count} frames, {duration:.1f}s)"
        self._rec_msg_time = time.time()

        # Sidecar write must never crash the simulator on stop — the video
        # is already saved, missing metadata is recoverable.
        try:
            self._write_recording_metadata()
        except Exception as e:
            self._rec_msg = f"WARN: sidecar write failed: {e}"
            self._rec_msg_time = time.time()

    def _render_to_rec_fbo(self):
        """Render the scene into the recording FBO at fixed resolution."""
        # Save current state
        old_w, old_h = self.width, self.height
        old_viewport = self.ctx.viewport

        # Bind recording FBO and set viewport to recording resolution
        self._rec_fbo.use()
        self.width = self._rec_width
        self.height = self._rec_height
        self.ctx.viewport = (0, 0, self._rec_width, self._rec_height)

        try:
            # Render the scene (same as screen, just different resolution)
            if self.renderer_mode == 'voxel':
                self._render_voxels()
            else:
                self._render_volume()
        finally:
            # Restore screen state even if rendering fails
            self.ctx.screen.use()
            self.width = old_w
            self.height = old_h
            self.ctx.viewport = old_viewport

    def _record_frame(self):
        """Read back the recording FBO and queue frame data for ffmpeg."""
        if not self._recording or not self._rec_process:
            return

        # Render scene to recording FBO
        self._render_to_rec_fbo()

        # Read pixels via moderngl (handles format matching internally)
        data = self._rec_fbo.read(components=3)
        try:
            self._rec_write_queue.put_nowait(bytes(data))
            self._rec_frame_count += 1
        except _queue.Full:
            # ffmpeg is falling behind — record the loss visibly so users notice
            # before checking the resulting MP4. Throttle the log so a sustained
            # backlog doesn't spam stderr.
            self._rec_dropped_frames = getattr(self, '_rec_dropped_frames', 0) + 1
            if self._rec_dropped_frames % 30 == 1:
                sys.stderr.write(
                    f"[recording] dropped {self._rec_dropped_frames} frame(s); "
                    f"ffmpeg writer queue is saturated\n")
                self._rec_msg = (
                    f"WARN: dropped {self._rec_dropped_frames} frame(s)")
                self._rec_msg_time = time.time()

    def _write_recording_metadata(self):
        """Write a JSON sidecar with rule info, params, and description."""
        meta_path = self._rec_filename.rsplit('.', 1)[0] + '.json'
        preset = RULE_PRESETS[self.rule_name]
        meta = {
            'rule': self.rule_name,
            'label': preset['label'],
            'description': preset.get('description', ''),
            'params': {k: float(v) if isinstance(v, (int, float)) else v
                       for k, v in self.params.items()},
            'dt': self.dt,
            'size': self.size,
            'seed': self.seed,
            'renderer_mode': self.renderer_mode,
            'colormap': self.colormap,
            'frames': self._rec_frame_count,
            'fps': self._rec_fps,
            'resolution': [self._rec_width, self._rec_height],
            'duration_sec': round(time.time() - self._rec_start_time, 2),
        }
        if self.discovery_index >= 0 and self.discovery_index < len(self.discoveries):
            disc = self.discoveries[self.discovery_index]
            meta['discovery_score'] = disc.get('score', 0)
            meta['discovery_index'] = self.discovery_index
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self):
        frame_times = []
        try:
            while not glfw.window_should_close(self.window):
                t0 = time.time()
                glfw.poll_events()
                self.imgui_renderer.process_inputs()

                # Update framebuffer size
                fb_w, fb_h = glfw.get_framebuffer_size(self.window)
                if fb_w > 0 and fb_h > 0:
                    self.width = fb_w
                    self.height = fb_h
                    self.ctx.viewport = (0, 0, fb_w, fb_h)

                # Simulation step (rate-limited)
                if not self.paused:
                    now = time.time()
                    should_step = False
                    if self.target_sps <= 0:
                        should_step = True  # unlimited — step every frame
                    else:
                        interval = 1.0 / self.target_sps
                        if now - self._last_step_time >= interval:
                            should_step = True
                            self._last_step_time = now

                    if should_step:
                        for _ in range(self.sim_speed):
                            self._step_sim()

                        # Live scoring (periodic GPU reduction — no full readback)
                        self._score_frame += 1
                        if self._score_frame >= self._score_interval:
                            self._score_frame = 0
                            self._update_score()

                # Render — skip if scene state is identical to last frame
                # (camera still, sim paused or between steps, UI knobs idle).
                # Paying ~0.3 ms for a screen-size blit beats a full ray-march
                # which at 512³ can be 10–30 ms. Recording forces a full render
                # because the video pipeline reads the backbuffer each frame.
                cur_hash = self._scene_hash()
                if (cur_hash == self._last_scene_hash
                        and self._scene_cache_fbo is not None
                        and self._scene_cache_size == (self.width, self.height)
                        and not self._recording):
                    self._restore_scene_snapshot()
                else:
                    self._render()
                    self._snapshot_scene()
                    self._last_scene_hash = cur_hash

                # Capture frame for video (scene only, before UI overlay)
                if self._recording:
                    self._record_frame()

                # Render UI
                imgui.new_frame()
                self._draw_ui()
                imgui.render()
                self.imgui_renderer.render(imgui.get_draw_data())

                glfw.swap_buffers(self.window)

                # FPS tracking
                dt_frame = time.time() - t0
                frame_times.append(dt_frame)
                if len(frame_times) > 60:
                    frame_times.pop(0)
                if len(frame_times) >= 10:
                    avg = sum(frame_times) / len(frame_times)
                    fps = 1.0 / avg if avg > 0 else 0
                    glfw.set_window_title(self.window,
                        f"3D CA — {RULE_PRESETS[self.rule_name]['label']} [{self.renderer_mode}] — {fps:.0f} FPS — {self.size}³")
        finally:
            # Even if _cleanup raises, we MUST tear down GLFW — otherwise the
            # zombie X window can wedge the compositor and any leaked GL
            # context can crash the kernel module on next start.
            try:
                self._cleanup()
            except Exception as e:
                print(f"[shutdown] cleanup raised: {e!r}", flush=True)
                try:
                    glfw.terminate()
                except Exception:
                    pass

    def _cleanup(self):
        """Release all GPU resources in safe dependency order, then tear down window.

        Why the careful ordering matters:
          - Mesa NVK / Nouveau is documented to deadlock the kernel module
            (requiring a power cycle) when a GL context is destroyed with
            unreleased compute shaders, image bindings, or 3-D textures.
          - moderngl tracks every object; if any survives until Python GC
            runs after `glfw.terminate()`, its `__del__` will issue GL calls
            into a dead context — classic use-after-free.
          - imgui_renderer holds its own GL state (font atlas, shader,
            VBO/VAO). It must be torn down WHILE the GL context is still
            current, but BEFORE the moderngl context itself is released.
        """
        # 1. Stop the recording subprocess and writer thread before touching GL.
        #    _stop_recording is bounded by ffmpeg's wait timeout; if that hangs
        #    we'd rather kill ffmpeg than block forever during shutdown.
        if self._recording:
            try:
                self._stop_recording()
            except Exception:
                pass

        # 2. Flush all in-flight GPU work. Without this, a still-running
        #    compute dispatch can outlive its program object and crash the
        #    driver when we release that program below.
        try:
            GL.glFinish()
        except Exception:
            pass

        # 3. Tear down imgui FIRST — it owns GL textures/programs and needs
        #    a live, current context to release them cleanly.
        try:
            self.imgui_renderer.shutdown()
        except Exception:
            pass
        try:
            imgui.destroy_context()
        except Exception:
            pass

        # 4. Release every GL resource in strict dependency order.
        #
        #    Order rules:
        #      a) VAOs reference programs + buffers — release first.
        #      b) Programs (graphics + compute) — release second.
        #      c) Buffers, textures, FBOs, renderbuffers — leaf resources, last.
        #
        #    The previous list was missing 11 objects (acceleration textures,
        #    half-res FBO, compute raymarcher, occupancy/min-max programs and
        #    textures, upsample program/VAO). Those leaks were what made NVK
        #    hang on close.
        release_order = [
            # ── VAOs (must precede any program or buffer they reference) ──
            'voxel_vao', 'vao', '_upsample_vao',

            # ── Programs (graphics + compute) ──
            'compute_prog',          # active CA compute shader
            'voxel_cull_prog',       # voxel face-culling compute
            'voxel_prog',            # instanced voxel raster
            'render_prog',           # full-screen volume raymarcher (fragment)
            '_metrics_prog',         # GPU metrics reduction
            '_indirect_reset_prog',  # indirect-draw counter reset
            '_compute_ray_prog',     # compute-shader raymarcher
            '_accel_prog',           # fused occupancy + min/max builder
            '_view_build_prog',      # reduced-precision view tex builder
            '_upsample_prog',        # half-res upsample fragment program

            # ── Buffers (SSBOs, VBOs, indirect, staging) ──
            'voxel_buffer',
            'voxel_indirect_buffer',
            '_voxel_total_counter',
            '_metrics_ssbo',
            'vbo',
            'element_ssbo',
            '_total_staging',

            # ── 3-D simulation textures (largest allocations — safest last) ──
            'tex_a', 'tex_b',

            # ── Acceleration textures (R8UI occupancy + RG16F min/max + R16F view) ──
            '_occ_tex', '_mm_tex', '_view_tex',

            # ── Scene cache (idle-frame blit target) ──
            '_scene_cache_fbo', '_scene_cache_tex',

            # ── Recording FBO + renderbuffers ──
            '_rec_fbo', '_rec_rbo_color', '_rec_rbo_depth',

            # ── Half-res rendering pair (FBO before its color attachment) ──
            '_half_res_fbo', '_half_res_tex',

            # ── Compute-raymarcher output pair (FBO before color attachment) ──
            '_cr_fbo', '_cr_output_tex',
        ]
        for attr in release_order:
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            try:
                obj.release()
            except Exception:
                pass
            # Clear the attribute so a later __del__ during Python GC
            # cannot double-release into a dead context.
            setattr(self, attr, None)

        # 5. Second glFinish: any release() above may have queued deferred
        #    work (e.g. NVK schedules texture frees on a worker queue).
        try:
            GL.glFinish()
        except Exception:
            pass

        # 6. Release the moderngl Context itself. This catches anything we
        #    forgot above and tells moderngl not to chase its own dead
        #    references during interpreter shutdown.
        try:
            self.ctx.release()
        except Exception:
            pass
        # Drop our reference so Python GC won't touch the Context again.
        self.ctx = None

        # 7. Finally tear down GLFW. After this point the GL context is
        #    gone — any surviving GL handle that hits __del__ is a bug.
        try:
            glfw.terminate()
        except Exception:
            pass


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3D CA Simulator")
    parser.add_argument('--size', type=int, default=64, help='Grid size (default: 64)')
    parser.add_argument('--rule', type=str, default='game_of_life_3d',
                        choices=list(RULE_PRESETS.keys()),
                        help='Rule preset (default: game_of_life_3d)')
    parser.add_argument('--discovery', type=str, default=None,
                        help='Load discovery from JSON file (index with --discovery-index)')
    parser.add_argument('--discovery-index', type=int, default=0,
                        help='Which discovery to load (0-based index)')
    args = parser.parse_args()

    # Load discovery params if specified
    if args.discovery:
        import json
        with open(args.discovery) as f:
            discoveries = json.load(f)
        if args.discovery_index >= len(discoveries):
            print(f"Discovery index {args.discovery_index} out of range (have {len(discoveries)})")
            sys.exit(1)
        disc = discoveries[args.discovery_index]
        args.rule = disc['rule']
        sim = Simulator(size=args.size, rule=args.rule)
        # Override params from discovery
        for k, v in disc['params'].items():
            if k in sim.params:
                sim.params[k] = v
        if 'dt' in disc:
            sim.dt = disc['dt']
        if 'seed' in disc:
            sim.seed = disc['seed']
            sim._reset()
        print(f"Loaded discovery #{args.discovery_index}: {disc['rule']} "
              f"score={disc.get('score', '?')} params={disc['params']}")
    else:
        sim = Simulator(size=args.size, rule=args.rule)

    sim.run()


if __name__ == '__main__':
    main()
