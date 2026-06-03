
#version 430
layout(local_size_x=8, local_size_y=8, local_size_z=8) in;

layout(rgba32f, binding=0) uniform image3D u_src;
layout(rgba32f, binding=1) uniform image3D u_dst;

// `u_size` is declared early because the optional deposit() helper
// below needs it for SSBO indexing.  All other CA uniforms follow
// the deposit block.
uniform int u_size;

// ── Optional particle→field deposit channel ────────────────────────
// When the active preset enables deposit (any non-zero deposit_strength
// or trail_strength), the simulator compiles every CA pass with
// DEPOSIT_ENABLED=1 and binds an r32f image at unit 12 holding the
// per-voxel sum of particle splats from this step.  CA shaders that
// want to react to particles call `deposit(pos)` and add the result
// (scaled by `u_deposit_strength`) into a chosen channel.  Shaders
// that ignore deposit pay zero cost — `deposit()` returns 0 and the
// optimiser deletes the load+multiply.
#ifndef DEPOSIT_ENABLED
#define DEPOSIT_ENABLED 0
#endif
uniform float u_deposit_strength;     // always declared; 0 when no deposit
uniform int   u_deposit_channel;      // informational; shader picks where
#if DEPOSIT_ENABLED
// SSBO instead of image: image units are limited to 8 on this driver,
// so we use SSBO binding 15 (the image namespace was full).  Atomic
// float adds on SSBOs are provided by GL_NV_shader_atomic_float in the
// particle shader; the CA shader only reads.
layout(std430, binding = 15) readonly buffer DepositBuf { float data[]; } u_deposit_buf;
float deposit(ivec3 p) {
    // Use u_dims (W,H,D) for correct linear indexing on non-cubic grids.
    // u_dims is injected by _compile_compute alongside u_size.
    int idx = (p.z * u_dims.y + p.y) * u_dims.x + p.x;
    return u_deposit_buf.data[idx];
}
#else
float deposit(ivec3 p) { return 0.0; }
#endif

uniform float u_dt;
uniform float u_param0;
uniform float u_param1;
uniform float u_param2;
uniform float u_param3;
uniform int u_boundary;  // 0 = toroidal (wrap), 1 = clamped (Dirichlet, zero outside), 2 = mirror (Neumann, zero-flux)
uniform int u_frame;     // step counter for temporal noise

// ── Viewport pose (used by procedural "viewport" kind shaders, e.g. fractal
// zoom-throughs). For ordinary CA shaders these are bound but ignored.
// u_origin = world-space point at the centre of the cube
// u_zoom   = half-width of the cube in world units (cube spans u_origin ± u_zoom)
uniform vec3  u_origin;
uniform float u_zoom;

// ── Auxiliary uniforms for multi-fractal viewport shaders ─────────
// u_aux3   = generic vec3 slot (e.g. Julia c-vector)
// u_aux_a  = generic float (e.g. Mandelbox scale, Menger fold count)
// u_aux_b  = generic float (e.g. Mandelbox min radius)
// All three are bound from the named preset params 'Julia cx/cy/cz',
// 'Box scale', 'Folds', 'Min radius' when present. Optimised out of
// shaders that don't reference them.
uniform vec3  u_aux3;
uniform float u_aux_a;
uniform float u_aux_b;

// ── Grid-spacing scale factors ──────────────────────────────────────
// PDE rules use discrete Laplacians on a grid with voxel spacing h = REF/size
// (in "reference" units — physical domain is fixed at REF_SIZE, independent
// of grid resolution). The continuous Laplacian is
//     ∇²U ≈ (sum_neighbors - 6*center) / h²
// so rules MULTIPLY the raw stencil by h_sq to obtain ∇²U. That makes h_sq
// equal to 1/h² = (size/REF)², not h² itself — the variable name refers to
// "the factor we multiply by", not "h squared" literally.
// At REF_SIZE both definitions collapse to 1, which is why this bug went
// unnoticed for a long time (presets were tuned at REF_SIZE=128).
// h_inv = size/REF for scaling radii and gradients (physical length → voxels).
const float REF_SIZE = 128.0;
float h_inv = float(u_size) / REF_SIZE;                 // multiply radii by this
float h_sq  = h_inv * h_inv;                            // multiply Laplacians by this (= 1/h²)

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

// Reciprocal of UINT32_MAX — multiplying is ~2x faster than dividing on GPU.
// Used to normalise Wang-hashed uint seeds to [0,1).
const float INV_U32_MAX = 1.0 / 4294967295.0;

// Temporal hash: changes every frame (unlike position-only fract(sin(...)))
// Uses Wang hash on (pos + frame) for high-quality pseudo-random numbers
float hash_temporal(ivec3 p, int channel) {
    uint seed = uint(p.x * 73856093) ^ uint(p.y * 19349663) ^ uint(p.z * 83492791)
              ^ uint(u_frame * 2654435761u) ^ uint(channel * 668265263u);
    seed = (seed ^ (seed >> 16u)) * 0x45d9f3bu;
    seed = (seed ^ (seed >> 16u)) * 0x45d9f3bu;
    seed = seed ^ (seed >> 16u);
    return float(seed) * INV_U32_MAX;
}

// Quenched (frozen) hash: depends on position only — same value every frame.
// Use this for representing material heterogeneities like crystal lattice
// defects, dopant atoms, or surface impurities that don't move with time.
// Different `channel` values give independent random fields.
float hash_static(ivec3 p, int channel) {
    uint seed = uint(p.x * 73856093) ^ uint(p.y * 19349663) ^ uint(p.z * 83492791)
              ^ uint(channel * 2246822519u);
    seed = (seed ^ (seed >> 16u)) * 0x45d9f3bu;
    seed = (seed ^ (seed >> 16u)) * 0x45d9f3bu;
    seed = seed ^ (seed >> 16u);
    return float(seed) * INV_U32_MAX;
}

// SMOOTH VALUE NOISE on a unit-spaced lattice (8-corner trilinear interp,
// smoothstep blend). Output ~uniform [0,1]. Building block for fbm3.
float value_noise3(vec3 p, int channel) {
    vec3 pf = floor(p);
    vec3 t  = p - pf;
    t = t * t * (3.0 - 2.0 * t);  // C1-continuous smoothstep blend
    ivec3 i = ivec3(pf);
    float c000 = hash_static(i + ivec3(0,0,0), channel);
    float c100 = hash_static(i + ivec3(1,0,0), channel);
    float c010 = hash_static(i + ivec3(0,1,0), channel);
    float c110 = hash_static(i + ivec3(1,1,0), channel);
    float c001 = hash_static(i + ivec3(0,0,1), channel);
    float c101 = hash_static(i + ivec3(1,0,1), channel);
    float c011 = hash_static(i + ivec3(0,1,1), channel);
    float c111 = hash_static(i + ivec3(1,1,1), channel);
    float x00 = mix(c000, c100, t.x);
    float x10 = mix(c010, c110, t.x);
    float x01 = mix(c001, c101, t.x);
    float x11 = mix(c011, c111, t.x);
    float y0  = mix(x00, x10, t.y);
    float y1  = mix(x01, x11, t.y);
    return mix(y0, y1, t.z);
}

// FRACTIONAL BROWNIAN MOTION (3D, octave sum of value noise).
//
// fbm3(p, base_period, octaves, channel)
//   base_period: lattice cells per period at the LOWEST frequency octave
//                (octave 0 has wavelength ~ base_period cells)
//   octaves:    number of doublings; each octave halves wavelength and amp
//   channel:    seed offset; each octave gets channel + i*17 internally
//
// Output normalised to ~[0,1] (centred at ~0.5). The point of this over
// raw white noise: pattern-forming PDEs on a uniform grid can only resolve
// instabilities at one characteristic wavelength (the linear M-S/Turing
// length), which collapses emergent shapes to a single scale (sphere,
// octahedron, hex). Forcing with multi-octave noise injects
// perturbations at several wavelengths simultaneously so branches can
// have sub-branches "for free", giving a fractal envelope rather than a
// clean smooth one.
float fbm3(vec3 p, float base_period, int octaves, int channel) {
    float amp  = 0.5;
    float freq = 1.0 / max(base_period, 1e-3);
    float sum  = 0.0;
    float norm = 0.0;
    for (int i = 0; i < octaves; ++i) {
        sum  += amp * value_noise3(p * freq, channel + i * 17);
        norm += amp;
        amp  *= 0.5;
        freq *= 2.0;
    }
    return sum / max(norm, 1e-6);
}

// TEMPORAL FBM: same as fbm3 but the lowest octave drifts in time, so the
// noise field evolves rather than being frozen. Each octave is given an
// independent time offset so they don't all march together (which would
// look like uniform translation rather than evolution). The drift speed
// per octave scales with the octave's frequency so finer detail flickers
// faster than coarse structure -- matches the way real diffusive noise
// has a power-law temporal spectrum.
float fbm3_temporal(vec3 p, float base_period, int octaves, int channel) {
    float amp  = 0.5;
    float freq = 1.0 / max(base_period, 1e-3);
    float sum  = 0.0;
    float norm = 0.0;
    float t    = float(u_frame) * 0.05;
    for (int i = 0; i < octaves; ++i) {
        // Per-octave time offset along an octave-specific axis so the
        // "wind direction" varies between scales (more organic motion).
        vec3 drift = vec3(float((i * 13) % 7), float((i * 19) % 5),
                          float((i * 23) % 11)) * t;
        sum  += amp * value_noise3(p * freq + drift, channel + i * 17);
        norm += amp;
        amp  *= 0.5;
        freq *= 2.0;
    }
    return sum / max(norm, 1e-6);
}

// ── Isotropic 19-point compact Laplacian (Patra-Karttunen / Mehrstellen) ──
//
// Standard 6-point stencil:   ∇²f ≈ (Σ_face f - 6f₀) / h²
//   Leading error:   (h²/12) · Σ_i ∂⁴f/∂xᵢ⁴   ← axis-aligned, anisotropic
//
// 19-point compact (this function):
//   ∇²f ≈ (⅓ Σ_face f + ⅙ Σ_edge f - 4f₀) / h²
//   Leading error:   (h²/12) · ∇²(∇²f)        ← rotationally symmetric, isotropic
//
// Visible consequence: BZ spirals stop being square-ish, Cahn-Hilliard
// droplets become spherical, wavefronts radiate evenly in all directions.
// Cost is ~3× memory traffic vs the 6-point fetch (18 vs 6 fetches).
// Coefficient sanity: 6·(1/3) + 12·(1/6) − 4 = 0 (constants give zero).
//
// Returns the per-component Laplacian (multiplied by h_sq so it equals ∇²f
// in continuum units, matching the 6-point convention used elsewhere).
vec4 lap19_v4(ivec3 pos, vec4 self_val) {
    vec4 face = fetch(pos + ivec3( 1, 0, 0)) + fetch(pos + ivec3(-1, 0, 0))
              + fetch(pos + ivec3( 0, 1, 0)) + fetch(pos + ivec3( 0,-1, 0))
              + fetch(pos + ivec3( 0, 0, 1)) + fetch(pos + ivec3( 0, 0,-1));
    vec4 edge = fetch(pos + ivec3( 1, 1, 0)) + fetch(pos + ivec3( 1,-1, 0))
              + fetch(pos + ivec3(-1, 1, 0)) + fetch(pos + ivec3(-1,-1, 0))
              + fetch(pos + ivec3( 1, 0, 1)) + fetch(pos + ivec3( 1, 0,-1))
              + fetch(pos + ivec3(-1, 0, 1)) + fetch(pos + ivec3(-1, 0,-1))
              + fetch(pos + ivec3( 0, 1, 1)) + fetch(pos + ivec3( 0, 1,-1))
              + fetch(pos + ivec3( 0,-1, 1)) + fetch(pos + ivec3( 0,-1,-1));
    return ((1.0/3.0) * face + (1.0/6.0) * edge - 4.0 * self_val) * h_sq;
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
