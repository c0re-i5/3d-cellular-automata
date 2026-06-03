
#version 430
#extension GL_NV_shader_atomic_float : enable
layout(local_size_x = 64) in;

layout(std430, binding = 9) buffer Particles { vec4 data[]; } P;
// Particle i occupies P.data[2*i + 0] = (pos, life)
//                      P.data[2*i + 1] = (vel, mass)

#ifndef NN_HID
#define NN_HID 0
#endif
#ifndef NN_SHARED
#define NN_SHARED 0
#endif

#ifndef DEPOSIT_ENABLED
#define DEPOSIT_ENABLED 0
#endif

#if DEPOSIT_ENABLED
// Each particle does an atomic float add into the (newly cleared)
// deposit SSBO at its post-update voxel.  Requires
// GL_NV_shader_atomic_float — fp32 atomics on SSBOs are NVIDIA-only,
// but every target device for this simulator (GeForce/Quadro from
// Maxwell onward) supports them.  We use an SSBO instead of an image
// because the driver only exposes 8 image units, all of which are
// already taken by field src/dst/extras bindings.
layout(std430, binding = 15) coherent buffer DepositBuf { float data[]; } u_deposit_buf;
uniform float u_deposit_amount;       // mass added per particle per step
uniform int   u_deposit_radius;       // 0 = single voxel, 1 = 3³ cube splat
uniform ivec3 u_dep_dims;             // (W,H,D) — needed for correct linear
                                      // indexing on non-cubic grids; u_size
                                      // is the back-compat scalar max(W,H,D).
#endif

#if NN_HID > 0
// Per-agent brain weight stride (floats):
//   W1: 6 * NN_HID, b1: NN_HID, W2: NN_HID * 3, b2: 3
//   total = 10 * NN_HID + 3
#define NN_STRIDE (10 * NN_HID + 3)
layout(std430, binding = 13) readonly buffer Brains { float w[]; } B;
#endif

uniform sampler3D u_field_tex;
uniform int   u_size;
uniform int   u_count;
uniform int   u_force_mode;   // 0=none, 1=velocity_rgb, 2=grad_r, 3=grad_neg_r, 4=curl_rgb, 5=nn_brain
uniform float u_force_scale;
uniform float u_dt;
uniform float u_drag;         // 0..1 fraction of velocity removed per step
uniform float u_life_decay;   // amount subtracted from life each step
uniform int   u_respawn;      // 0/1
uniform uint  u_frame;        // for respawn RNG
uniform vec3  u_spawn_min;
uniform vec3  u_spawn_max;
uniform float u_spawn_speed;  // initial random speed magnitude on respawn

// ── Particle Lenia (force mode 6) parameters ──────────────────────
// All dimensions are in cell units.  Matches Mordvintsev et al. 2022
// ("Particle Lenia and the Energy-Based Behaviour of Pattern
// Formation"): each particle feels a kernel-summed potential U from
// its neighbours and moves down the energy E(x) = R(x) - G(U(x)),
// where R is short-range repulsion and G is the bell-shaped growth
// function from standard Lenia.  Self-organising clusters, gliders,
// and metastable structures emerge from pure pairwise dynamics — no
// background field required.
uniform float u_pl_mu_k;     // kernel centre radius
uniform float u_pl_sigma_k;  // kernel ring thickness
uniform float u_pl_mu_g;     // growth-function centre (target U)
uniform float u_pl_sigma_g;  // growth-function width
uniform float u_pl_c_rep;    // repulsion strength
uniform float u_pl_r_rep;    // repulsion cutoff radius

float fs = float(u_size);

vec4 sampleField(vec3 p) {
    vec3 uv = (p + 0.5) / fs;
    return texture(u_field_tex, uv);
}

vec3 gradient_r(vec3 p) {
    float h = 1.0;
    float xp = sampleField(p + vec3(h,0,0)).r;
    float xm = sampleField(p - vec3(h,0,0)).r;
    float yp = sampleField(p + vec3(0,h,0)).r;
    float ym = sampleField(p - vec3(0,h,0)).r;
    float zp = sampleField(p + vec3(0,0,h)).r;
    float zm = sampleField(p - vec3(0,0,h)).r;
    return 0.5 * vec3(xp - xm, yp - ym, zp - zm);
}

vec3 curl_rgb(vec3 p) {
    float h = 1.0;
    vec3 xp = sampleField(p + vec3(h,0,0)).rgb;
    vec3 xm = sampleField(p - vec3(h,0,0)).rgb;
    vec3 yp = sampleField(p + vec3(0,h,0)).rgb;
    vec3 ym = sampleField(p - vec3(0,h,0)).rgb;
    vec3 zp = sampleField(p + vec3(0,0,h)).rgb;
    vec3 zm = sampleField(p - vec3(0,0,h)).rgb;
    vec3 dFdx = 0.5 * (xp - xm);
    vec3 dFdy = 0.5 * (yp - ym);
    vec3 dFdz = 0.5 * (zp - zm);
    return vec3(dFdy.z - dFdz.y, dFdz.x - dFdx.z, dFdx.y - dFdy.x);
}

#if NN_HID > 0
// Tiny MLP forward pass: 6 inputs → NN_HID hidden (tanh) → 3 outputs (tanh).
// `agent_id` selects per-agent weights (or 0 if NN_SHARED).
vec3 nn_eval(uint agent_id, vec3 grad, vec3 vel) {
    uint base = (NN_SHARED != 0) ? 0u : agent_id * uint(NN_STRIDE);
    float in_vec[6];
    in_vec[0] = grad.x; in_vec[1] = grad.y; in_vec[2] = grad.z;
    in_vec[3] = vel.x;  in_vec[4] = vel.y;  in_vec[5] = vel.z;
    float hid[NN_HID];
    // Layer 1: hid[h] = tanh(b1[h] + sum_i W1[i,h] * in[i])
    uint w1_base = base;                   // W1: 6 × NN_HID, row-major (i*H + h)
    uint b1_base = base + 6u * uint(NN_HID);
    for (int h = 0; h < NN_HID; ++h) {
        float s = B.w[b1_base + uint(h)];
        for (int i = 0; i < 6; ++i) {
            s += B.w[w1_base + uint(i) * uint(NN_HID) + uint(h)] * in_vec[i];
        }
        hid[h] = tanh(s);
    }
    // Layer 2: out[o] = tanh(b2[o] + sum_h W2[h,o] * hid[h])
    uint w2_base = b1_base + uint(NN_HID); // W2: NN_HID × 3, (h*3 + o)
    uint b2_base = w2_base + uint(NN_HID) * 3u;
    vec3 outv;
    for (int o = 0; o < 3; ++o) {
        float s = B.w[b2_base + uint(o)];
        for (int h = 0; h < NN_HID; ++h) {
            s += B.w[w2_base + uint(h) * 3u + uint(o)] * hid[h];
        }
        outv[o] = tanh(s);
    }
    return outv;
}
#endif

// Cheap hash → uniform [0,1) in 3 components.
vec3 hash3(uint i, uint salt) {
    uint h = i * 1664525u + salt * 1013904223u;
    h ^= (h >> 16); h *= 0x85ebca6bu;
    h ^= (h >> 13); h *= 0xc2b2ae35u;
    h ^= (h >> 16);
    uint h2 = h * 1597334677u + 0x9e3779b9u;
    uint h3 = h2 * 1597334677u + 0x9e3779b9u;
    return vec3(float(h & 0xffffffu), float(h2 & 0xffffffu),
                float(h3 & 0xffffffu)) / 16777215.0;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_count)) return;

    vec4 a = P.data[2u * i + 0u];   // pos.xyz, life
    vec4 b = P.data[2u * i + 1u];   // vel.xyz, mass
    vec3 pos = a.xyz;
    vec3 vel = b.xyz;
    float life = a.w;

    if (life <= 0.0) {
        if (u_respawn != 0) {
            vec3 r = hash3(i, u_frame);
            pos = mix(u_spawn_min, u_spawn_max, r);
            vec3 r2 = hash3(i, u_frame + 17u) - 0.5;
            vel = u_spawn_speed * r2;
            life = 1.0 + hash3(i, u_frame + 31u).x;
        } else {
            // dead and not respawning — write back unchanged
            P.data[2u * i + 0u] = vec4(pos, life);
            P.data[2u * i + 1u] = vec4(vel, b.w);
            return;
        }
    }

    // ── Force from field ──
    vec3 force = vec3(0.0);
    if (u_force_mode == 1) {
        vec3 v = sampleField(pos).rgb - 0.5;
        force = u_force_scale * v;
    } else if (u_force_mode == 2) {
        force = u_force_scale * gradient_r(pos);
    } else if (u_force_mode == 3) {
        force = -u_force_scale * gradient_r(pos);
    } else if (u_force_mode == 4) {
        force = u_force_scale * curl_rgb(pos);
    }
#if NN_HID > 0
    else if (u_force_mode == 5) {
        // NN brain: inputs = (gradient_r, vel) → outputs = steering force.
        vec3 g = gradient_r(pos);
        vec3 act = nn_eval(i, g, vel);
        force = u_force_scale * act;
    }
#endif
    else if (u_force_mode == 6) {
        // ── Particle Lenia (Mordvintsev et al., 2022) ──
        // Two N² passes over neighbours: pass 1 sums U at this point,
        // pass 2 sums ∇U and ∇R for the force.  Toroidal min-image
        // distances keep the dynamics consistent with the wrapped
        // particle positions.  Cutoffs prune work outside the kernel
        // and repulsion supports.
        float mu_k    = u_pl_mu_k;
        float sigma_k = max(u_pl_sigma_k, 0.001);
        float mu_g    = u_pl_mu_g;
        float sigma_g = max(u_pl_sigma_g, 0.001);
        float c_rep   = u_pl_c_rep;
        float r_rep   = max(u_pl_r_rep, 0.001);
        float k_cut   = mu_k + 4.0 * sigma_k;
        float k_cut2  = k_cut * k_cut;
        float r_rep2  = r_rep * r_rep;

        // Pass 1: kernel-summed potential at this particle.
        float U = 0.0;
        for (uint j = 0u; j < uint(u_count); ++j) {
            if (j == i) continue;
            vec3 pj = P.data[2u * j + 0u].xyz;
            vec3 d = pos - pj;
            // Toroidal min-image displacement (matches mod(pos,fs) wrap).
            d -= fs * round(d / fs);
            float r2 = dot(d, d);
            if (r2 > k_cut2) continue;
            float r = sqrt(r2);
            float z = (r - mu_k) / sigma_k;
            U += exp(-0.5 * z * z);
        }

        // G(U) and dG/dU:  G(U) = exp(-½ z²),  z = (U-μ_g)/σ_g
        //                 G'(U) = -G·z/σ_g
        float zg     = (U - mu_g) / sigma_g;
        float G_at   = exp(-0.5 * zg * zg);
        float Gprime = -G_at * zg / sigma_g;

        // Pass 2: ∇U (kernel) and ∇R (repulsion) at this particle.
        // ∂K(r)/∂x_i = -K·z/(σ_k·r) · d        with d = x_i - x_j
        // ∂R(r)/∂x_i = -2·c_rep·(1 - r/r_rep)/(r_rep·r) · d   for r<r_rep
        vec3 grad_U = vec3(0.0);
        vec3 grad_R = vec3(0.0);
        for (uint j = 0u; j < uint(u_count); ++j) {
            if (j == i) continue;
            vec3 pj = P.data[2u * j + 0u].xyz;
            vec3 d = pos - pj;
            d -= fs * round(d / fs);
            float r2 = dot(d, d);
            if (r2 > k_cut2 && r2 > r_rep2) continue;
            float r = sqrt(max(r2, 1e-8));
            if (r2 <= k_cut2) {
                float z = (r - mu_k) / sigma_k;
                float K = exp(-0.5 * z * z);
                grad_U += -K * z / (sigma_k * r) * d;
            }
            if (r < r_rep) {
                float t = 1.0 - r / r_rep;
                grad_R += -2.0 * c_rep * t / (r_rep * r) * d;
            }
        }

        // F = -∇E = -(∇R - G'·∇U) = G'·∇U - ∇R
        force = u_force_scale * (Gprime * grad_U - grad_R);
    }

    // Semi-implicit Euler with linear drag.
    vel = vel * (1.0 - u_drag) + force * u_dt;
    pos = pos + vel * u_dt;
    life -= u_life_decay;

    // Wrap (periodic) — keeps particles on screen for visual continuity.
    pos = mod(pos, fs);

    P.data[2u * i + 0u] = vec4(pos, life);
    P.data[2u * i + 1u] = vec4(vel, b.w);

#if DEPOSIT_ENABLED
    // Splat into the deposit field at the particle's new voxel.  We
    // round-to-nearest rather than floor so a particle at the centre
    // of a voxel deposits into that voxel (not the one to its lower
    // corner).  The deposit texture is cleared every step before this
    // pass, so we accumulate "this step's worth" only — no temporal
    // smearing here (the CA shader can integrate over time if it wants
    // a persistent trail).  Atomic add tolerates the rare race where
    // two particles land in the same voxel; without atomics we'd lose
    // ~N²/V deposits per step at high particle density.
    if (life > 0.0 && u_deposit_amount != 0.0) {
        ivec3 dp = ivec3(floor(pos + 0.5));
        dp = clamp(dp, ivec3(0), u_dep_dims - ivec3(1));
        if (u_deposit_radius == 0) {
            int idx = (dp.z * u_dep_dims.y + dp.y) * u_dep_dims.x + dp.x;
            atomicAdd(u_deposit_buf.data[idx], u_deposit_amount);
        } else {
            // 3³ Gaussian-ish splat with weights 1, 1/2, 1/4 by Chebyshev
            // distance.  Total energy ≈ 1 + 6·½ + 12·¼ + 8·¼ = 9 — caller
            // should scale u_deposit_amount accordingly if they want unit
            // total mass per particle.
            for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                ivec3 q = clamp(dp + ivec3(dx, dy, dz),
                                ivec3(0), u_dep_dims - ivec3(1));
                int cheby = max(max(abs(dx), abs(dy)), abs(dz));
                float w = (cheby == 0) ? 1.0 : (cheby == 1 ? 0.5 : 0.25);
                int idx = (q.z * u_dep_dims.y + q.y) * u_dep_dims.x + q.x;
                atomicAdd(u_deposit_buf.data[idx], u_deposit_amount * w);
            }
        }
    }
#endif
}
