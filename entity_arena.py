"""Entity Arena — GPU substrate for voxel-based agent simulations.

A reusable substrate that lets a CA preset declare a population of typed
entities ("kinds") that live alongside the voxel field, move continuously
through 3D space, see their neighbours via a spatial hash, and rasterize
themselves back into the field for the existing volumetric renderer.

DESIGN GOALS
------------
- One header, one set of SSBOs, one set of standard passes — every
  scenario reuses these and only writes the per-step `entity_step`
  shader (and optionally a custom paint shader).
- Hybrid control: GPU runs per-entity physics each frame; Python
  (CPU) handles spawning, despawning, scoring, goal updates, and any
  game logic that doesn't need to fit inside a parallel kernel.
- Voxel rendering throughout: entities paint themselves into the
  existing field channels so the volumetric raymarcher works
  unchanged.

SSBO BINDINGS (this module owns)
-------------------------------
  9  EntityBuf       array of Entity (80 B each)
 10  TeamBuf         array of Team (64 B each)
 11  GoalBuf         array of Goal (64 B each)
 12  HashCountBuf    uint per spatial cell (atomic-incremented)
 13  HashEntriesBuf  uint per (cell, slot) — entity ids

(Bindings 0,2,3,4,5,6,7,8 are already used by the simulator.)

ENTITY LAYOUT (std430, 80 bytes)
-------------------------------
  vec4  pos_radius        // .xyz=position in [0,size), .w=radius (cells)
  vec4  vel_energy        // .xyz=velocity (cells/step), .w=energy
  uvec4 kind_team_role_flags  // .x=kind, .y=team, .z=role, .w=flags
  uvec4 target_partner_timer_payload // .x=target_id .y=partner_id .z=timer .w=payload
  vec4  genome            // 4 free per-entity floats

Slot 0 of EntityBuf is RESERVED as the "null" / dead entity (kind==KIND_DEAD).

PASS KINDS (added to the simulator's pass executor)
--------------------------------------------------
  entity_clear_hash   — zeroes HashCountBuf
  entity_build_hash   — each live entity atomicAdds itself to its cell
  entity_step         — per-entity update (PER-SCENARIO shader)
  entity_paint        — rasterizes entities into voxel field
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SSBO bindings owned by this module.
BIND_ENTITIES   = 9
BIND_TEAMS      = 10
BIND_GOALS      = 11
BIND_HASH_COUNT = 12
BIND_HASH_ENTRY = 13

# Per-entity record size in bytes (5 vec4s).
ENTITY_BYTES = 80
TEAM_BYTES   = 64
GOAL_BYTES   = 64

# Default capacity hard caps (override per-preset in 'entity_arena' dict).
DEFAULT_MAX_ENTITIES = 65536
DEFAULT_MAX_TEAMS    = 8
DEFAULT_MAX_GOALS    = 64

# Spatial hash: cube cell size in voxel-cells. Must divide u_size in the
# common case (we wrap, so non-divisors still work but waste a bit of cells).
DEFAULT_HASH_CELL = 8
HASH_MAX_PER_CELL = 32  # max entities per spatial cell — overflow drops

# Reserved kind ids:
KIND_DEAD = 0  # slot is empty / available for reuse


# ---------------------------------------------------------------------------
# GLSL header — included in every entity_* shader
# ---------------------------------------------------------------------------

# This header defines the structs, the binding points, and helpers for
# spatial hashing. It is concatenated AFTER `#version 430\nlayout(...) in;`
# so callers must declare local_size first.
ENTITY_GLSL_HEADER = """
// ----- Entity Arena: shared bindings & helpers --------------------------

struct Entity {
    vec4  pos_radius;        // xyz=pos, w=radius
    vec4  vel_energy;        // xyz=vel, w=energy
    uvec4 kind_team_role_flags;
    uvec4 target_partner_timer_payload;
    vec4  genome;
};

struct Team {
    vec4  color;                   // rgb + a (unused / opacity)
    vec4  spawn_pos_radius;        // xyz center, w radius
    uvec4 score_alive_kills_flags; // x=score y=alive_count z=kills w=flags
    uvec4 target_kind_count_pad;   // x=target_kind y=score_target z=pad w=pad
};

struct Goal {
    vec4  pos_radius;            // xyz center, w radius
    uvec4 kind_team_required_count; // x=goal_kind y=team z=required_kind w=required_count
    uvec4 progress_flags_pad_pad;   // x=progress y=flags
    uvec4 _pad;
};

layout(std430, binding=9)  buffer EntityBuf    { Entity entities[]; };
layout(std430, binding=10) buffer TeamBuf      { Team   teams[];    };
layout(std430, binding=11) buffer GoalBuf      { Goal   goals[];    };
layout(std430, binding=12) buffer HashCountBuf { uint   hash_count[]; };
layout(std430, binding=13) buffer HashEntryBuf { uint   hash_entry[]; };

// Filled by host on every dispatch.
uniform int u_size;              // voxel grid edge (replaced w/ const at compile)
uniform float u_dt;
uniform int u_frame;
uniform int u_pass;
uniform int u_boundary;
uniform float u_param0;
uniform float u_param1;
uniform float u_param2;
uniform float u_param3;
uniform int  u_entity_count;     // total slots (live + dead)
uniform int  u_team_count;
uniform int  u_goal_count;
uniform int  u_hash_cell;        // edge length in voxels of each hash cell
uniform int  u_hash_dim;         // grid is (u_hash_dim)^3 cells
uniform int  u_hash_max_per_cell;
uniform float u_world_size;      // == float(u_size); duplicated for clarity

// Spatial hash helpers --------------------------------------------------
ivec3 hash_cell_of(vec3 pos) {
    // pos is in [0, u_world_size); wrap defensively in case scenarios drift.
    vec3 wrapped = mod(pos, u_world_size);
    ivec3 c = ivec3(floor(wrapped / float(u_hash_cell)));
    c = ((c % u_hash_dim) + u_hash_dim) % u_hash_dim;
    return c;
}

uint hash_cell_index(ivec3 c) {
    return uint(c.x + c.y * u_hash_dim + c.z * u_hash_dim * u_hash_dim);
}

// Iterate neighbours within radius `r` cells around `pos`. Caller passes
// a max neighbour-count cap to keep loops bounded. Visits each candidate
// AT MOST ONCE per hash cell; further radius checks are caller's job.
//
// Usage pattern:
//   uint nb_ids[64]; int nb_n = 0;
//   gather_neighbours(pos, 8.0, nb_ids, nb_n, 64);
//   for (int i=0;i<nb_n;i++) { Entity o = entities[nb_ids[i]]; ... }
void gather_neighbours(vec3 pos, float radius, inout uint out_ids[64],
                       inout int out_count, int cap) {
    int span = int(ceil(radius / float(u_hash_cell)));
    ivec3 base = hash_cell_of(pos);
    out_count = 0;
    for (int dz = -span; dz <= span; ++dz)
    for (int dy = -span; dy <= span; ++dy)
    for (int dx = -span; dx <= span; ++dx) {
        ivec3 c = base + ivec3(dx, dy, dz);
        c = ((c % u_hash_dim) + u_hash_dim) % u_hash_dim;
        uint ci = hash_cell_index(c);
        uint n = min(hash_count[ci], uint(u_hash_max_per_cell));
        for (uint k = 0u; k < n; ++k) {
            if (out_count >= cap) return;
            uint id = hash_entry[ci * uint(u_hash_max_per_cell) + k];
            out_ids[out_count] = id;
            out_count += 1;
        }
    }
}

// Distance with periodic wrap on a u_world_size cube.
vec3 wrap_delta(vec3 a, vec3 b) {
    vec3 d = a - b;
    d -= u_world_size * round(d / u_world_size);
    return d;
}

// True if entity slot is occupied (kind != 0).
bool is_alive(Entity e) { return e.kind_team_role_flags.x != 0u; }
"""


# ---------------------------------------------------------------------------
# Standard pre-built shaders (bodies — caller wraps with header)
# ---------------------------------------------------------------------------

# Per-entity workgroup of 64. Used by clear_hash, build_hash, step, and
# any per-entity scenario shader.
ENTITY_LAYOUT_HEADER = "#version 430\nlayout(local_size_x=64, local_size_y=1, local_size_z=1) in;\n"

# Per-spatial-cell workgroup of 64 (used by clear_hash).
HASH_LAYOUT_HEADER = "#version 430\nlayout(local_size_x=64, local_size_y=1, local_size_z=1) in;\n"


SHADER_CLEAR_HASH = """
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= hash_count.length()) return;
    hash_count[i] = 0u;
}
"""

SHADER_BUILD_HASH = """
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_entity_count)) return;
    Entity e = entities[i];
    if (!is_alive(e)) return;
    ivec3 c = hash_cell_of(e.pos_radius.xyz);
    uint ci = hash_cell_index(c);
    uint slot = atomicAdd(hash_count[ci], 1u);
    if (slot < uint(u_hash_max_per_cell)) {
        hash_entry[ci * uint(u_hash_max_per_cell) + slot] = i;
    }
    // overflow drops silently (cell saturated; behavior is approximate).
}
"""

# Default rasterizer: paints each entity as a sphere of radius
# `pos_radius.w`.
#
# Channel layout — designed to play nicely with the simulator's
# single-channel volumetric renderer:
#   R (ch0) — SIGNED team mass: team 0 paints +w, team 1 paints -w
#             (set vis_abs=2 in the preset to map [-1,+1]->[0,1] and
#              get a diverging team-vs-team colormap visualisation).
#   G (ch1) — team 0 mass (positive only) — for single-team views.
#   B (ch2) — team 1 mass (positive only) — for single-team views.
#   A (ch3) — total density (any team) — useful for alpha / iso.
#
# DETERMINISM (Bug O fix, May 2026):
# Original implementation dispatched per entity and did a non-atomic RMW
# (imageLoad → max() → imageStore). When two entities painted overlapping
# voxels the read could miss a concurrent write, producing run-to-run
# variation on the GPU.
#
# This shader is now dispatched per VOXEL (size^3 threads). Each voxel
# computes its own max-blend by iterating over the entities in its own
# hash cell + the 26 neighbour cells (entities can have radius ≤ 4 and
# hash_cell defaults to 8, so a 1-cell span covers every possible
# contributor). Since max() is associative & commutative, the iteration
# order inside each voxel does not matter — the result is deterministic
# regardless of which entities the hash builder happened to slot first.
#
# The dispatch sizing change (per-entity → per-voxel) is handled by the
# simulator's entity_paint kind branch; see simulator.py search for
# kind == 'entity_paint'.
SHADER_PAINT = """
layout(rgba32f, binding=0) uniform image3D u_grid_r;
layout(rgba32f, binding=1) uniform image3D u_grid_w;

void main() {
    uint flat_idx = gl_GlobalInvocationID.x;
    uint total = uint(u_size) * uint(u_size) * uint(u_size);
    if (flat_idx >= total) return;

    int S = u_size;
    int idx = int(flat_idx);
    int z = idx / (S * S);
    int rem = idx - z * S * S;
    int y = rem / S;
    int x = rem - y * S;
    ivec3 p = ivec3(x, y, z);

    // Start from the read-side value so we preserve anything the
    // pre-paint passes (clear, eco field, etc.) wrote.
    vec4 result = imageLoad(u_grid_r, p);

    // Hash cell containing this voxel. Span ±1 cell — sufficient because
    // entity radius is clamped to 4 and the hash cell edge is u_hash_cell
    // (default 8). If a future scenario sets either larger this span
    // would need to grow.
    ivec3 base = ivec3(p.x / u_hash_cell, p.y / u_hash_cell, p.z / u_hash_cell);
    int half_s = S / 2;

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        ivec3 c = base + ivec3(dx, dy, dz);
        c = ((c % u_hash_dim) + u_hash_dim) % u_hash_dim;
        uint ci = hash_cell_index(c);
        uint n = min(hash_count[ci], uint(u_hash_max_per_cell));
        for (uint k = 0u; k < n; ++k) {
            uint eid = hash_entry[ci * uint(u_hash_max_per_cell) + k];
            Entity e = entities[eid];
            if (!is_alive(e)) continue;

            // Signed wrap of (p - center) onto [-S/2, S/2).
            ivec3 center = ivec3(floor(e.pos_radius.xyz));
            ivec3 off = p - center;
            off.x = ((off.x % S) + S) % S; if (off.x > half_s) off.x -= S;
            off.y = ((off.y % S) + S) % S; if (off.y > half_s) off.y -= S;
            off.z = ((off.z % S) + S) % S; if (off.z > half_s) off.z -= S;

            int rmax = int(ceil(e.pos_radius.w));
            rmax = clamp(rmax, 0, 4);
            if (abs(off.x) > rmax || abs(off.y) > rmax || abs(off.z) > rmax)
                continue;

            float dist = length(vec3(off));
            if (dist > e.pos_radius.w) continue;
            float inv_r = (e.pos_radius.w > 0.001) ? 1.0 / e.pos_radius.w : 1.0;
            float w = max(0.0, 1.0 - dist * inv_r);

            uint team_id = e.kind_team_role_flags.y;
            float signed_sign = (team_id == 0u) ? 1.0
                              : (team_id == 1u) ? -1.0
                              : 0.0;

            // Saturating max-blend per channel — deterministic because
            // this voxel is owned by a single thread.
            if (signed_sign > 0.0) {
                result.r = max(result.r, signed_sign * w);
            } else if (signed_sign < 0.0) {
                result.r = min(result.r, signed_sign * w);
            }
            if (team_id == 0u) result.g = max(result.g, w);
            if (team_id == 1u) result.b = max(result.b, w);
            result.a = max(result.a, w);
        }
    }

    imageStore(u_grid_w, p, result);
}
"""

# Optional: clear the entity-paint channels at start of a frame. Many
# scenarios will want this so painted entities don't leave smears. Uses
# 1D flat indexing over the (u_size^3) voxel grid so it shares the same
# local_size_x=64 layout as every other entity pass.
SHADER_PAINT_CLEAR = """
layout(rgba32f, binding=0) uniform image3D u_grid_r;
layout(rgba32f, binding=1) uniform image3D u_grid_w;

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint total = uint(u_size) * uint(u_size) * uint(u_size);
    if (i >= total) return;
    int sz = u_size;
    int x = int(i % uint(sz));
    int y = int((i / uint(sz)) % uint(sz));
    int z = int(i / uint(sz * sz));
    imageStore(u_grid_w, ivec3(x, y, z), vec4(0.0));
}
"""


# ---------------------------------------------------------------------------
# Validation scenario: wandering voxels (proves substrate works end-to-end)
# ---------------------------------------------------------------------------

# Minimal entity_step: each entity does brownian motion + a tiny pull
# towards its team's spawn center. No interaction yet. Used to validate
# the substrate is wired up correctly.
SHADER_WANDERING_STEP = """
// Cheap hash for per-step jitter.
float h11(uint x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15);
    return float(x & 0x00FFFFFFu) / float(0x01000000u);
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_entity_count)) return;
    Entity e = entities[i];
    if (!is_alive(e)) return;

    uint seed = i * 1973u + uint(u_frame) * 9277u;
    vec3 jitter = vec3(h11(seed), h11(seed + 1u), h11(seed + 2u)) - 0.5;

    // Velocity = brownian + spring-like attraction to team spawn center.
    // Use linear (NOT normalized) restoring force so distant entities
    // feel a stronger pull than nearby ones — gives a stable cloud
    // around home instead of unbounded diffusion.
    uint tid = e.kind_team_role_flags.y;
    vec3 home = (tid < uint(u_team_count))
        ? teams[tid].spawn_pos_radius.xyz
        : vec3(u_world_size * 0.5);
    vec3 toward = wrap_delta(home, e.pos_radius.xyz);
    // Spring constant: param1 is the user-facing "Pull" knob. Scale by
    // 1/world_size so the same value behaves consistently across grid
    // sizes (force at the far side of the cube ≈ Pull * 0.5).
    vec3 pull = toward * (u_param1 / u_world_size);

    vec3 v = e.vel_energy.xyz * 0.85 + jitter * u_param0 + pull;
    vec3 p = e.pos_radius.xyz + v * u_dt;
    p = mod(p, u_world_size);

    e.pos_radius.xyz = p;
    e.vel_energy.xyz = v;
    entities[i] = e;
}
"""


# ---------------------------------------------------------------------------
# Predator-Prey ecosystem v1 — full-scale agent-based simulation
# ---------------------------------------------------------------------------
#
# Field channel layout (rgba32f voxel grid):
#   ch0 (R) — composite display channel, signed bipolar:
#               +0.30 * food (background)  +  +1.0 prey  +  -1.0 predator
#               Use vis_abs=2 + colormap=6 (Diverging) for blue/red view.
#   ch1 (G) — prey density (0..1) — for "prey only" channel toggle
#   ch2 (B) — predator density (0..1) — for "predator only" toggle
#   ch3 (A) — FOOD field (0..1) — persists across frames; prey graze it,
#             logistic regrowth rebuilds it. THE FIELD'S MEMORY.
#
# Pipeline (passes per frame):
#   1. eco_field_update    (entity_field) — regrow ch3 food + diffuse +
#                                            recompose ch0 background +
#                                            clear ch1/ch2
#   2. entity_clear_hash
#   3. entity_build_hash
#   4. prey_step           (entity_step)  — wander + graze ch3 + age + die
#   5. predator_step       (entity_step)  — hunt prey via hash + eat + die
#   6. eco_paint           (entity_paint) — additive blend prey/predator
#                                            into ch0/ch1/ch2
#
# Entity kind ids (caller's choice; convention used by these shaders):
#   1 = PREY      (team 0)   genome.x = max_speed
#                            vel_energy.w = energy
#   2 = PREDATOR  (team 1)   genome.x = max_speed
#                            genome.y = sight_radius
#                            vel_energy.w = energy
#
# CPU on_tick handles reproduction (entity with energy > threshold spawns
# child with mutated genome; halves parent energy) and population stats.

# ── Shader 1 of 4: food field update (3D voxel pass) ────────────────────
SHADER_FOOD_FIELD_UPDATE = """
// 3D voxel pass — uses standard COMPUTE_HEADER (u_src/u_dst, u_param0..3).
// Reads previous field from u_src, writes new field to u_dst.
//
// Params:
//   u_param0 — food regrow rate (logistic dr/dt = r * f * (1-f))
//   u_param1 — food diffusion rate (per step, fraction blended with neighbours)
//   u_param2 — UNUSED here (other passes consume it)
//   u_param3 — UNUSED here

void main() {
    ivec3 p = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(p, ivec3(u_size)))) return;

    vec4 c = imageLoad(u_src, p);
    float food = c.a;

    // Diffuse: 6-neighbour box blend (toroidal wrap).
    int sz = u_size;
    float n = 0.0;
    n += imageLoad(u_src, ivec3((p.x+1) % sz, p.y, p.z)).a;
    n += imageLoad(u_src, ivec3((p.x+sz-1) % sz, p.y, p.z)).a;
    n += imageLoad(u_src, ivec3(p.x, (p.y+1) % sz, p.z)).a;
    n += imageLoad(u_src, ivec3(p.x, (p.y+sz-1) % sz, p.z)).a;
    n += imageLoad(u_src, ivec3(p.x, p.y, (p.z+1) % sz)).a;
    n += imageLoad(u_src, ivec3(p.x, p.y, (p.z+sz-1) % sz)).a;
    food = mix(food, n / 6.0, clamp(u_param1 * u_dt, 0.0, 1.0));

    // Logistic regrowth.
    food += u_param0 * food * (1.0 - food) * u_dt;
    food = clamp(food, 0.0, 1.0);

    // Composite background for ch0 display channel: faint warm tint
    // proportional to food. Entities will overwrite this with stronger
    // values via max/min blend in the paint pass.
    float bg = food * 0.30;

    // Write the updated field. Clear prey/predator density channels —
    // they get rebuilt every frame by the paint pass.
    imageStore(u_dst, p, vec4(bg, 0.0, 0.0, food));
}
"""


# ── Shader 2 of 4: prey step (entity_step, read+write field) ───────────
SHADER_PREY_STEP = """
// Per-entity prey update. Bound to image unit 0 read+write so we can
// graze food from ch3.
//
// Active iff this entity is alive AND kind == 1 (PREY). We don't have a
// dispatch-level filter so each invocation re-checks.
//
// Params (passed via u_param0..3):
//   u_param0 — prey wander jitter scale (random-walk noise)
//   u_param1 — prey graze rate (energy/food per step)
//   u_param2 — UNUSED
//   u_param3 — global metabolism multiplier (energy decay/step)
//
// genome layout (per entity):
//   genome.x — max_speed (cells/step)

layout(rgba32f, binding=0) uniform image3D u_field;

float h11(uint x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15);
    return float(x & 0x00FFFFFFu) / float(0x01000000u);
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_entity_count)) return;
    Entity e = entities[i];
    if (!is_alive(e)) return;
    if (e.kind_team_role_flags.x != 1u) return;  // not prey

    uint seed = i * 1973u + uint(u_frame) * 9277u;
    vec3 jitter = vec3(h11(seed), h11(seed + 1u), h11(seed + 2u)) - 0.5;

    // Wander velocity — bounded random walk with slight inertia.
    float max_speed = max(e.genome.x, 0.01);
    vec3 v = e.vel_energy.xyz * 0.85 + jitter * u_param0;
    float vlen = length(v);
    if (vlen > max_speed) v *= max_speed / vlen;

    vec3 pos = mod(e.pos_radius.xyz + v * u_dt, u_world_size);

    // Graze: read food at current voxel, take a bite, write it back.
    ivec3 vp = ivec3(floor(pos));
    vp = ((vp % u_size) + u_size) % u_size;
    vec4 vc = imageLoad(u_field, vp);
    float food = vc.a;
    float bite = min(food, u_param1 * u_dt);
    vc.a = food - bite;
    imageStore(u_field, vp, vc);

    // Energy: gain from grazing, lose to metabolism.
    float energy = e.vel_energy.w + bite - u_param3 * u_dt;

    if (energy <= 0.0) {
        // Starve: mark dead. CPU will reclaim slot on next pull.
        e.kind_team_role_flags.x = 0u;
        entities[i] = e;
        return;
    }

    // Cap energy so reproduction doesn't go infinite without CPU lag.
    energy = min(energy, 3.0);

    e.pos_radius.xyz = pos;
    e.vel_energy.xyz = v;
    e.vel_energy.w   = energy;
    entities[i] = e;
}
"""


# ── Shader 3 of 4: predator step (entity_step, hunts prey) ──────────────
SHADER_PREDATOR_STEP = """
// Per-entity predator update. Bound to image unit 0 read+write (we don't
// write the field but binding is shared with prey step).
//
// Active iff entity is alive AND kind == 2 (PREDATOR).
//
// Params:
//   u_param0 — predator wander jitter (when no prey in sight)
//   u_param1 — UNUSED here (prey graze rate)
//   u_param2 — predator sight radius scale (multiplies entity.genome.y)
//   u_param3 — global metabolism multiplier

layout(rgba32f, binding=0) uniform image3D u_field;

float h11(uint x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15);
    return float(x & 0x00FFFFFFu) / float(0x01000000u);
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_entity_count)) return;
    Entity e = entities[i];
    if (!is_alive(e)) return;
    if (e.kind_team_role_flags.x != 2u) return;  // not predator

    float max_speed = max(e.genome.x, 0.01);
    float sight     = max(e.genome.y, 1.0) * max(u_param2, 0.01);

    // Find nearest live prey within sight via spatial hash.
    uint nb_ids[64]; int nb_n = 0;
    gather_neighbours(e.pos_radius.xyz, sight, nb_ids, nb_n, 64);

    int best = -1;
    float best_d2 = sight * sight;
    for (int k = 0; k < nb_n; ++k) {
        uint id = nb_ids[k];
        if (id == i) continue;
        Entity o = entities[id];
        if (o.kind_team_role_flags.x != 1u) continue;  // only chase prey
        vec3 d = wrap_delta(o.pos_radius.xyz, e.pos_radius.xyz);
        float d2 = dot(d, d);
        if (d2 < best_d2) {
            best = int(id);
            best_d2 = d2;
        }
    }

    vec3 v;
    if (best >= 0) {
        // Pursue: steer toward prey at max_speed.
        Entity prey = entities[best];
        vec3 toward = wrap_delta(prey.pos_radius.xyz, e.pos_radius.xyz);
        float dist = length(toward) + 1e-5;
        v = (toward / dist) * max_speed;

        // Eat if close enough (within 1.5 cells). Use atomicCompSwap on the
        // prey's kind to ensure only one predator wins.
        if (dist < 1.5) {
            uint won = atomicCompSwap(entities[best].kind_team_role_flags.x, 1u, 0u);
            if (won == 1u) {
                e.vel_energy.w += 1.2;  // gulp — must cover several steps of metabolism
            }
        }
    } else {
        // Wander.
        uint seed = i * 4079u + uint(u_frame) * 7561u;
        vec3 jitter = vec3(h11(seed), h11(seed + 1u), h11(seed + 2u)) - 0.5;
        v = e.vel_energy.xyz * 0.9 + jitter * u_param0;
        float vlen = length(v);
        if (vlen > max_speed) v *= max_speed / vlen;
    }

    vec3 pos = mod(e.pos_radius.xyz + v * u_dt, u_world_size);

    // Predators have slightly higher metabolism than prey, but not so
    // high that a single bad search kills them.
    float energy = e.vel_energy.w - u_param3 * 1.1 * u_dt;
    if (energy <= 0.0) {
        e.kind_team_role_flags.x = 0u;
        entities[i] = e;
        return;
    }
    energy = min(energy, 3.0);

    e.pos_radius.xyz = pos;
    e.vel_energy.xyz = v;
    e.vel_energy.w   = energy;
    entities[i] = e;
}
"""


# ── Shader 4 of 4: ecosystem paint (preserves food background) ──────────
SHADER_ECO_PAINT = """
// Per-voxel variant of SHADER_PAINT for the predator/prey scenario:
//   - blends prey (kind=1) as POSITIVE additive into ch0 (max blend)
//   - blends predator (kind=2) as NEGATIVE additive into ch0 (min blend)
//   - accumulates per-species density into ch1 (prey) / ch2 (predator)
//   - DOES NOT touch ch3 (food field — preserved across the paint pass)
//
// See SHADER_PAINT for the per-voxel determinism rationale.
layout(rgba32f, binding=0) uniform image3D u_grid_r;
layout(rgba32f, binding=1) uniform image3D u_grid_w;

void main() {
    uint flat_idx = gl_GlobalInvocationID.x;
    uint total = uint(u_size) * uint(u_size) * uint(u_size);
    if (flat_idx >= total) return;

    int S = u_size;
    int idx = int(flat_idx);
    int z = idx / (S * S);
    int rem = idx - z * S * S;
    int y = rem / S;
    int x = rem - y * S;
    ivec3 p = ivec3(x, y, z);

    vec4 result = imageLoad(u_grid_r, p);
    // Ch3 (food) is explicitly preserved at write time below.
    float preserved_food = result.a;

    ivec3 base = ivec3(p.x / u_hash_cell, p.y / u_hash_cell, p.z / u_hash_cell);
    int half_s = S / 2;

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        ivec3 c = base + ivec3(dx, dy, dz);
        c = ((c % u_hash_dim) + u_hash_dim) % u_hash_dim;
        uint ci = hash_cell_index(c);
        uint n = min(hash_count[ci], uint(u_hash_max_per_cell));
        for (uint k = 0u; k < n; ++k) {
            uint eid = hash_entry[ci * uint(u_hash_max_per_cell) + k];
            Entity e = entities[eid];
            if (!is_alive(e)) continue;

            uint kind = e.kind_team_role_flags.x;
            float signed_w_sign = (kind == 1u) ? +1.0
                                : (kind == 2u) ? -1.0
                                : 0.0;
            if (signed_w_sign == 0.0) continue;

            ivec3 center = ivec3(floor(e.pos_radius.xyz));
            ivec3 off = p - center;
            off.x = ((off.x % S) + S) % S; if (off.x > half_s) off.x -= S;
            off.y = ((off.y % S) + S) % S; if (off.y > half_s) off.y -= S;
            off.z = ((off.z % S) + S) % S; if (off.z > half_s) off.z -= S;

            int rmax = int(ceil(e.pos_radius.w));
            rmax = clamp(rmax, 0, 4);
            if (abs(off.x) > rmax || abs(off.y) > rmax || abs(off.z) > rmax)
                continue;

            float dist = length(vec3(off));
            if (dist > e.pos_radius.w) continue;
            float inv_r = (e.pos_radius.w > 0.001) ? 1.0 / e.pos_radius.w : 1.0;
            float w = max(0.0, 1.0 - dist * inv_r);

            // ch0: prey adds positively (max), predator subtracts (min).
            if (signed_w_sign > 0.0) {
                result.r = max(result.r,  w);
            } else {
                result.r = min(result.r, -w);
            }
            if (kind == 1u) result.g = max(result.g, w);
            if (kind == 2u) result.b = max(result.b, w);
        }
    }

    imageStore(u_grid_w, p, vec4(result.r, result.g, result.b, preserved_food));
}
"""


# ---------------------------------------------------------------------------
# CPU-side container
# ---------------------------------------------------------------------------

class EntityArena:
    """Holds the CPU-side staging arrays for entities/teams/goals plus
    helpers to spawn, despawn, push to GPU.

    Lifecycle:
      arena = EntityArena(ctx, size, max_entities=...)
      arena.alloc_gpu()          # creates SSBOs
      arena.spawn(kind=..., team=..., pos=..., ...)
      arena.push_entities()      # before GPU step
      ... GPU step runs ...
      arena.pull_entities()      # if CPU needs to read back
      arena.release()
    """

    # numpy dtype matching ENTITY_BYTES layout (std430-safe: vec4 only).
    ENTITY_DTYPE = np.dtype([
        ('pos_radius',  np.float32, 4),
        ('vel_energy',  np.float32, 4),
        ('ktrf',        np.uint32,  4),  # kind, team, role, flags
        ('tptp',        np.uint32,  4),  # target, partner, timer, payload
        ('genome',      np.float32, 4),
    ])
    assert ENTITY_DTYPE.itemsize == ENTITY_BYTES, \
        f"Entity dtype is {ENTITY_DTYPE.itemsize}B, expected {ENTITY_BYTES}B"

    TEAM_DTYPE = np.dtype([
        ('color',       np.float32, 4),
        ('spawn_pr',    np.float32, 4),
        ('saskf',       np.uint32,  4),
        ('tkc',         np.uint32,  4),
    ])
    assert TEAM_DTYPE.itemsize == TEAM_BYTES

    GOAL_DTYPE = np.dtype([
        ('pos_radius',  np.float32, 4),
        ('ktrc',        np.uint32,  4),
        ('progress',    np.uint32,  4),
        ('_pad',        np.uint32,  4),
    ])
    assert GOAL_DTYPE.itemsize == GOAL_BYTES

    def __init__(self, ctx, size,
                 max_entities=DEFAULT_MAX_ENTITIES,
                 max_teams=DEFAULT_MAX_TEAMS,
                 max_goals=DEFAULT_MAX_GOALS,
                 hash_cell=DEFAULT_HASH_CELL):
        self.ctx = ctx
        self.size = int(size)
        self.max_entities = int(max_entities)
        self.max_teams = int(max_teams)
        self.max_goals = int(max_goals)
        self.hash_cell = int(hash_cell)
        # hash dim: number of cells per axis; ceil so we cover the world.
        self.hash_dim = max(1, (self.size + self.hash_cell - 1) // self.hash_cell)
        self.hash_total = self.hash_dim ** 3

        # Staging arrays. Slot 0 reserved as null entity.
        self.entities = np.zeros(self.max_entities, dtype=self.ENTITY_DTYPE)
        self.teams    = np.zeros(self.max_teams,    dtype=self.TEAM_DTYPE)
        self.goals    = np.zeros(self.max_goals,    dtype=self.GOAL_DTYPE)
        # Default white color so paint isn't black if scenario forgets.
        self.teams['color'][:] = (1.0, 1.0, 1.0, 1.0)

        # Free-slot stack (LIFO). Slot 0 is reserved → never freed.
        self._free_slots = list(range(self.max_entities - 1, 0, -1))

        # GPU resources (allocated lazily).
        self.entity_ssbo = None
        self.team_ssbo   = None
        self.goal_ssbo   = None
        self.hash_count_ssbo = None
        self.hash_entry_ssbo = None

        # Set when first team/goal is created so push only sends in-use slots.
        self.team_count = 0
        self.goal_count = 0

    # -- GPU resource lifecycle --------------------------------------------

    def alloc_gpu(self):
        if self.entity_ssbo is not None:
            return
        self.entity_ssbo = self.ctx.buffer(data=self.entities.tobytes())
        self.team_ssbo   = self.ctx.buffer(data=self.teams.tobytes())
        self.goal_ssbo   = self.ctx.buffer(data=self.goals.tobytes())
        # Hash buffers: zeros at allocation; rebuilt every step.
        self.hash_count_ssbo = self.ctx.buffer(
            data=np.zeros(self.hash_total, dtype=np.uint32).tobytes())
        self.hash_entry_ssbo = self.ctx.buffer(
            data=np.zeros(self.hash_total * HASH_MAX_PER_CELL,
                          dtype=np.uint32).tobytes())

    def release(self):
        for attr in ('entity_ssbo', 'team_ssbo', 'goal_ssbo',
                     'hash_count_ssbo', 'hash_entry_ssbo'):
            buf = getattr(self, attr, None)
            if buf is not None:
                try:
                    buf.release()
                except Exception:  # noqa: BLE001  GL resource release, never fatal
                    pass
                setattr(self, attr, None)

    def bind_all(self):
        """Bind all owned SSBOs to their canonical slots. Call before any
        entity_* dispatch."""
        self.entity_ssbo.bind_to_storage_buffer(BIND_ENTITIES)
        self.team_ssbo.bind_to_storage_buffer(BIND_TEAMS)
        self.goal_ssbo.bind_to_storage_buffer(BIND_GOALS)
        self.hash_count_ssbo.bind_to_storage_buffer(BIND_HASH_COUNT)
        self.hash_entry_ssbo.bind_to_storage_buffer(BIND_HASH_ENTRY)

    def set_uniforms(self, prog):
        """Set entity-arena uniforms on a compute program if they exist."""
        for name, val in (
            ('u_entity_count',      self.max_entities),
            ('u_team_count',        self.team_count),
            ('u_goal_count',        self.goal_count),
            ('u_hash_cell',         self.hash_cell),
            ('u_hash_dim',          self.hash_dim),
            ('u_hash_max_per_cell', HASH_MAX_PER_CELL),
            ('u_world_size',        float(self.size)),
        ):
            cu = prog.get(name, None)
            if cu is not None:
                cu.value = val

    # -- Spawn / despawn ---------------------------------------------------

    def spawn(self, kind, team=0, pos=(0, 0, 0), vel=(0, 0, 0),
              radius=1.0, energy=1.0, role=0, flags=0,
              target=0, partner=0, timer=0, payload=0,
              genome=(0, 0, 0, 0)):
        """Allocate a slot and write the entity. Returns slot id, or -1
        if the arena is full. `kind` MUST be != 0 (0 is the dead sentinel)."""
        if kind == 0:
            raise ValueError("kind=0 is reserved for the dead/null sentinel")
        if not self._free_slots:
            return -1
        idx = self._free_slots.pop()
        e = self.entities[idx]
        e['pos_radius'][:3] = pos
        e['pos_radius'][3]  = radius
        e['vel_energy'][:3] = vel
        e['vel_energy'][3]  = energy
        e['ktrf'] = (kind, team, role, flags)
        e['tptp'] = (target, partner, timer, payload)
        e['genome'] = genome
        return idx

    def despawn(self, idx):
        """Mark slot dead and return it to the free list."""
        if idx <= 0 or idx >= self.max_entities:
            return
        self.entities[idx] = 0  # zero everything → kind=0 → dead
        self._free_slots.append(idx)

    def alive_count(self):
        return int(np.count_nonzero(self.entities['ktrf'][:, 0]))

    # -- Team / goal config ------------------------------------------------

    def set_team(self, idx, color=(1, 1, 1, 1), spawn_pos=(0, 0, 0),
                 spawn_radius=4.0, target_kind=0, score_target=0):
        if idx >= self.max_teams:
            raise IndexError(f"team idx {idx} >= max_teams {self.max_teams}")
        t = self.teams[idx]
        t['color'] = color
        t['spawn_pr'][:3] = spawn_pos
        t['spawn_pr'][3]  = spawn_radius
        t['saskf'] = (0, 0, 0, 0)
        t['tkc']   = (target_kind, score_target, 0, 0)
        self.team_count = max(self.team_count, idx + 1)

    def add_goal(self, pos, radius, kind=1, team=0,
                 required_kind=0, required_count=1):
        if self.goal_count >= self.max_goals:
            return -1
        idx = self.goal_count
        g = self.goals[idx]
        g['pos_radius'][:3] = pos
        g['pos_radius'][3]  = radius
        g['ktrc'] = (kind, team, required_kind, required_count)
        g['progress'] = (0, 0, 0, 0)
        self.goal_count += 1
        return idx

    # -- CPU<->GPU sync ----------------------------------------------------

    def push_entities(self):
        self.entity_ssbo.write(self.entities.tobytes())

    def push_teams(self):
        self.team_ssbo.write(self.teams.tobytes())

    def push_goals(self):
        self.goal_ssbo.write(self.goals.tobytes())

    def push_all(self):
        self.push_entities()
        self.push_teams()
        self.push_goals()

    def pull_entities(self):
        raw = self.entity_ssbo.read()
        self.entities = np.frombuffer(raw, dtype=self.ENTITY_DTYPE).copy()
        # Rebuild free list: slot 0 reserved + every dead slot (kind==0).
        kinds = self.entities['ktrf'][:, 0]
        dead = np.where(kinds == 0)[0]
        # Exclude slot 0 from free list (reserved sentinel).
        self._free_slots = [int(i) for i in dead[::-1] if i != 0]

    def pull_teams(self):
        raw = self.team_ssbo.read()
        self.teams = np.frombuffer(raw, dtype=self.TEAM_DTYPE).copy()


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def entity_groups(max_entities):
    """Workgroup count for a per-entity dispatch (local_size_x=64)."""
    return (max_entities + 63) // 64


def hash_groups(hash_total):
    """Workgroup count for a per-hash-cell dispatch (local_size_x=64)."""
    return (hash_total + 63) // 64
