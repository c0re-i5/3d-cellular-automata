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
# 14 reserved (NCA weights), 15 reserved (deposit_tex); use 16 for accum,
# 17 for per-entity scratch (used by 2-phase election patterns: e.g.
# predators atomicMin their id into ent_scratch[prey_id] then check
# winner in a second pass; deterministic vs. atomicCompSwap race).
BIND_ACCUM       = 16
BIND_ENT_SCRATCH = 17

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
# Max entities tracked per spatial hash cell. The build_hash pass keeps
# the K SMALLEST entity slot IDs that map to each cell (atomicMin-chain
# insertion). When more than K candidates map to one cell, the K
# survivors are deterministic across runs regardless of GPU dispatch
# order. Raised from 32 -> 128 (May 2026, M3.5 "Bug O5" fix) so dense
# clustered presets (wandering_voxels, ant colonies, termite mounds) do
# not silently truncate. Memory cost: hash_total * K * 4 bytes; at
# size=128 / hash_cell=8 this is 4096 * 128 * 4 = 2 MB.
HASH_MAX_PER_CELL = 128
HASH_EMPTY_SLOT = 0xFFFFFFFF  # sentinel: slot is empty / available

# Atomic-uint accumulator field (Infra A: many-to-one deterministic deposit
# into voxel-indexed scalar fields). Floats are encoded as fixed-point
# uints (multiply by SCALE, round). SCALE=1024 gives ~10 fractional bits;
# max representable amount ≈ 4.2 million which is more than enough for
# bounded-deposit rules (pheromone, food consumption, build deposits).
DEFAULT_ACCUM_SCALE      = 1024.0
DEFAULT_MAX_ACCUM_CHANNELS = 4

# Reserved kind ids:
KIND_DEAD = 0  # slot is empty / available for reuse


# ---------------------------------------------------------------------------
# M5: Named aux-field DSL helpers
# ---------------------------------------------------------------------------
#
# Presets can declare named aux/scratch fields:
#
#   "entity_arena": {
#       ...
#       "aux_fields":     ["food_supply", "graze_demand"],
#       "scratch_fields": ["predator_claim"],
#   }
#
# Channel indices are assigned in list order (food_supply=0, ...). The
# arena's accum_channels and scratch_channels are auto-derived from the
# list lengths.
#
# Pass specs reference fields by name; these are the recognized keys
# and the integer-channel keys they expand into:
#
#   "field"          → "channel"      (single-channel passes: clear/decay)
#   "src_field"      → "src_channel"  (encode pass)
#   "dst_field"      → "dst_channel"  (decode pass)
#   "supply_field"   → "channel"      ) resolve_demand pass: supply, demand
#   "demand_field"   → "dst_channel"  )
#   "scratch_field"  → "channel"      (scratch_clear pass)
#
# Pre-existing integer "channel"/"src_channel"/"dst_channel" keys still
# work as a fallback. Unknown field names raise KeyError at
# normalization (no silent fallthrough to channel 0).

# Names of pass-spec keys we resolve and the integer-channel key each
# expands into. (named_key → int_key)
_AUX_FIELD_KEYS = {
    'field':         'channel',
    'src_field':     'src_channel',
    'dst_field':     'dst_channel',
    'supply_field':  'channel',
    'demand_field':  'dst_channel',
}
_SCRATCH_FIELD_KEYS = {
    'scratch_field': 'channel',
}


def resolve_named_fields(spec, aux_field_names, scratch_field_names,
                         preset_label=None):
    """Mutate `spec` in place, expanding any named-field keys into their
    integer-channel equivalents using the preset's aux/scratch field lists.

    `aux_field_names` and `scratch_field_names` are tuples/lists of
    names in channel order (channel = index in list).

    Raises KeyError if a named field is not declared in the preset.
    Raises ValueError if both a named and integer key are set for the
    same target (caller's spec is ambiguous).
    """
    def _lookup(name, names, kind):
        try:
            return names.index(name)
        except ValueError:
            avail = ', '.join(names) if names else '(none declared)'
            prefix = f"preset {preset_label!r}: " if preset_label else ""
            raise KeyError(
                f"{prefix}pass shader {spec.get('shader')!r} "
                f"references unknown {kind} field {name!r}. "
                f"Declared {kind} fields: {avail}") from None

    for named_key, int_key in _AUX_FIELD_KEYS.items():
        if named_key not in spec:
            continue
        if int_key in spec:
            raise ValueError(
                f"pass {spec.get('shader')!r} sets both {named_key!r} "
                f"and {int_key!r}; pick one.")
        spec[int_key] = _lookup(spec.pop(named_key),
                                aux_field_names, 'aux')
    for named_key, int_key in _SCRATCH_FIELD_KEYS.items():
        if named_key not in spec:
            continue
        if int_key in spec:
            raise ValueError(
                f"pass {spec.get('shader')!r} sets both {named_key!r} "
                f"and {int_key!r}; pick one.")
        spec[int_key] = _lookup(spec.pop(named_key),
                                scratch_field_names, 'scratch')
    return spec


def extract_aux_field_config(arena_cfg):
    """Pop and validate aux_fields / scratch_fields from an
    `entity_arena` config dict. Returns (aux_names, scratch_names,
    accum_channels, scratch_channels).

    Mutates arena_cfg by popping the four keys (so the caller's
    leftover-key check still works).

    Behaviour:
      - If `aux_fields` is given, channels = len(aux_fields) and any
        explicit `accum_channels` must match.
      - Same for `scratch_fields` / `scratch_channels`.
      - Either or both lists may be omitted (channels default to 0
        or to the explicit count).
    """
    aux_names = tuple(arena_cfg.pop('aux_fields', ()) or ())
    scratch_names = tuple(arena_cfg.pop('scratch_fields', ()) or ())
    explicit_accum = arena_cfg.pop('accum_channels', None)
    explicit_scratch = arena_cfg.pop('scratch_channels', None)

    if aux_names:
        ac = len(aux_names)
        if explicit_accum is not None and int(explicit_accum) != ac:
            raise ValueError(
                f"aux_fields={list(aux_names)} implies "
                f"accum_channels={ac}, but preset sets "
                f"accum_channels={explicit_accum}.")
        accum_channels = ac
    else:
        accum_channels = int(explicit_accum) if explicit_accum is not None else 0

    if scratch_names:
        sc = len(scratch_names)
        if explicit_scratch is not None and int(explicit_scratch) != sc:
            raise ValueError(
                f"scratch_fields={list(scratch_names)} implies "
                f"scratch_channels={sc}, but preset sets "
                f"scratch_channels={explicit_scratch}.")
        scratch_channels = sc
    else:
        scratch_channels = int(explicit_scratch) if explicit_scratch is not None else 0

    if len(set(aux_names)) != len(aux_names):
        raise ValueError(f"aux_fields contains duplicates: {list(aux_names)}")
    if len(set(scratch_names)) != len(scratch_names):
        raise ValueError(f"scratch_fields contains duplicates: {list(scratch_names)}")

    return aux_names, scratch_names, accum_channels, scratch_channels


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
// AccumBuf — voxel-indexed atomic uint accumulator (Infra A). Sized as
// (u_size^3 * u_accum_channels) when arena.accum_channels > 0. Floats are
// encoded as fixed-point uints (multiply by u_accum_scale, round). Layout
// is voxel-major: index = voxel_flat * u_accum_channels + ch.
layout(std430, binding=16) buffer AccumBuf    { uint   accum[]; };
// EntScratchBuf — per-entity-slot atomic uint scratch (Infra B for
// 2-phase election). Sized max_entities * u_ent_scratch_channels when
// arena.scratch_channels > 0; layout is entity-major:
//   index = entity_id * u_ent_scratch_channels + ch.
// Sentinel value 0xFFFFFFFF (== ENT_SCRATCH_EMPTY in this header) means
// "no winner yet". Use atomicMin to propose, then re-read and compare
// in a separate pass to commit deterministically.
layout(std430, binding=17) buffer EntScratchBuf { uint ent_scratch[]; };
#define ENT_SCRATCH_EMPTY 0xFFFFFFFFu

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
// Accumulator-field controls (Infra A). Channels==0 means the AccumBuf
// is not allocated and the accum_* helpers are no-ops.
uniform int   u_accum_channels;
uniform float u_accum_scale;
uniform float u_accum_inv_scale;
// Per-pass parameters used by built-in accumulator passes
// (accum_clear / accum_decay / accum_decode). Shader bodies that don't
// need them simply ignore them.
uniform int   u_accum_ch;        // which accum channel the pass targets
uniform int   u_accum_dst_ch;    // which voxel grid channel decode writes
uniform float u_accum_rate;      // decay multiplier or diffuse fraction
// Per-entity scratch (Infra B). 0 means EntScratchBuf is not allocated;
// the scratch_* helpers are no-ops in that case. Currently the only
// pass that touches scratch is SHADER_ENT_SCRATCH_CLEAR plus user
// shaders that hand-roll atomicMin/atomicCompSwap on it.
uniform int u_ent_scratch_channels;

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

// ── Accumulator field helpers (Infra A) ─────────────────────────────────
// These are no-ops when u_accum_channels == 0. Always safe to call.

uint voxel_flat_index(ivec3 p) {
    // 1D linear index for voxel p ∈ [0, u_size)^3. Z-major, then Y, X.
    return uint((p.z * u_size + p.y) * u_size + p.x);
}

uint accum_buf_index(int ch, ivec3 p) {
    return voxel_flat_index(p) * uint(u_accum_channels) + uint(ch);
}

// Atomic-add a scalar amount into the accumulator at (channel, voxel).
// Negative amounts are clamped to 0 — use a separate consume helper for
// subtraction (the unsigned accumulator can't go below 0). Tiny amounts
// that round to 0 uints are dropped (sub-quantization noise).
void accum_deposit(int ch, ivec3 p, float amount) {
    if (u_accum_channels <= 0 || ch < 0 || ch >= u_accum_channels) return;
    if (amount <= 0.0) return;
    uint q = uint(amount * u_accum_scale + 0.5);
    if (q == 0u) return;
    atomicAdd(accum[accum_buf_index(ch, p)], q);
}

// Atomically consume up to `request` from (ch, p). Returns the float
// amount actually consumed (≤ request, ≥ 0). Implemented as a bounded
// CAS loop so concurrent consumers each get a deterministic share
// (specifically: the first to arrive at any given iteration takes its
// requested amount; subsequent arrivals see the reduced balance). The
// per-thread result depends on arrival order, but the total consumed
// across all threads is bounded by the initial value — no over-draw.
// For perfectly arrival-order-independent consumption use a 2-phase
// reduce pattern (see Infra C in the design doc).
float accum_consume(int ch, ivec3 p, float request) {
    if (u_accum_channels <= 0 || ch < 0 || ch >= u_accum_channels) return 0.0;
    if (request <= 0.0) return 0.0;
    uint q_req = uint(request * u_accum_scale + 0.5);
    if (q_req == 0u) return 0.0;
    uint idx = accum_buf_index(ch, p);
    // CAS-loop: read current, compute take=min(cur, q_req), CAS to cur-take.
    for (int tries = 0; tries < 16; ++tries) {
        uint cur = accum[idx];
        if (cur == 0u) return 0.0;
        uint take = min(cur, q_req);
        uint observed = atomicCompSwap(accum[idx], cur, cur - take);
        if (observed == cur) {
            return float(take) * u_accum_inv_scale;
        }
        // Contended — retry. Bounded loop keeps the worst case finite.
    }
    return 0.0;
}

// Sample (read) the accumulator value at (ch, p) as a float.
float accum_sample(int ch, ivec3 p) {
    if (u_accum_channels <= 0 || ch < 0 || ch >= u_accum_channels) return 0.0;
    return float(accum[accum_buf_index(ch, p)]) * u_accum_inv_scale;
}

// Toroidal-wrap variant for periodic boundaries.
float accum_sample_wrap(int ch, ivec3 p) {
    if (u_accum_channels <= 0 || ch < 0 || ch >= u_accum_channels) return 0.0;
    ivec3 q = ((p % u_size) + u_size) % u_size;
    return float(accum[accum_buf_index(ch, q)]) * u_accum_inv_scale;
}

// ── Sensor helpers (M2) ────────────────────────────────────────────────
// Continuous-position sample of an accumulator channel with toroidal
// wrap. `wp` is in world/voxel coordinates (0..u_size). Uses floor —
// no interpolation. Adequate for Physarum-style discrete sensors.
float accum_sample_world_wrap(int ch, vec3 wp) {
    return accum_sample_wrap(ch, ivec3(floor(wp)));
}

// An arbitrary unit vector perpendicular to `v` (assumed non-zero).
// Stable: picks a world axis that is least parallel to v, then cross.
vec3 perp_axis(vec3 v) {
    vec3 vn = normalize(v);
    vec3 ref = (abs(vn.y) < 0.9) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    return normalize(cross(vn, ref));
}

// Rodrigues' rotation: rotate `v` around unit `axis` by `angle` radians.
vec3 rotate_around(vec3 v, vec3 axis, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return v * c
         + cross(axis, v) * s
         + axis * dot(axis, v) * (1.0 - c);
}

// Three-sensor read (left, forward, right) used by Physarum-style
// steering. `pos` is the agent world position; `heading` the unit
// forward vector; `axis` the unit rotation axis (e.g. perp_axis(heading)
// for a stable per-agent frame, or vec3(0,1,0) for world-yaw); `angle`
// the sensor half-spread in radians; `dist` the sensor offset in voxel
// units. Returns (.x = left, .y = forward, .z = right).
vec3 sense_three_3d(int ch, vec3 pos, vec3 heading, vec3 axis,
                    float angle, float dist) {
    vec3 hL = rotate_around(heading, axis,  angle);
    vec3 hF = heading;
    vec3 hR = rotate_around(heading, axis, -angle);
    return vec3(
        accum_sample_world_wrap(ch, pos + hL * dist),
        accum_sample_world_wrap(ch, pos + hF * dist),
        accum_sample_world_wrap(ch, pos + hR * dist)
    );
}
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
// One thread per hash cell. Resets the per-cell entity count and writes
// the EMPTY sentinel (0xFFFFFFFF) into every slot of that cell so the
// atomicMin chain in SHADER_BUILD_HASH can detect "first writer wins".
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= hash_count.length()) return;
    hash_count[i] = 0u;
    uint K = uint(u_hash_max_per_cell);
    uint base = i * K;
    for (uint k = 0u; k < K; ++k) {
        hash_entry[base + k] = 0xFFFFFFFFu;
    }
}
"""

SHADER_BUILD_HASH = """
// Deterministic hash insertion. The cell will contain the K smallest
// entity slot IDs that map to it (a min-heap by slot id, sorted ascending).
// Algorithm (atomicMin chain):
//   for k in 0..K-1:
//     old = atomicMin(slot[k], candidate)
//     if old == EMPTY:   we took an empty slot, increment count, done
//     else if old > candidate: we displaced 'old', set candidate=old, retry
//     else:              slot already had a smaller id, keep candidate, retry
//   (if loop ends with no empty slot found, candidate is dropped)
//
// Order-independence proof: the final state is determined entirely by
// the SET of (entity_id, cell) tuples, not the order they are inserted.
// Each slot can only transition EMPTY -> non-EMPTY once (atomicMin
// monotonically decreases), so 'count' = # of non-EMPTY slots is exactly
// correct. Slots stay compact (no holes) because chains always take the
// first EMPTY slot they encounter while scanning from k=0.
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_entity_count)) return;
    Entity e = entities[i];
    if (!is_alive(e)) return;
    ivec3 c = hash_cell_of(e.pos_radius.xyz);
    uint ci = hash_cell_index(c);
    uint K = uint(u_hash_max_per_cell);
    uint base = ci * K;
    uint candidate = i;
    for (uint k = 0u; k < K; ++k) {
        uint old = atomicMin(hash_entry[base + k], candidate);
        if (old == 0xFFFFFFFFu) {
            // Took a previously-empty slot. Increment cell count.
            atomicAdd(hash_count[ci], 1u);
            return;
        }
        if (old > candidate) {
            // We displaced 'old'; it now needs a new home.
            candidate = old;
        }
        // else: slot already held a smaller id; keep candidate, try next k.
    }
    // Loop exhausted; 'candidate' is greater than all K smallest ids in
    // the cell -> truncated. Deterministic across runs because the SET
    // of all candidates mapping to this cell is the same every run.
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

    // Running per-team magnitudes for the signed channel reconcile.
    // Accumulating max() inside the loop is associative & commutative
    // (unlike the old chained max/min which depended on iteration order
    // when both teams overlapped in the same voxel).
    float pos_max = 0.0;
    float neg_max = 0.0;

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
            // Track per-team running max separately so the final value
            // does not depend on the (atomicAdd-ordered) iteration order
            // of hash entries. Reconciled after the loop below.
            if (team_id == 0u) {
                pos_max = max(pos_max, w);
                result.g = max(result.g, w);
            } else if (team_id == 1u) {
                neg_max = max(neg_max, w);
                result.b = max(result.b, w);
            }
            result.a = max(result.a, w);
        }
    }

    // Order-independent reconcile for the signed display channel.
    // Pick the team with greater magnitude in this voxel; ties go
    // positive. Then combine with the read-side value via the same
    // max(+) / min(-) rule the per-entity loop used to use.
    float signed_win = (pos_max >= neg_max) ? pos_max : -neg_max;
    if (signed_win > 0.0) {
        result.r = max(result.r, signed_win);
    } else if (signed_win < 0.0) {
        result.r = min(result.r, signed_win);
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


# ── Infra A built-in passes: accumulator clear / decay / decode ─────────
#
# All three are per-voxel 1D dispatches (voxel_count threads, workgroup
# 64). They use the standard ENTITY_LAYOUT_HEADER so no special compile
# path is needed.
#
# The per-pass uniforms (u_accum_ch, u_accum_dst_ch, u_accum_rate) are
# populated by the simulator from the pass spec:
#     {"kind": "entity_accum_clear",  "channel":  0}
#     {"kind": "entity_accum_decay",  "channel":  0, "rate": 0.95}
#     {"kind": "entity_accum_decode", "channel":  0, "dst_ch": 1}
#                                                   #  0=r 1=g 2=b 3=a

# Clear: zero out one channel of the accumulator across every voxel.
SHADER_ACCUM_CLEAR = """
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint total = uint(u_size) * uint(u_size) * uint(u_size);
    if (i >= total) return;
    if (u_accum_channels <= 0) return;
    int ch = u_accum_ch;
    if (ch < 0 || ch >= u_accum_channels) return;
    uint idx = i * uint(u_accum_channels) + uint(ch);
    accum[idx] = 0u;
}
"""

# Decay: multiplies one channel by u_accum_rate. Decode → multiply →
# re-encode. Race-free because each thread reads & writes exactly its own
# voxel's uint slot — no neighbour access.
SHADER_ACCUM_DECAY = """
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint total = uint(u_size) * uint(u_size) * uint(u_size);
    if (i >= total) return;
    if (u_accum_channels <= 0) return;
    int ch = u_accum_ch;
    if (ch < 0 || ch >= u_accum_channels) return;
    uint idx = i * uint(u_accum_channels) + uint(ch);
    uint cur = accum[idx];
    if (cur == 0u) return;
    float f = float(cur) * u_accum_inv_scale;
    f *= clamp(u_accum_rate, 0.0, 1.0);
    uint q = uint(f * u_accum_scale + 0.5);
    accum[idx] = q;
}
"""

# Decode: copy the float value at accum[ch, voxel] into one channel of
# the rgba32f voxel grid. Standard src→dst pattern (same binding scheme
# as entity_paint). Caller chooses dst_ch in 0..3 (r,g,b,a). All other
# channels of the destination are preserved from u_grid_r.
SHADER_ACCUM_DECODE = """
layout(rgba32f, binding=0) uniform image3D u_grid_r;
layout(rgba32f, binding=1) uniform image3D u_grid_w;

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint total = uint(u_size) * uint(u_size) * uint(u_size);
    if (i >= total) return;
    int S = u_size;
    int idx = int(i);
    int z = idx / (S * S);
    int rem = idx - z * S * S;
    int y = rem / S;
    int x = rem - y * S;
    ivec3 p = ivec3(x, y, z);

    vec4 v = imageLoad(u_grid_r, p);
    float f = 0.0;
    if (u_accum_channels > 0 && u_accum_ch >= 0
        && u_accum_ch < u_accum_channels) {
        uint q = accum[i * uint(u_accum_channels) + uint(u_accum_ch)];
        f = float(q) * u_accum_inv_scale;
    }
    int dst = clamp(u_accum_dst_ch, 0, 3);
    if      (dst == 0) v.r = f;
    else if (dst == 1) v.g = f;
    else if (dst == 2) v.b = f;
    else               v.a = f;
    imageStore(u_grid_w, p, v);
}
"""

# Encode: read one channel of the rgba32f voxel grid and store the
# fixed-point uint representation into accum[ch, voxel]. Inverse of
# DECODE. Used when grid-side passes (diffusion, regrowth) own the
# field and per-entity reservation logic (accum_consume) needs an
# atomic view of the same data for one step. Caller chooses src
# channel via u_accum_dst_ch (overloaded — same uniform, opposite
# direction) and target accum channel via u_accum_ch.
# Race-free: one thread per voxel; only writes own slot.
SHADER_ACCUM_ENCODE = """
layout(rgba32f, binding=0) uniform image3D u_grid_r;
layout(rgba32f, binding=1) uniform image3D u_grid_w;  // unused; bound for parity

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint total = uint(u_size) * uint(u_size) * uint(u_size);
    if (i >= total) return;
    if (u_accum_channels <= 0) return;
    int ch = u_accum_ch;
    if (ch < 0 || ch >= u_accum_channels) return;

    int S = u_size;
    int idx = int(i);
    int z = idx / (S * S);
    int rem = idx - z * S * S;
    int y = rem / S;
    int x = rem - y * S;
    ivec3 p = ivec3(x, y, z);

    vec4 v = imageLoad(u_grid_r, p);
    int src = clamp(u_accum_dst_ch, 0, 3);  // overloaded: src channel here
    float f = (src == 0) ? v.r : (src == 1) ? v.g : (src == 2) ? v.b : v.a;
    f = max(f, 0.0);
    // Saturate at scale to avoid wraparound when float exceeds uint range.
    float scaled = min(f * u_accum_scale, float(0xFFFFFF00u));
    accum[i * uint(u_accum_channels) + uint(ch)] = uint(scaled + 0.5);
}
"""


# Per-entity scratch clear (Infra B). One thread per entity slot; sets
# ent_scratch[i*channels + ch] = 0xFFFFFFFF (no winner). Caller chooses
# channel via u_accum_ch (reused uniform — scratch passes are rare
# enough that adding a dedicated uniform isn't worth it).
# Race-free: each thread writes exactly its own slot.
SHADER_ENT_SCRATCH_CLEAR = """
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_entity_count)) return;
    if (u_ent_scratch_channels <= 0) return;
    int ch = u_accum_ch;
    if (ch < 0 || ch >= u_ent_scratch_channels) return;
    ent_scratch[i * uint(u_ent_scratch_channels) + uint(ch)] = ENT_SCRATCH_EMPTY;
}
"""


# Resolve demand-vs-supply for proportional fair-share consumption
# (M4c — addresses the residual nondeterminism in `accum_consume`'s
# CAS-loop where multiple consumers of a scarce resource get different
# bite amounts depending on warp execution order).
#
# Use pattern (e.g. predator-prey grazing):
#   ch_supply (e.g. 0)  — current food/resource per voxel (encoded floats)
#   ch_demand (e.g. 1)  — atomicAdd'd request from each consumer
#   After all consumers' demand pass + barrier, consumers read both
#   channels to compute their share = request * min(1, supply/demand)
#   (deterministic — atomicAdd is commutative, individual divisions
#   are pure functions of the same inputs).
#   Then this resolve pass subtracts demand from supply (clamped to 0)
#   and clears demand for the next frame.
#
# Channels: u_accum_ch = supply, u_accum_dst_ch = demand (overloaded;
# same convention as ACCUM_ENCODE).
SHADER_ACCUM_RESOLVE_DEMAND = """
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint total = uint(u_size) * uint(u_size) * uint(u_size);
    if (i >= total) return;
    if (u_accum_channels <= 0) return;
    int chs = u_accum_ch;
    int chd = u_accum_dst_ch;
    if (chs < 0 || chs >= u_accum_channels) return;
    if (chd < 0 || chd >= u_accum_channels) return;
    if (chs == chd) return;

    uint sidx = i * uint(u_accum_channels) + uint(chs);
    uint didx = i * uint(u_accum_channels) + uint(chd);
    uint supply = accum[sidx];
    uint demand = accum[didx];
    accum[sidx] = (demand >= supply) ? 0u : (supply - demand);
    accum[didx] = 0u;
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
# Physarum-3D — slime-mould-style agent ensemble using the accumulator
# ---------------------------------------------------------------------------
#
# Each agent maintains a unit heading in vel_energy.xyz. Per step it
# samples the trail accumulator (channel 0) at three points (left,
# forward, right of heading), rotates toward the strongest, advances by
# `speed`, and atomic-deposits trail at its new voxel. The trail decays
# multiplicatively each frame via SHADER_ACCUM_DECAY and is read back
# into the grid via SHADER_ACCUM_DECODE for visualisation.
#
# Uniforms (mapped from preset params via `param_names`):
#   u_param0 -- sensor_angle  (radians; spread of L/R sensors from forward)
#   u_param1 -- sensor_dist   (voxels; how far ahead sensors look)
#   u_param2 -- turn_angle    (radians; per-step rotation magnitude)
#   u_param3 -- deposit_amt   (consumed by SHADER_PHYSARUM_DEPOSIT only)
#
# Speed is fixed at 1.0 voxel per step (u_dt scales). Trail decay rate
# lives on the decay pass spec, not here.
# Two-pass design eliminates the read-after-write race that would otherwise
# arise from agents sensing the same accum field that other agents in the
# same dispatch are depositing into. SHADER_PHYSARUM_SENSE only reads accum
# (and writes own entity slot); SHADER_PHYSARUM_DEPOSIT only writes accum
# (one atomicAdd per agent). Reads and writes are now temporally separated.
SHADER_PHYSARUM_SENSE = """
// Small int hash for per-step jitter.
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

    vec3 pos = e.pos_radius.xyz;
    vec3 heading = e.vel_energy.xyz;
    // First-frame guard: agents spawn with zero velocity. Pick a random
    // unit heading per agent (deterministic from slot id).
    if (dot(heading, heading) < 1e-6) {
        heading = normalize(vec3(h11(i * 7u    ) - 0.5,
                                 h11(i * 7u + 1u) - 0.5,
                                 h11(i * 7u + 2u) - 0.5) + vec3(1e-5, 0.0, 0.0));
    } else {
        heading = normalize(heading);
    }

    float sensor_angle = u_param0;
    float sensor_dist  = u_param1;
    float turn_angle   = u_param2;
    // u_param3 (deposit_amt) is consumed by SHADER_PHYSARUM_DEPOSIT.
    float speed        = 1.0;

    vec3 axis = perp_axis(heading);
    vec3 s = sense_three_3d(0, pos, heading, axis, sensor_angle, sensor_dist);
    float L = s.x, F = s.y, R = s.z;

    float turn = 0.0;
    if (F >= L && F >= R) {
        turn = 0.0;
    } else if (L > R) {
        turn = +turn_angle;
    } else if (R > L) {
        turn = -turn_angle;
    } else {
        // Tie-break: random nudge so agents don't all freeze identically.
        turn = (h11(seed + 3u) - 0.5) * turn_angle * 2.0;
    }
    heading = rotate_around(heading, axis, turn);

    // Advance + wrap.
    pos = mod(pos + heading * speed * u_dt, u_world_size);

    e.pos_radius.xyz = pos;
    e.vel_energy.xyz = heading;
    entities[i] = e;
}
"""

SHADER_PHYSARUM_DEPOSIT = """
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_entity_count)) return;
    Entity e = entities[i];
    if (!is_alive(e)) return;
    // atomicAdd of fixed-point uint — commutative; race-free regardless of
    // how many agents land in the same voxel within one dispatch.
    accum_deposit(0, ivec3(floor(e.pos_radius.xyz)), u_param3);
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
#
# LEGACY — kept for reference. Use SHADER_PREY_DEMAND + SHADER_PREY_CONSUME
# below for bit-deterministic grazing. accum_consume's CAS loop is
# atomic but distribution-nondeterministic: when two prey contend for
# scarce food, the warp that wins the first CAS takes its full bite
# and the loser retries against a reduced cur, getting less. Totals
# match across runs but individual bites diverge.
SHADER_PREY_STEP = """
// Per-entity prey update. Grazes a food field stored in accum channel 0
// (encoded from grid ch3 once per frame by SHADER_ACCUM_ENCODE; decoded
// back to grid ch3 after this pass by SHADER_ACCUM_DECODE).
//
// O3 fix (M4): grazing now uses accum_consume — a bounded CAS loop on
// the per-voxel uint slot — so multiple prey in the same voxel cannot
// overconsume food regardless of warp execution order. The previous
// imageLoad/imageStore-on-ch3 implementation was non-atomic and let two
// prey in the same voxel both 'see' the full bite amount.
//
// Active iff this entity is alive AND kind == 1 (PREY).
//
// Params (passed via u_param0..3):
//   u_param0 — prey wander jitter scale (random-walk noise)
//   u_param1 — prey graze rate (energy/food per step)
//   u_param2 — UNUSED
//   u_param3 — global metabolism multiplier (energy decay/step)
//
// genome layout (per entity):
//   genome.x — max_speed (cells/step)

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

    // Atomic graze: request u_param1*u_dt units; actual = what was there.
    // accum_consume does a CAS loop on the uint slot, so any number of
    // prey can graze the same voxel in the same dispatch without
    // overconsumption.
    ivec3 vp = ivec3(floor(pos));
    vp = ((vp % u_size) + u_size) % u_size;
    float bite = accum_consume(0, vp, u_param1 * u_dt);

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


# ── Shader 2a / 2b: deterministic fair-share grazing (M4c) ─────────────
#
# Replaces SHADER_PREY_STEP with two passes that yield bit-identical
# per-entity energies across runs:
#
#   2a SHADER_PREY_DEMAND
#     - each prey atomicAdds its requested bite (u_param1*u_dt) into
#       accum channel 1 (demand) at its voxel. atomicAdd of uint is
#       commutative → sum is bit-identical regardless of warp order.
#     - does NOT move the entity; only deposits demand. Position used
#       for binning is the prey's CURRENT position (not post-move),
#       which matches SHADER_PREY_STEP's read-before-move semantics.
#
#   2b SHADER_PREY_CONSUME
#     - reads supply = accum[0, vp], demand = accum[1, vp]. Both are
#       frozen between the demand pass barrier and this pass.
#     - share = request * min(1, supply/demand) — pure function of two
#       float reads, deterministic.
#     - updates energy / pos / vel / starvation.
#
# After both passes, SHADER_ACCUM_RESOLVE_DEMAND (per-voxel) subtracts
# demand from supply (clamped to 0) and clears demand to 0. Then the
# regular SHADER_ACCUM_DECODE writes the new food back to grid ch3.
#
# Total cost: 2 entity passes + 1 voxel pass (instead of 1 entity pass),
# all O(N) or O(V). Worth it for full determinism.

SHADER_PREY_DEMAND = """
// Active iff alive AND kind == 1 (PREY). See SHADER_PREY_STEP for params.

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_entity_count)) return;
    Entity e = entities[i];
    if (!is_alive(e)) return;
    if (e.kind_team_role_flags.x != 1u) return;
    // Demand is request * dt. Deposit at current voxel (pre-move).
    ivec3 vp = ivec3(floor(e.pos_radius.xyz));
    vp = ((vp % u_size) + u_size) % u_size;
    accum_deposit(1, vp, u_param1 * u_dt);
}
"""


SHADER_PREY_CONSUME = """
// Per-entity prey update — reads supply & demand (both frozen since
// SHADER_PREY_DEMAND barrier), computes proportional bite, applies
// energy/motion/death logic. Bit-deterministic.

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
    if (e.kind_team_role_flags.x != 1u) return;

    uint seed = i * 1973u + uint(u_frame) * 9277u;
    vec3 jitter = vec3(h11(seed), h11(seed + 1u), h11(seed + 2u)) - 0.5;

    float max_speed = max(e.genome.x, 0.01);
    vec3 v = e.vel_energy.xyz * 0.85 + jitter * u_param0;
    float vlen = length(v);
    if (vlen > max_speed) v *= max_speed / vlen;

    vec3 pos = mod(e.pos_radius.xyz + v * u_dt, u_world_size);

    // Bite at PRE-MOVE voxel (matches demand deposit). Read frozen
    // supply & demand; bite = request * min(1, supply/demand).
    ivec3 vp = ivec3(floor(e.pos_radius.xyz));
    vp = ((vp % u_size) + u_size) % u_size;
    float supply  = accum_sample(0, vp);
    float demand  = accum_sample(1, vp);
    float request = u_param1 * u_dt;
    float bite = 0.0;
    if (demand > 0.0) {
        // request is exactly this prey's contribution to demand, so
        // share = request when demand<=supply, else request*supply/demand.
        bite = (demand <= supply) ? request : request * (supply / demand);
    }

    float energy = e.vel_energy.w + bite - u_param3 * u_dt;
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



#
# M4b fix for Bug O4: the original single-pass SHADER_PREDATOR_STEP used
# `atomicCompSwap(prey.kind, 1, 0)` to claim a kill. atomicCompSwap is
# atomic but the WINNER depends on which warp reaches the instruction
# first — pure GPU scheduling. When multiple predators were within 1.5
# cells of the same prey, the kill assignment differed across runs,
# cascading into CPU reproduction divergence.
#
# Replaced with 2-phase election using per-entity scratch (Infra B):
#   Pass A  SHADER_ENT_SCRATCH_CLEAR  — scratch[prey_id] = 0xFFFFFFFF
#   Pass B  SHADER_PREDATOR_PROPOSE   — each predator finds its target,
#           then atomicMin(scratch[prey_id], my_id+1). Winner is the
#           lowest-id predator — a function of state alone, not warp
#           scheduling.
#   Pass C  SHADER_PREDATOR_COMMIT    — each predator re-reads
#           scratch[its_target], and if (claim == my_id+1) does the kill
#           (writes prey.kind=0) and gains energy. Also runs the wander
#           branch + position/energy update for ALL predators.
#
# The propose pass writes nothing to the entity struct — only to scratch.
# The commit pass writes the full entity struct (pos, vel, energy) for
# every alive predator. Because the kill is committed by exactly one
# predator (the winner), no race exists on prey.kind.

SHADER_PREDATOR_PROPOSE = """
// Phase B: each alive predator finds its nearest live prey within
// sight, STORES that target in its own entity.target field (so COMMIT
// reads it without re-deriving), and — if within bite range (dist<1.5)
// — proposes itself via atomicMin into the prey's scratch slot.
//
// Only own-slot writes (entity.target) + per-prey atomicMin. No reads
// of mutable shared state besides the (now post-O3-deterministic)
// hash and entity positions, so this pass is bit-deterministic.
//
// Params: u_param2 — sight scale.

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_entity_count)) return;
    Entity e = entities[i];
    if (!is_alive(e)) return;
    if (e.kind_team_role_flags.x != 2u) return;  // not predator
    if (u_ent_scratch_channels <= 0) return;     // misconfigured preset

    float sight = max(e.genome.y, 1.0) * max(u_param2, 0.01);

    uint nb_ids[64]; int nb_n = 0;
    gather_neighbours(e.pos_radius.xyz, sight, nb_ids, nb_n, 64);

    // Find nearest live prey within sight. Tie-break by id (the
    // first-encountered hash-ordered match wins; hash is deterministic
    // post-O5 so this is reproducible).
    int best = -1;
    float best_d2 = sight * sight;
    for (int k = 0; k < nb_n; ++k) {
        uint id = nb_ids[k];
        if (id == i) continue;
        Entity o = entities[id];
        if (o.kind_team_role_flags.x != 1u) continue;
        vec3 d = wrap_delta(o.pos_radius.xyz, e.pos_radius.xyz);
        float d2 = dot(d, d);
        if (d2 < best_d2) {
            best = int(id);
            best_d2 = d2;
        }
    }

    // Store target+1 in our own slot (own-write, no race). 0 = no target.
    uint target_plus1 = (best >= 0) ? uint(best) + 1u : 0u;
    entities[i].target_partner_timer_payload.x = target_plus1;

    // Propose for kill iff within bite range.
    if (best >= 0 && best_d2 < 1.5 * 1.5) {
        atomicMin(ent_scratch[uint(best) * uint(u_ent_scratch_channels)],
                  uint(i) + 1u);
    }
}
"""


SHADER_PREDATOR_COMMIT = """
// Phase C: every alive predator reads its OWN target field (set by
// PROPOSE — no re-derivation) and updates pos / vel / energy. If the
// target is within bite range AND we won the phase-B election
// (scratch[target] == self_id+1), commit the kill.
//
// Race safety:
//   * Each predator writes only entities[i] (its own slot) for pos/vel/
//     energy. No two predators write the same predator slot.
//   * Kill write entities[best].kind_team_role_flags.x = 0u happens
//     for exactly one predator per prey (the unique scratch winner).
//   * Reads of entities[best].pos_radius for movement are race-free
//     because positions are never written in PROPOSE or COMMIT.
//
// Params: u_param0=wander jitter, u_param3=metabolism.

layout(rgba32f, binding=0) uniform image3D u_field;  // unused; binding parity

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
    uint target_plus1 = e.target_partner_timer_payload.x;

    vec3 v;
    bool ate = false;
    if (target_plus1 > 0u) {
        uint best = target_plus1 - 1u;
        // Read prey position (race-free — positions not written in
        // either phase). Note: we do NOT re-check prey.kind; PROPOSE
        // already saw it alive, and even if another predator killed it
        // in this commit pass we still want to pursue toward its
        // last-known position this frame.
        Entity prey = entities[best];
        vec3 toward = wrap_delta(prey.pos_radius.xyz, e.pos_radius.xyz);
        float dist = length(toward) + 1e-5;
        v = (toward / dist) * max_speed;

        // Try to commit the kill iff we won the phase-B election.
        if (dist < 1.5 && u_ent_scratch_channels > 0) {
            uint claim_idx = uint(best) * uint(u_ent_scratch_channels);
            uint winner = ent_scratch[claim_idx];
            if (winner == uint(i) + 1u) {
                // We are the sole winner; kill the prey.
                entities[best].kind_team_role_flags.x = 0u;
                ate = true;
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

    float energy = e.vel_energy.w - u_param3 * 1.1 * u_dt;
    if (ate) energy += 1.2;  // gulp — covers several steps of metabolism
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


# Legacy single-pass predator (BUG O4: atomicCompSwap order race). Kept
# only for reference / older presets that may reference it; new code
# should compose PROPOSE + COMMIT (with an ENT_SCRATCH_CLEAR before).
SHADER_PREDATOR_STEP = """
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
    if (e.kind_team_role_flags.x != 2u) return;

    float max_speed = max(e.genome.x, 0.01);
    float sight     = max(e.genome.y, 1.0) * max(u_param2, 0.01);

    uint nb_ids[64]; int nb_n = 0;
    gather_neighbours(e.pos_radius.xyz, sight, nb_ids, nb_n, 64);

    int best = -1;
    float best_d2 = sight * sight;
    for (int k = 0; k < nb_n; ++k) {
        uint id = nb_ids[k];
        if (id == i) continue;
        Entity o = entities[id];
        if (o.kind_team_role_flags.x != 1u) continue;
        vec3 d = wrap_delta(o.pos_radius.xyz, e.pos_radius.xyz);
        float d2 = dot(d, d);
        if (d2 < best_d2) { best = int(id); best_d2 = d2; }
    }

    vec3 v;
    if (best >= 0) {
        Entity prey = entities[best];
        vec3 toward = wrap_delta(prey.pos_radius.xyz, e.pos_radius.xyz);
        float dist = length(toward) + 1e-5;
        v = (toward / dist) * max_speed;
        if (dist < 1.5) {
            uint won = atomicCompSwap(entities[best].kind_team_role_flags.x, 1u, 0u);
            if (won == 1u) e.vel_energy.w += 1.2;
        }
    } else {
        uint seed = i * 4079u + uint(u_frame) * 7561u;
        vec3 jitter = vec3(h11(seed), h11(seed + 1u), h11(seed + 2u)) - 0.5;
        v = e.vel_energy.xyz * 0.9 + jitter * u_param0;
        float vlen = length(v);
        if (vlen > max_speed) v *= max_speed / vlen;
    }

    vec3 pos = mod(e.pos_radius.xyz + v * u_dt, u_world_size);
    float energy = e.vel_energy.w - u_param3 * 1.1 * u_dt;
    if (energy <= 0.0) { e.kind_team_role_flags.x = 0u; entities[i] = e; return; }
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

    // Track per-kind running max separately so the final signed channel
    // value does not depend on the (atomicAdd-ordered) iteration order
    // over hash entries — same fix as SHADER_PAINT's O6 retrofit.
    float prey_max = 0.0;
    float pred_max = 0.0;

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
            if (kind != 1u && kind != 2u) continue;

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

            if (kind == 1u) {
                prey_max  = max(prey_max,  w);
                result.g  = max(result.g,  w);
            } else {  // kind == 2u
                pred_max  = max(pred_max,  w);
                result.b  = max(result.b,  w);
            }
        }
    }

    // Order-independent reconcile for signed display channel.
    // Pick the kind with greater magnitude; ties go positive (prey).
    float signed_win = (prey_max >= pred_max) ? prey_max : -pred_max;
    if (signed_win > 0.0) {
        result.r = max(result.r,  signed_win);
    } else if (signed_win < 0.0) {
        result.r = min(result.r,  signed_win);
    }

    imageStore(u_grid_w, p, vec4(result.r, result.g, result.b, preserved_food));
}
"""


# ---------------------------------------------------------------------------
# M6 termites: chip pickup / drop / pheromone-trail emergence
# ---------------------------------------------------------------------------
#
# Classical termite-mound simulation (Wilson 1976 / Resnick 1994 style)
# lifted to deterministic GPU. Each termite either CARRIES a chip
# (tptp.x == 1) or doesn't (tptp.x == 0). Carrying termites drop their
# chip with probability proportional to local chip density (positive
# feedback → mound formation). Empty termites pick up chips from cells
# that contain them, using the same fair-share demand pattern as
# predator-prey grazing.
#
# Three named aux fields (validates the M5 DSL under realistic load):
#   chip_supply    — voxel chip density (encoded from grid ch1 each
#                    frame, modified by pickup/drop, decoded back)
#   pickup_demand  — atomicAdd'd by empty termites on chip cells;
#                    cleared and resolved each frame
#   pheromone      — atomicAdd'd by carrying termites; persists with
#                    decay; biases carrier movement toward existing
#                    trails (helps clusters form)
#
# Channels (chosen to keep the paint pass simple — it only touches
# the .b channel and preserves everything else):
#   grid ch0 (r)  — termite paint (written by SHADER_TERMITE_PAINT)
#   grid ch1 (g)  — chip_supply visualization (decoded each frame)
#   grid ch2 (b)  — pheromone visualization (decoded each frame)
#   grid ch3 (a)  — unused
#
# Termite entity state:
#   kind_team_role_flags.x = 1u (single kind)
#   target_partner_timer_payload.x = carrying flag (0 empty / 1 holding)
#   genome.x = per-termite max speed
#
# Uniform contract (u_param0..u_param3) for all three termite shaders:
#   u_param0 — termite move speed (max velocity in voxel-cells/step)
#   u_param1 — pheromone deposit rate per carrier per dt
#   u_param2 — drop affinity scale (drop_prob = min(1, supply * scale))
#   u_param3 — pheromone-attraction strength (carrier movement bias)

SHADER_TERMITE_PICKUP_DEMAND = """
// Empty termites on a chip cell register a request to pick up one
// chip. Pure atomicAdd → bit-deterministic sum.
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_entity_count)) return;
    Entity e = entities[i];
    if (!is_alive(e)) return;
    if (e.kind_team_role_flags.x != 1u) return;       // termite kind
    if (e.target_partner_timer_payload.x != 0u) return; // already carrying

    ivec3 vp = ivec3(floor(e.pos_radius.xyz));
    vp = ((vp % u_size) + u_size) % u_size;
    float supply = accum_sample(0, vp);  // chip_supply
    if (supply <= 0.0) return;

    accum_deposit(1, vp, 1.0);  // pickup_demand
}
"""


SHADER_TERMITE_PICKUP_STEP = """
// Empty termites: stochastic pickup decision (fair-share via
// supply/demand), then random-walk move. Pickup probability is
// (supply / max(demand, supply)) — deterministic per (id, frame).
// Carrying termites are handled in SHADER_TERMITE_DROP_STEP.
float th11(uint x) {
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
    if (e.kind_team_role_flags.x != 1u) return;
    if (e.target_partner_timer_payload.x != 0u) return; // skip carriers

    ivec3 vp = ivec3(floor(e.pos_radius.xyz));
    vp = ((vp % u_size) + u_size) % u_size;

    // ---- pickup decision (before move) -------------------------------
    float supply = accum_sample(0, vp);
    float demand = accum_sample(1, vp);
    uint seed = i * 1973u + uint(u_frame) * 9277u;
    if (supply > 0.0 && demand > 0.0) {
        float bite_prob = (demand <= supply) ? 1.0 : (supply / demand);
        float r = th11(seed + 13u);
        if (r < bite_prob) {
            e.target_partner_timer_payload.x = 1u;  // becomes carrier
        }
    }

    // ---- random-walk move (Brownian + velocity inertia) --------------
    vec3 jitter = vec3(th11(seed),
                       th11(seed + 1u),
                       th11(seed + 2u)) - 0.5;
    float max_speed = max(e.genome.x, 0.01);
    vec3 v = e.vel_energy.xyz * 0.80 + jitter * u_param0;
    float vlen = length(v);
    if (vlen > max_speed) v *= max_speed / vlen;
    vec3 pos = mod(e.pos_radius.xyz + v * u_dt, u_world_size);

    e.pos_radius.xyz = pos;
    e.vel_energy.xyz = v;
    entities[i] = e;
}
"""


SHADER_TERMITE_DROP_STEP = """
// Carrying termites: stochastic drop decision (probability rises with
// local chip density → positive feedback / mound formation) + pheromone-
// biased move. Reads only (chip_supply, pheromone are frozen from prior
// passes); all deposits happen in the separate SHADER_TERMITE_DEPOSIT
// pass to keep this kernel bit-deterministic (no intra-pass deposit/
// sample race).
//
// The "did_drop" decision is stored in payload.y (1 = drop this frame,
// 0 = keep carrying). SHADER_TERMITE_DEPOSIT consumes & clears it.
float td11(uint x) {
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
    if (e.kind_team_role_flags.x != 1u) return;
    if (e.target_partner_timer_payload.x == 0u) return; // only carriers

    ivec3 vp = ivec3(floor(e.pos_radius.xyz));
    vp = ((vp % u_size) + u_size) % u_size;

    // ---- drop decision (reads only) ----------------------------------
    // drop_prob = clamp(supply * drop_scale, 0, 1).  Empty cell → 0
    // probability; dense cell → near-certain drop. Combined with the
    // pheromone deposit this creates the classic positive-feedback
    // mound-formation dynamic.
    uint seed = i * 1973u + uint(u_frame) * 9277u;
    float supply = accum_sample(0, vp);
    float drop_prob = clamp(supply * u_param2, 0.0, 1.0);
    // Always allow a tiny baseline drop chance so the very first chip
    // ever placed somewhere doesn't have zero probability of seeding
    // a new cluster.
    drop_prob = max(drop_prob, 0.02);
    float r = td11(seed + 17u);
    e.target_partner_timer_payload.y = (r < drop_prob) ? 1u : 0u;

    // ---- pheromone-biased move ---------------------------------------
    // Sample pheromone at 6 axis-neighbours, pick the strongest as a
    // bias direction. Deterministic (pure reads of frozen accum).
    int S = u_size;
    ivec3 pe = ((vp + ivec3( 1, 0, 0)) % S + S) % S;
    ivec3 pw = ((vp + ivec3(-1, 0, 0)) % S + S) % S;
    ivec3 pn = ((vp + ivec3( 0, 1, 0)) % S + S) % S;
    ivec3 ps = ((vp + ivec3( 0,-1, 0)) % S + S) % S;
    ivec3 pu = ((vp + ivec3( 0, 0, 1)) % S + S) % S;
    ivec3 pd = ((vp + ivec3( 0, 0,-1)) % S + S) % S;
    float xe = accum_sample(2, pe);
    float xw = accum_sample(2, pw);
    float xn = accum_sample(2, pn);
    float xs = accum_sample(2, ps);
    float xu = accum_sample(2, pu);
    float xd = accum_sample(2, pd);
    vec3 grad = vec3(xe - xw, xn - xs, xu - xd);
    float glen = length(grad);
    vec3 bias = (glen > 1e-5) ? (grad / glen) * u_param3 : vec3(0.0);

    vec3 jitter = vec3(td11(seed),
                       td11(seed + 1u),
                       td11(seed + 2u)) - 0.5;
    float max_speed = max(e.genome.x, 0.01);
    vec3 v = e.vel_energy.xyz * 0.80 + jitter * u_param0 + bias;
    float vlen = length(v);
    if (vlen > max_speed) v *= max_speed / vlen;
    vec3 pos = mod(e.pos_radius.xyz + v * u_dt, u_world_size);

    e.pos_radius.xyz = pos;
    e.vel_energy.xyz = v;
    entities[i] = e;
}
"""


SHADER_TERMITE_DEPOSIT = """
// Apply pending carrier writes after SHADER_TERMITE_DROP_STEP.  Splits
// deposit-side work into its own pass so the decide kernel can be
// purely read-only (eliminates deposit/sample race that would otherwise
// make the gradient non-deterministic across runs).
//
// For each carrier (kind==1, payload.x==1):
//   - always deposit pheromone at current voxel
//   - if payload.y == 1 (drop_this_frame, set by DROP_STEP) then
//     deposit chip into chip_supply and flip payload.x → 0
//   - clear payload.y
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_entity_count)) return;
    Entity e = entities[i];
    if (!is_alive(e)) return;
    if (e.kind_team_role_flags.x != 1u) return;
    if (e.target_partner_timer_payload.x == 0u) return; // only carriers

    ivec3 vp = ivec3(floor(e.pos_radius.xyz));
    vp = ((vp % u_size) + u_size) % u_size;

    // Pheromone (commutative atomicAdd → deterministic sum).
    accum_deposit(2, vp, u_param1 * u_dt);

    if (e.target_partner_timer_payload.y == 1u) {
        accum_deposit(0, vp, 1.0);                  // drop chip back
        e.target_partner_timer_payload.x = 0u;       // become empty
    }
    e.target_partner_timer_payload.y = 0u;
    entities[i] = e;
}
"""


SHADER_TERMITE_PAINT = """
// Minimal voxel-paint pass for the termite preset: writes ONLY the
// .r channel (termite presence) and preserves r=undef? no — actually
// we paint into result.r and leave g/b/a alone. Channel layout:
//   r (ch0) — termite paint  (this pass)
//   g (ch1) — chip_supply    (decoded earlier)
//   b (ch2) — pheromone      (decoded earlier)
//   a (ch3) — unused (preserved)
//
// Carriers paint brighter than empty termites so structures of
// activity stand out from background wanderers.
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
    float termite_max = 0.0;

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
            if (e.kind_team_role_flags.x != 1u) continue;

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
            // Carriers paint 1.6x brighter than empty wanderers.
            if (e.target_partner_timer_payload.x != 0u) w *= 1.6;
            termite_max = max(termite_max, w);
        }
    }

    // Overwrite ch0 directly (NOT max-blend with prior value).  The
    // decoded chip_supply / pheromone fields live in g/b and were
    // written this frame by the accum-decode passes; we preserve them.
    // Using max() against imageLoad's r would smear stale paint from
    // previous frames into a static crust and the termites would look
    // frozen even though they're moving.
    result.r = termite_max;
    imageStore(u_grid_w, p, result);
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
                 hash_cell=DEFAULT_HASH_CELL,
                 accum_channels=0,
                 accum_scale=DEFAULT_ACCUM_SCALE,
                 scratch_channels=0):
        self.ctx = ctx
        self.size = int(size)
        self.max_entities = int(max_entities)
        self.max_teams = int(max_teams)
        self.max_goals = int(max_goals)
        self.hash_cell = int(hash_cell)
        # hash dim: number of cells per axis; ceil so we cover the world.
        self.hash_dim = max(1, (self.size + self.hash_cell - 1) // self.hash_cell)
        self.hash_total = self.hash_dim ** 3
        # Accumulator field (Infra A). 0 = disabled.
        self.accum_channels = int(accum_channels)
        if self.accum_channels < 0 or self.accum_channels > DEFAULT_MAX_ACCUM_CHANNELS:
            raise ValueError(
                f"accum_channels must be in [0, {DEFAULT_MAX_ACCUM_CHANNELS}], "
                f"got {self.accum_channels}")
        self.accum_scale = float(accum_scale)
        # Per-entity scratch (Infra B). 0 = disabled.
        self.scratch_channels = int(scratch_channels)
        if self.scratch_channels < 0 or self.scratch_channels > 4:
            raise ValueError(
                f"scratch_channels must be in [0, 4], got {self.scratch_channels}")
        self.voxel_count = self.size ** 3

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
        self.accum_ssbo = None
        self.scratch_ssbo = None

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
        # Accumulator buffer (Infra A). Sized voxel_count * channels.
        if self.accum_channels > 0:
            self.accum_ssbo = self.ctx.buffer(
                data=np.zeros(self.voxel_count * self.accum_channels,
                              dtype=np.uint32).tobytes())
        # Per-entity scratch buffer (Infra B). Sized max_entities * channels.
        # Initial contents don't matter — every pass that reads it must
        # be preceded by ENT_SCRATCH_CLEAR in the same frame.
        if self.scratch_channels > 0:
            self.scratch_ssbo = self.ctx.buffer(
                data=np.zeros(self.max_entities * self.scratch_channels,
                              dtype=np.uint32).tobytes())

    def release(self):
        for attr in ('entity_ssbo', 'team_ssbo', 'goal_ssbo',
                     'hash_count_ssbo', 'hash_entry_ssbo', 'accum_ssbo',
                     'scratch_ssbo'):
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
        if self.accum_ssbo is not None:
            self.accum_ssbo.bind_to_storage_buffer(BIND_ACCUM)
        if self.scratch_ssbo is not None:
            self.scratch_ssbo.bind_to_storage_buffer(BIND_ENT_SCRATCH)

    def set_uniforms(self, prog):
        """Set entity-arena uniforms on a compute program if they exist."""
        inv_scale = (1.0 / self.accum_scale) if self.accum_scale > 0 else 1.0
        for name, val in (
            ('u_entity_count',      self.max_entities),
            ('u_team_count',        self.team_count),
            ('u_goal_count',        self.goal_count),
            ('u_hash_cell',         self.hash_cell),
            ('u_hash_dim',          self.hash_dim),
            ('u_hash_max_per_cell', HASH_MAX_PER_CELL),
            ('u_world_size',        float(self.size)),
            ('u_accum_channels',    self.accum_channels),
            ('u_accum_scale',       self.accum_scale),
            ('u_accum_inv_scale',   inv_scale),
            ('u_ent_scratch_channels', self.scratch_channels),
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
