# Lenia Catalogue — Handoff Notes (for improving results)

Written 2026-07-06. Goal: give the next agent a fast, accurate foothold on the
Lenia-family rules — what they actually do today, what they're *meant* to do,
where the bugs/weak spots are, and the highest-leverage next steps.

All line numbers are approximate (the file moves); grep the symbol if it drifted.

---

## TL;DR — the single most important finding

There are **1580 Lenia discoveries scoring 0.97–0.98** yet the visual/behavioural
results are weak (and "Lenia (Geminium)" is a YouTube underperformer). The reason
is **the search optimises the wrong thing.** `score_interestingness()`
(`test_harness.py` ~L2323) is a *generic* proxy built from three bell curves:
`alive_ratio`, per-step `activity`, and `surface_ratio`. It rewards any field that
is "partly full + changing + surfacey." It **cannot tell a coherent, localised,
motile Lenia organism from an amorphous churning soup** with the same aggregate
stats. So the optimiser happily parks in soup/breathing-blob regions of parameter
space that max the proxy but look nothing like real Lenia creatures.

**Improving the Lenia catalogue is mostly a metric problem, not a parameter-search
problem.** Fix/augment the score with Lenia-specific structure measures and re-search;
everything else below is secondary.

---

## Catalogue inventory

| Preset (`RULE_PRESETS` key) | Shader | Substrate | Discoveries | Notes |
|---|---|---|---|---|
| `lenia_3d` | `lenia_3d` | voxel, 1-ch | 460 (best 0.986) | Canonical single-channel 3D Lenia. `precision: fp32`. |
| `lenia_geminium` | `lenia_3d` | voxel, 1-ch | 502 (best 0.981) | Same shader, bigger kernel (R=7), tighter growth. **No `precision` key — see BUG #1.** |
| `lenia_multi` | `lenia_multi_3d` | voxel, 3-ch | 503 (best 0.978) | Non-reciprocal cyclic cross-channel coupling. `MAX_R=3` (small). `default_size=80`. |
| `flow_lenia` | `flow_lenia_potential` + `flow_lenia_advect` | voxel, 2-pass | **0** | Mass-conserving upwind FV transport. Most sophisticated, **totally unsearched**. |
| `lenia_rings` (ADDED 2026-07-16) | `lenia_rings_potential` + `lenia_rings_growth` | voxel, 2-pass | 9 | Canonical Chan multi-ring kernel: 3 searchable ring amplitudes (B vector) + shared-mem radial LUT (no exp() in inner loop). New rule so lenia_3d's catalogue stays hash-valid. |
| `particle_lenia` | `wave_3d` (dummy) + particle SSBO | particle | **0** | force_mode==6 in `PARTICLE_UPDATE_COMPUTE.glsl`. Different substrate; hand-tuned only. |
| `flagship_dual_lenia` | compose `lenia_3d`(a_) + `lenia_geminium`(b_) | voxel compose | 115 (best 0.973) | Two Lenia fields in one rule. Determinism history (see BUG #5). |

Key files:
- Shaders: `simulator.py` — `lenia_3d` ~L3846, `lenia_multi_3d` ~L3965, `flow_lenia_potential` ~L4086, `flow_lenia_advect` ~L4173.
- Presets: `simulator.py` — `lenia_3d` ~L15767, `lenia_geminium` ~L15793, `lenia_multi` ~L15811, `flow_lenia` ~L15839, `particle_lenia` ~L15883, `flagship_dual_lenia` ~L18219.
- Inits: `init_lenia_fbm` ~L18485, `init_lenia_multi_fbm` ~L18514, `init_lenia_blobs` ~L19296.
- Score: `test_harness.py` `score_interestingness` ~L2323; per-run metrics in `ca_debug/metrics.py`.
- Golden snapshots: `ca_debug/golden/{lenia_3d,lenia_geminium,lenia_multi,flow_lenia,flagship_dual_lenia}.json`.
- Particle probe: `scripts/probe_particle_lenia.py`.

---

## How each works NOW vs what it OUGHT to do

### `lenia_3d` (single-channel, canonical)
- **Now:** For each cell, convolve neighbours with a single Gaussian *ring* kernel
  (radius `R = clamp(round(radius·h_inv), 3, MAX_R=7)`, ring at `ring_pos∈[0.1,0.9]`,
  Gaussian thickness `0.15·R`). Potential `U = Σ(v·k)/Σk` (volume-normalised mean).
  Growth `G = 2·exp(-½·((U-μ)/σ)²) - 1 ∈ [-1,1]`. Forward Euler `A += dt·G`,
  clamp[0,1], then **death floor: `A<0.005 → 0`**. `dt=0.05`.
- **Ought to:** Produce localised, self-sustaining, ideally *motile* solitons
  (orbium/glider analogues) that hold shape and travel.
- **Failure modes / weak spots:**
  - **Kernel is resolution-starved at small R.** At the R=3 floor, a ring at
    `ring_pos=0.5` only spans ~1–2 voxels of shell — too coarse to be a real ring,
    so dynamics degrade toward a blur. Crisp creatures need larger `Kernel radius`
    and/or larger world (but capped at `MAX_R=7`, see BUG #2).
  - **Death floor breaks mass conservation** and can erode the faint trailing edge
    of a traveling structure, discouraging persistent gliders. Worth A/B testing
    with the floor lowered to ~0.001 or removed.
  - Growth band `μ,σ` transferability depends on kernel normalisation + R; a preset
    tuned at one size may not behave the same at another once R saturates.

### `lenia_geminium`
- Same shader as `lenia_3d`, `Kernel radius=7`, tighter `Growth width` (0.025),
  `dt=0.04`. Meant to be "fat pulsing creatures that divide and merge."
- **BUG #1 (below): it lacks `precision: fp32`** that `lenia_3d` has, despite using
  the identical kernel-convolution shader that was documented to diverge in fp16.

### `lenia_multi` (3-channel)
- **Now:** Two fixed shells (inner r=0.3, outer r=0.7, thickness 0.12), `MAX_R=3`.
  Per-channel potential = own inner-shell mean + `cross`-weighted *non-reciprocal
  cyclic* outer-shell coupling (A←+B−C, B←+C−A, C←+A−B) with a small `μ` offset per
  channel. Intended to make the three channels perpetually chase each other so
  structures travel/reorganise instead of relaxing to static blobs.
- **Ought to:** "Creatures with internal organs / differentiation" (per the description).
- **Weak spots:** `MAX_R=3` gives a very coarse kernel; the shell radii and cross
  topology are *hard-coded*, so search only tweaks `μ, σ, radius, cross`. Likely
  produces tri-colour churn (the exact thing the generic score rewards). This rule
  is a custom, **non-canonical** multi-Lenia — treat it as its own beast.

### `flow_lenia` (2-pass, mass-conserving) — **highest upside, unsearched**
- **Now:** Pass 1 (`flow_lenia_potential`) builds growth field `G(K*A)` with a
  **multi-shell kernel** (`Kernel shape` fades inner/outer shells in, breaking
  spherical symmetry). Pass 2 (`flow_lenia_advect`) transports mass along
  `v = α·∇G + swirl·(∇G × ẑ)` with a **conservative upwind finite-volume** scheme
  across 6 faces + a small conservative diffusion. Claims exact `∫A` conservation
  (no clamp, no death floor). `dt=0.10`. Params: Growth center/width, Kernel radius,
  Kernel shape, Flow strength, Swirl (6 total).
- **Ought to:** Solid creatures that collide/merge/split/deform without evaporating —
  the whole point of Flow Lenia.
- **Opportunities/verify:**
  - **Zero discoveries.** First action: confirm the search harness drives multi-pass
    rules correctly (per-pass `param_names`), then run a real search.
  - **Verify the "exact conservation" claim** empirically (`ca_debug/conservation.py`
    or a headless `Σ A` trace). The `swirl` curl term and the added diffusion are
    non-canonical extensions; confirm they don't leak mass.
  - The 6-D param space is large — a Lenia-aware metric matters even more here.

### `particle_lenia`
- **Now:** Lives entirely in the particle SSBO (`force_mode==6` in
  `shaders/PARTICLE_UPDATE_COMPUTE.glsl` ~L59): N² pairwise kernel sum `U` + repulsion.
  Piggybacks on `wave_3d` as a do-nothing voxel base (renders nothing from voxels).
  Hand-tuned defaults for size 64 + 1500 particles. "Atoms assembling" look.
- **Ought to:** Mordvintsev-style particle Lenia clusters/organisms.
- **Opportunities:** Zero discoveries — the voxel search harness almost certainly
  doesn't cover particle rules. Use / extend `scripts/probe_particle_lenia.py` and
  build a particle-specific evaluation (RDF shell structure, cluster count, motility).

### `flagship_dual_lenia`
- Composes `lenia_3d`(prefix `a_`) + `lenia_geminium`(prefix `b_`) into one rule.
- History: `ca_debug/bug_b_dual_lenia.py` investigated cross-process determinism
  divergence at size 384. Re-verify determinism if you touch composition.

---

## Suspected bugs / things to verify (concrete)

1. **BUG #1 (FIXED 2026-07-06) — `lenia_geminium` / `lenia_multi` / `flow_lenia` were missing `precision: fp32`.**
   Precision resolution order (~L24605) is `force_precision > preset['precision'] >
   _tex_format_for_size(size)`, and `_tex_format_for_size` (~L202) silently drops to
   `rgba16f` once `size³·16 > 1 GB` — i.e. **any grid ≥ ~397 (so all 512 flagship/video
   runs) was silently fp16** for these rules, while `lenia_3d` was protected (it
   documents fp16 diverging ~0.6 within 80 steps on the same shader). All three now
   carry `"precision": "fp32"`; `flagship_dual_lenia` inherits fp32 via
   `_resolve_composed_preset` (base = first sub-rule). Verified: all five resolve fp32.

2. **`MAX_R=7` cap + R≥3 floor = narrow usable kernel-fidelity window.** Above 4×
   REF_SIZE the kernel saturates (no extra detail from bigger worlds); at the floor
   the ring is under-resolved. Document/validate the size×radius sweet spot; consider
   raising `MAX_R` (perf permitting) for crisper large-world creatures.

3. **Death floor (`<0.005 → 0`) erodes faint traveling structure** and breaks
   conservation in `lenia_3d`/`lenia_multi`. A/B test lower/removed floor for motility.

4. **`flow_lenia` conservation claim unverified in-repo.** Add a `Σ A(t)` trace; the
   `swirl` curl term + conservative diffusion are extensions to the paper scheme.

5. **`flagship_dual_lenia` determinism** at large size (see `bug_b_dual_lenia.py`).
   Re-run before trusting composed-Lenia recordings.

6. **STALE GOLDENS (found 2026-07-06): `lenia_multi` and `flow_lenia` golden
   snapshots fail [crit] on unmodified code.** They pre-date the dynamics rewrites
   (multi's non-reciprocal coupling; flow's clamp-free upwind FV — the golden diff
   shows `ch0.max → 1.726`, impossible under the old clamped scheme). `lenia_3d`,
   `lenia_geminium`, `flagship_dual_lenia` goldens pass. Regenerate the two stale
   goldens once dynamics are settled — until then they provide no regression signal.

7. **STALE DISCOVERIES (found 2026-07-06): ALL 503 `lenia_multi` discoveries are
   invalid.** Their `rule_code_hash` is `d3d4e00bbc` (or absent) vs current
   `e06557187a` — they pre-date the non-reciprocal-coupling shader rewrite. Replaying
   the top organism-ranked entry (glider 0.90!) under current code → **total death**
   at both size 48 and 80, original seed included. They were also all captured at
   size 48, below the preset's own documented minimum of 80 (the harness now
   auto-bumps). The `lenia_multi` catalogue must be re-searched from scratch.
   ~550 older entries across 3d/geminium carry no hash at all (legacy backfill) —
   treat as unverified. Only 207 entries are hash-valid (`32204efa1c`).

8. **HARNESS PAIR-2 BUG (found+FIXED 2026-07-16, beyond Lenia):** headless
   `_needs_p2` checked only pass `writes` — but live `_needs_field2()` also binds
   pair 2 for `extra_fields>=1` and for `crystal_growth` (single-pass shader that
   reads/writes `u_src2/u_dst2` internally). Result: **all 8 crystal presets ran
   headless with bindings 2/3 unbound** — solute physics dead, every crystal
   search/audit result physics-wrong (~2,900 discoveries suspect). Worse, the
   crystal goldens were blessed under the same bug and captured *stale-GL-binding
   garbage* as truth. Fixed in `test_harness.py` (mirrors live logic); all 12
   failing goldens re-blessed 2026-07-16 (crystal×8, galaxy, flocking_3d,
   flow_lenia, lenia_multi); full sweep now 90 ok / 0 crit / 138 rules.

9. **CATALOGUE HEALTH (audited 2026-07-16):** of 12,320 total discoveries, only
   **32% are hash-valid**; 7,911 unhashed (legacy pre-stamping), 480 stale-hash,
   plus the entire crystal family physics-invalid regardless of hash. Worst:
   crystal_* (0% trustworthy), lenia_multi 4%, lenia_3d 9%, wave_3d 13%,
   kuramoto_3d 23%. Best-attested: game_of_life_3d / 445_rule (64%).

10. **Scoring — ADDRESSED, see next section.**

---

## The scoring problem + the fix (IMPLEMENTED 2026-07-06)

`score_interestingness()` rewards `alive_ratio` (bell ~0.05–0.4), sustained
`activity`, and `surface_ratio`, with penalties for dead/frozen/saturated/static-
cluster states. **None of these measure spatial coherence, localisation, or
organism-ness.** Churning soup and a clean glider can score identically.

**Implemented: `--metric lenia_organism`** (`_get_metric` in `test_harness.py`,
registered in `_DYNAMICS_METRICS` + all four CLI choice lists). Composition:
- generic score < 0.05 → hard gate (dead/saturated/frozen);
- **0.35 motility** (`translation_score` — the crown jewel);
- **0.20 localisation×fragmentation** (log-bell on alive fraction peaking ~6%,
  faded to 0 beyond ~200 connected components);
- 0.15 `cluster_score`, 0.10 `period_score`, 0.10 `spatial_variation`,
  0.10 generic score;
- −0.25 percolation penalty (alive>0.25 and largest component >80% of alive).

Validation on the existing catalogue: `corr(organism, generic) = −0.19` — it
measures something genuinely different. Re-ranking the 207 hash-valid entries
surfaced a **reproducible glider regime for `lenia_geminium`**:
`Growth center≈0.31, Growth width≈0.065, Kernel radius≈4.5, Ring position≈0.58,
dt≈0.013, init=lenia_blobs` — replayed across 4 seeds: glider 0.44–0.45,
7–10 compact clusters, alive 1–3%, org 0.49–0.61 every time. Note the regime:
**Growth center ~0.3 and dt ~0.013, far from preset defaults (0.12, 0.04–0.05).**

---

## Search campaigns run 2026-07-06 (`--metric lenia_organism`, 400 trials, 400 steps)

Gotcha: `--steps` is a GLOBAL arg and must come **before** the subcommand:
`python3 test_harness.py --steps 400 search <rule> --metric lenia_organism --trials 400`.
Default steps=100 is useless for glider detection (first run: all glider=0.000).

### `lenia_multi` (fresh catalogue, current hash `e06557187a`)
17 valid entries saved, org up to 0.44, good localisation (alive 1–10%, 11–100
clusters) — **but glider=0.000 universally, and this is REAL, not a detector bug.**
Replay diagnosis of the top entry: `frac>0.1` grows 0.01→0.51 over 400 steps — the
current non-reciprocal shader produces **sub-threshold fog that slowly fills the
volume**; the >0.5 "clusters" are peaks on rising fog. Also: nearly every elite
pinned `Growth width` at its 0.12 cap (the searcher exploits max band width to stay
alive — the fog driver). The old shader hit glider 0.93; the coupling rewrite
traded motility for churn. **Shader-level work needed** (e.g. re-add a death floor /
tighter growth, or revisit the coupling) before this rule can produce organisms.

### `flow_lenia` (first search EVER — multi-pass search confirmed working)
10 entries saved, org up to 0.76 with **real motility**. Consistent elite regime:
`Growth center≈0.30, Growth width≈0.04–0.07, Kernel radius≈4–4.6, Kernel shape≈0.85,
Flow strength≈1.2, Swirl≈2.4` (Swirl well above the 1.0 preset default; dt≈0.39 —
an edge-of-stability probe above dt_range max 0.30). Top candidate replayed across
4 seeds: **glider 0.30–0.38, org 0.49–0.52 every time** — fully reproducible.
Caveat: searched at default size 32; re-validate at 64+ before promoting defaults.

---

## External-repo takeaways implemented (2026-07-16)

Analyzed ThePathfindersCodex Godot repos (lenia / boids / particle-life-boids).
Where we're ahead: 3D, tiling, fp32 policy, mass conservation, headless search.
What we took:

1. **`lenia_rings` rule (DONE)** — canonical Chan multi-ring kernel: 3 equally
   spaced rings with independent searchable amplitudes (the `B` vector behind the
   2D creature zoo), two passes (convolution → pair 2; growth update), 64-bin
   radial **kernel LUT in shared memory** built cooperatively per workgroup — no
   `exp()` in the inner loop (nouveau path falls back to analytic). Deliberately a
   NEW rule so `lenia_3d`'s hash-valid catalogue survives. First searches:
   - size 32: found glider=0.951 — but replay shows a **volume-filling drifting
     wave** (alive 33%, lcf 0.72), seed-locked; not an organism. Lesson: size-32
     search rewards bulk waves.
   - size-64 re-search: **real compact gliders.** Best validated candidate:
     `Growth center 0.452, Growth width 0.076, Kernel radius 5.5, amps
     [0.58, 0.35, 0.16] (inner-weighted), dt 0.102, init=lenia_seeded` → 1–2
     compact organisms at 0.5–2% alive, glider 0.50–0.60 on 2/4 seeds (breathers
     otherwise). Elite regime overall: Growth center 0.36–0.45 with *ascending*
     (outer-weighted, Chan-canonical) amplitude profiles; the glider outlier is
     inner-weighted. The searchable B vector demonstrably matters. Note the top
     candidate's init is `lenia_seeded` — the seed library already feeds search.
2. **Seed library (DONE)** — `scripts/capture_lenia_seed.py` extracts the largest
   connected components from a validated run into `seeds/lenia/*.npz`; new init
   variant `lenia_seeded` (in `INIT_FUNCS` + lenia_3d/geminium/rings
   `init_variants`) stamps 1–3 crops with random octahedral orientation, falls back
   to `lenia_fbm` when the library is empty. Library seeded with 3 geminium-glider
   crops (30³/27³/22³). Round-trip validated: organisms survive, org 0.43–0.50 —
   though crops relax to breathers (motile context is lost in the crop; capturing
   *during* translation with velocity would be the upgrade).
3. **Not yet done:** spatial hash grid for `particle_lenia` (port from entity
   arena; would lift N from 1.5k to 20–50k), `mix_t`-style force blending,
   Plummer softening for particle modes.

---

## How to run / evaluate

```bash
# Parameter search for one rule (writes to discoveries.json)
python3 test_harness.py --search lenia_3d --trials 200 --top 10

# Render stills to eyeball results
python -m ca_debug.render_smoke --rules lenia_3d,lenia_geminium,flow_lenia --png /tmp/lenia_shots

# dt / stability convergence
python -m ca_debug.dt_convergence --rules lenia_3d,lenia_geminium

# Mass conservation (verify flow_lenia claim)
python -m ca_debug.conservation --rules flow_lenia      # confirm exact flags first

# Particle Lenia (separate substrate)
python scripts/probe_particle_lenia.py

# Golden snapshots live in ca_debug/golden/*.json — regen/compare with
# ca_debug/golden_snapshots.py after any dynamics change.
```

Interactive: launch `python3 simulator.py`, pick a Lenia preset, and use the
vis channels (State/Potential/Growth/|Δ| for single-channel; A/B/C/Activity for
multi) to see whether you have a coherent organism or soup.

---

## Suggested order of attack
1. ~~**Fix BUG #1 (fp32 keys)**~~ — DONE 2026-07-06 (all five rules resolve fp32).
2. ~~**Add the Lenia structure metric**~~ — DONE 2026-07-06 (`--metric lenia_organism`).
3. ~~**Re-search `lenia_multi` + `flow_lenia`**~~ — DONE 2026-07-06 (see campaigns above).
4. **`lenia_multi` shader work:** fix the fog-growth failure mode (sub-threshold
   background fills the volume; growth width pins at cap). Candidates: raise death
   floor, penalise wide `Growth width` in ranges, or revisit the coupling term. Then
   re-search.
5. **`flow_lenia` at scale:** re-validate the Swirl≈2.4 elite regime at size 64–128,
   verify ΣA conservation, consider updating preset defaults (Swirl 1.0 → ~2.4,
   Growth center 0.18 → ~0.30), refresh its golden.
6. **`lenia_geminium` defaults:** consider moving toward the validated glider regime
   (Growth center ~0.31, dt ~0.013) — verify visually at 512 first.
7. **Purge/flag stale discoveries** (hash ≠ current `rule_code_hash(rule)`); refresh
   the two stale goldens.
8. **Seed library:** replace/augment random `lenia_fbm` init with known-good 3D
   solitons so good creatures are reproducible rather than lottery-dependent.
9. Treat `particle_lenia` as its own track with a particle-specific evaluator.
```
