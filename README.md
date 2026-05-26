# 3D Cellular Automata

A GPU-accelerated 3D cellular automata simulator with real-time volumetric ray marching, 70+ distinct compute shaders, and 100+ built-in presets — from classic Game of Life to quantum mechanics (incl. self-interacting Schrödinger–Poisson and 3+1D Dirac), peridynamic fracture, Saffman–Taylor viscous fingering, compressible Euler with Sod / Kelvin–Helmholtz / blast initial conditions, and active nematic chirality. Ships with an extensive physical-correctness suite (`ca_debug.*`, 19 audit probes covering shader hygiene, spatial equivariance, parameter coupling, conservation laws, determinism, dt/grid convergence, init-variant smoke, long-run drift, discovery-replay fidelity, slider-endpoint runnability, render-side visibility, and golden-snapshot regression guards), plus an end-to-end recording → YouTube → Reddit publishing pipeline.

![OpenGL 4.3](https://img.shields.io/badge/OpenGL-4.3%20Compute-blue)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow)

> **Work in progress** — experimental voxel and volumetric CAs. Expect rough edges.

## Features

### Simulation
- **70+ GPU compute shaders** powering 100+ built-in presets (including 12 hand-curated flagship recordings)
- **Multi-pass physics** — presets can chain several shaders per logical step (e.g. 8× pressure-Jacobi + 1× transport for viscous fingers, 4× Poisson + 1× dynamics for galaxy formation), with per-pass parameter remapping and a second ping-pong texture pair (`u_src2`/`u_dst2`) for auxiliary fields like pressure, fear, magnetic flux, or strain
- **Grid sizes 32³ to 512³** with dynamic resizing (auto-switches rgba32f → rgba16f at large sizes)
- **Resolution-independent physics** — h² Laplacian scaling and h⁻¹ gradient scaling keeps behavior consistent across grid sizes
- **90+ initialization patterns** — sparse/dense random, centered blobs, crystal seeds, orbital wavefunctions, terrain, Winfree broken-wavefront scroll filaments, peridynamic pre-strain fields, and more
- **Boundary modes** — toroidal (wrap) or clamped (edge absorbing), switchable at runtime
- **Deterministic seeds** — reproducible simulations via explicit RNG seed control
- **Steps/batch** — run 1–20 simulation steps per render frame for fast-forward
- **Steps/sec limiter** — cap update rate (1, 2, 5, 10, 20, 30, 60, or unlimited)

### Rendering
- **Volumetric ray marching** — front-to-back emission-absorption compositing with adjustable density and brightness
- **Iso-surface mode** — adjustable threshold for surface extraction
- **Maximum Intensity Projection** (MIP) — highlights brightest features
- **Voxel renderer** — instanced cubes with adjustable gap, threshold, and opacity
- **Slice view** — X/Y/Z axis cross-section with position slider
- **6 colormaps** — Fire, Cool, Grayscale, Neon, Discrete, Spectral — with semantic legends per rule
- **Multi-channel transfer functions** — every preset stores up to 4 fields per voxel (RGBA `image3D`); the renderer can reduce them to a colour using one of five visualisation modes:
  - **Density** — single scalar channel through a colormap (legacy default)
  - **RGB channels** — paint ch0/1/2 directly into red/green/blue (e.g. forest_fire = state | was-burning | was-tree)
  - **HSV phase** — interpret ch0/ch1 as a 2D vector and use $\operatorname{atan2}$ to drive hue, magnitude to drive value (used by Kuramoto, BZ-CGL, and other oscillator/complex-amplitude rules)
  - **Bipolar** — signed channel through a diverging blue↔red colormap centred on zero (Cahn–Hilliard order parameter, EM E-field, signed displacement)
  - **RGBA blend** — ch0/1/2 as RGB and ch3 as opacity (forest fire's "just-ignited" sparks, Hodgepodge's Δ shock, Wireworld's conductor mass)
- **Per-preset `vis_mode` defaults** — promoted single-channel rules ship with the right TF wired up so loading "3D Game of Life" or "Wireworld" immediately shows the auxiliary fields, and a runtime ImGui dropdown lets you flip between modes mid-simulation
- **RGBA16F view texture** — channel-select, `vis_mode` reduction, and abs are baked into a single half-precision RGBA texture each frame; ray-marcher reads 8 B/voxel instead of redoing the math per ray sample
- **Half-resolution mode** — 2–4× faster volumetric rendering with catmull-rom upsampling
- **Compute raymarcher** — optional shared-memory cached ray-marching shader with occupancy-grid brick DDA (matches fragment path)
- **Adaptive brick DDA** — occupancy grid block edge is 4³ (small grids) or 8³ (≥192³) for fewer DDA steps and lower cache pressure
- **Idle-frame blit cache** — when camera, simulation state, and render knobs are all unchanged, the previous frame is re-displayed via a GPU framebuffer blit (~0.3 ms) instead of re-marching rays (5–30 ms at 512³)

### Recording
- **MP4 video capture** via ffmpeg (H.264 NVENC by default, libx264 fallback)
- **Resolution selector** — 4K, 1440p, 1080p, 720p, or vertical 1080×1920 (Shorts/Reels)
- **FPS selector** — 24 (cinematic), 30 (web), 60 (smooth motion, default)
- **Per-step capture cadence** (default) — locks one video frame to N sim steps so output duration is reproducible across rules regardless of how fast the simulator runs in real time. Eliminates the "this CA recorded as 3 s and that one as 90 s" problem caused by render-frame-paced capture
- **Steps-per-frame slider** (per-step mode) — fractional or integer, dial in time-lapse (`>1`) or slow-motion (`<1`) without changing duration math
- **Real-time cadence** — legacy mode preserved for users who want wall-clock-paced captures
- **Auto-stop at target duration** — pick 10 s / 30 s / 60 s / 2 min / 5 min and the recording terminates the moment the captured frame count reaches the target. Combined with per-step cadence this gives pixel-precise duration control
- **Publish-grade quality** — NVENC `cq 19` near-visually-lossless with spatial+temporal AQ, 3 B-frames, 32-frame lookahead, and a 120 Mb/s VBR ceiling so high-entropy frames don't bloat filesize unbounded; libx264 fallback uses `crf 20 / preset slow / tune film`
- **Async pixel-pack ping-pong** — two PBOs ping-ponged across frames so the GPU→CPU DMA overlaps with the next frame's render work, removing the per-frame pipeline stall
- **Pre-rendered overlay PNG** — title / description / parameter / seed / grid-size text is rasterised once at start and applied via a single `overlay` filter, replacing 4–5 per-frame `drawtext` invocations
- **Live stats overlay** — score, alive ratio, step count refresh ~10 Hz via ffmpeg `drawtext reload=1` on a temp textfile (disable with `CA_RECORDING_LIVE_STATS=0`)
- **JSON metadata sidecar** — full parameter snapshot, capture cadence, FPS, resolution, and discovery info alongside each `.mp4`
- **Background writer thread** — 120-frame queue absorbs encoder hiccups; saturated-queue events are logged so dropped frames don't go unnoticed
- **UI progress bar** when an auto-stop target is set; status line shows real-time + video duration during capture
- **Opt-in only** — recording requires `CA_RECORDING_ENABLED=1` in the environment; F5 is a no-op without it

### Publishing Pipeline
- **`youtube_pipeline/`** — OAuth 2.0 + chunked resumable upload of
  `recordings/*.mp4` to YouTube. Reads each clip's sidecar JSON to
  build the title, description, tags, and category; auto-detects
  vertical aspect ratio for Shorts; supports per-recording
  `_overrides.json` for one-off metadata tweaks. Run with
  `python -m youtube_pipeline` (see `youtube_pipeline/README.md` for
  Google Cloud Console + OAuth flow).
- **`reddit_pipeline/`** — PRAW link submission of uploaded videos to a
  subreddit, with a markdown reproduction comment containing the rule
  name, parameters, seed, and grid size. Reads
  `recordings/upload_log.jsonl` to find what's been uploaded to
  YouTube and dedupes against `reddit_log.jsonl`. Run with
  `python -m reddit_pipeline` or chain via `python -m youtube_pipeline
  --reddit` after upload. Defaults to the daily-cap-aware Shorts
  workflow.
- **Credentials are gitignored** — `**/credentials/`, `*.client_secret*`,
  and `token.json` never enter the repo. The pipeline code is public,
  the secrets stay local.

### Discovery System
- **Save discoveries** — store interesting parameter sets with interestingness score and metrics
- **Discovery browser** — scrollable list grouped by rule, sortable by score or recency
- **Filter by rule** — show all discoveries or only the current rule
- **Navigation** — `<` / `>` buttons to step through discoveries
- **Auto-restore** — loading a discovery restores rule, parameters, dt, and seed exactly

### Automated Search
- **Headless test harness** (`test_harness.py`) — GPU parameter sweeps without a visible window
- **Interestingness scoring** — composite metric from alive ratio, activity, surface complexity

### Interactive UI
- **ImGui control panel** — rule selector, parameter sliders, time step, playback controls
- **Live metrics** — GPU-computed alive count, activity, surface ratio, and interestingness score (color-coded)
- **NaN/Inf detection** — GPU metrics shader catches numerical instabilities
- **Sandbox mode** — paint elements, set temperature, erase with adjustable brush size
- **Element palette** — 16 common elements (H, C, N, O, Na, Fe, Cu, Au, etc.) for the Element Chemistry rule
- **Sandbox state save/load** — serialize full grid state to NPZ
- **FPS counter** in window title with rolling 60-frame average

### Rule Categories

| Category | Rules | Description |
|----------|-------|-------------|
| **Classic** | Game of Life, SmoothLife | Discrete and continuous life-like automata |
| **Reaction-Diffusion** | Gray-Scott, BZ, Barkley, Morphogen, BZ Turbulence | Chemical pattern formation — spots, spirals, scroll waves |
| **Continuous** | Lenia, Multi-channel Lenia | Kernel-based continuous CAs with lifelike organisms |
| **Physics** | Wave, EM Wave (Yee FDTD), Schrödinger (×6), Schrödinger–Poisson, 3+1D Dirac (4-spinor leapfrog) | Wave equations, full-vector electromagnetics, single-particle and self-interacting QM, relativistic spin-½ evolution |
| **Materials** | Crystal Growth (×5: Compact, Octahedral, Cubic, Dendritic, Snowflake), Cahn-Hilliard, Fracture (peridynamic), Erosion (hydraulic), Active Nematic (Q-tensor) | Phase separation, anisotropic solidification, brittle fracture, channelised erosion, defect-driven liquid-crystal flow |
| **Biology** | Predator-Prey, Flocking (Reynolds + predator), Physarum (adaptive flux), Mycelium (anastomosing), Lichen, Wandering Voxels (entity_arena demo) | Population dynamics, swarm behaviour, biological transport networks, validation harness for the entity-arena GPU substrate |
| **Astrophysics** | Galaxy Formation, Compressible Euler (Sod / Kelvin–Helmholtz / blast) | Self-gravity (Jacobi-Poisson) + semi-Lagrangian gas dynamics, finite-volume conservative compressible flow |
| **Chemistry** | Element CA, Viscous Fingers (Saffman–Taylor), Fire (combustion fluid), Smoke + Wind | 118-element particle chemistry, two-phase porous flow, reactive flow with vorticity confinement, advected scalar with prescribed wind field |
| **Networks** | Small-World CA (Watts–Strogatz on a 3D lattice) | Discrete CA on a graph that interpolates regular-lattice ↔ random-graph topology |
| **Geometric** | Mandelbulb, Mandelbox, Menger Sponge | Distance-estimator viewport SDFs (no voxel state — demonstrates the viewport renderer path) |

## Getting Started

### Requirements

- Python 3.10+
- OpenGL 4.3 capable GPU (NVIDIA, AMD, or Intel — nouveau has limited compute shader support)
- Linux, macOS, or Windows

### Installation

```bash
git clone https://github.com/c0re-i5/3d-cellular-automata.git
cd 3d-cellular-automata
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running

```bash
python simulator.py
```

Use the imgui panel to switch presets, adjust parameters, and control the simulation.

#### Keyboard Controls

| Key | Action |
|-----|--------|
| **Space** | Pause / Resume |
| **R** | Reset simulation |
| **→** | Single step (when paused) |
| **V** | Toggle volumetric ↔ voxel renderer |
| **B** | Toggle sandbox mode |
| **P** | Toggle paint mode |
| **1 / 2 / 3** | Brush tool: element / temperature / eraser |
| **F5** | Start / stop recording |
| **Escape** | Quit |

#### Mouse Controls

| Input | Action |
|-------|--------|
| **Left drag** | Orbit camera |
| **Right drag** | Pan camera |
| **Scroll** | Zoom |
| **Left click** | Place element (sandbox/paint mode) |
| **Right click** | Erase (sandbox/paint mode) |

#### Command Line

```bash
python simulator.py                          # default (64³ Game of Life)
python simulator.py --size 128               # custom grid size
python simulator.py --rule gray_scott        # specific rule preset
python simulator.py --discovery discoveries.json --discovery-index 5  # load saved discovery

# Enable video recording (opt-in; requires ffmpeg on PATH)
export CA_RECORDING_ENABLED=1
python simulator.py
```

### Headless Testing

The test harness runs GPU parameter sweeps without a window, scoring each trial by interestingness:

```bash
python test_harness.py                       # single headless run
```

### Correctness Probes

All probes run headless and exit non-zero on any `crit` finding, suitable for CI:

```bash
# Original five (static + per-step physics)
python -m ca_debug.shader_lint               # static GLSL review (~2 s)
python -m ca_debug.symmetry                  # spatial equivariance (~5 s, GPU)
python -m ca_debug.coupling                  # parameter sensitivity (~12 s, GPU)
python -m ca_debug.conservation              # mass/charge/prob drift (~10 s, GPU)
python -m ca_debug.limits                    # determinism + edge cases (~8 s, GPU)

# Extended suite (numerical correctness + coverage)
python -m ca_debug.determinism               # bit-identical replays (~8 s, GPU)
python -m ca_debug.aligned_sizes             # non-aligned grid sizes 65/100/127 (~12 s, GPU)
python -m ca_debug.dt_convergence            # halve-dt double-steps scaling (~25 s, GPU)
python -m ca_debug.boundary_honour           # toroidal/clamp/mirror honoured (~9 s, GPU)
python -m ca_debug.init_variants             # each init variant evolves cleanly (~6 s, GPU)
python -m ca_debug.long_run                  # 5000-step NaN/drift audit (~21 s, GPU)
python -m ca_debug.vis_channels              # declared channels carry signal (~10 s, GPU)
python -m ca_debug.recording_roundtrip       # recordings/*.json still replay (~5 s, GPU)
python -m ca_debug.init_density_scaling      # init density invariant across grid sizes (~30 s, GPU)
python -m ca_debug.param_coherence           # static u_paramK ↔ preset slot binding (~10 ms)
python -m ca_debug.discoveries_replay        # discoveries.json catalogue replayable (~9 s, GPU)
python -m ca_debug.param_endpoints           # every slider's full range is runnable (~8 s, GPU)
python -m ca_debug.dt_endpoints              # dt_range endpoints stay finite (~1 s, GPU)
python -m ca_debug.visibility_audit          # would-the-user-see-anything render mask (~15 s, GPU)
python -m ca_debug.golden_snapshots          # bit-exact + tol-stat regression vs blessed state (~18 s, GPU)
```

Each probe accepts `--rules <comma-separated>` to scope to a single preset, and `--size`/`--steps` to widen or narrow the test window.

## Architecture

```
simulator.py            — Main simulator: GLSL shaders, rendering, UI (~31 000 lines)
element_data.py         — Periodic table data for the Element Chemistry rule
entity_arena.py         — GPU substrate for typed voxel-resident agents
                          (predator/prey, wandering teams) with spatial-hash
                          neighbour queries and SSBO-backed entity storage
nca_trainer.py          — Offline trainer for the 3D neural CA preset; exports
                          MLP weights to .npz for runtime loading
trained_nca/            — Pre-trained NCA weight blobs (sphere, torus targets)
test_harness.py         — Headless parameter sweep and discovery engine (~4 200 lines)
snapshot_3d.py          — Headless renderer + multi-channel auditor (PNG strips, channel-utilisation reports)
schema.py               — Discovery-record schema v1 (canonical field list,
                          strict accessor, version gates). Producers and
                          consumers go through `get_field` so a missing v1
                          field on a v1+ entry raises immediately.
audit.py                — Corpus auditor for discoveries.json: schema
                          coverage, cross-reference of `derived_from`
                          links, optional GPU replay sample (`--replay K`)
                          that re-scores entries from their recorded
                          size/steps/seed/params and reports drift, and a
                          code-surface pass (bare-`except` triage, ruff
                          BLE001 enforcement). See "Reproducibility &
                          Audit" below.
refine.py / batch_refine.py
                        — Deep-refinement pipeline: takes a parent
                          discovery and runs a longer, larger trial with
                          extra dynamics analysis (period, growth, cluster,
                          translation), writing report.json into a
                          per-discovery refinement directory. The batch
                          driver fans this out across the corpus.
scripts/                — One-off maintenance scripts:
                          `annotate_bare_except.py` (mass-annotates
                          `except Exception:` with `# noqa: BLE001`
                          + reason) and `backfill_legacy_discoveries.py`
                          (adds size/steps + provenance marker to pre-v1
                          discovery entries).
ca_dashboard.py         — Read-only TUI dashboard over discoveries +
                          recordings (live previews, top-rule summary,
                          recent activity).
ca_debug/               — Unified debug + data-capture + correctness-probe package
                          (see Quality Assurance section below). Includes
                          `ca_debug/scratch/` — informal investigation scripts
                          accumulated while hunting specific bugs (density
                          audit, particle SSBO probe, per-rule validators).
lattice.py, fcc_field.py, fcc_render.py, fcc_rule_gray_scott.py,
fcc_viewer.py, lattice_gpu_check.py
                        — **Work in progress / on the back burner.**
                          Experimental face-centred-cubic (FCC) lattice
                          substrate: dense native storage in primitive
                          cell coordinates, 12-NN Laplacian, voxel
                          renderer with rhombohedral primitive cells. The
                          goal is denser sphere packing than the cubic
                          lattice (12 nearest neighbours instead of 6,
                          $\pi/\sqrt{18}\approx 74\%$ packing fraction
                          vs. cubic's $\pi/6\approx 52\%$) for more
                          isotropic discrete diffusion. One rule
                          (Gray-Scott) and a headless viewer exist as a
                          proof of concept; not wired into the main
                          simulator UI and not the path used by the 100+
                          cubic-lattice presets. The pre-FCC commit is
                          tagged `pre-fcc-transition` for rollback.
youtube_pipeline/       — OAuth + chunked resumable upload of `recordings/`
                          MP4s to YouTube. Reads sidecar JSON for titles,
                          descriptions, and Shorts detection. See
                          `youtube_pipeline/README.md` for OAuth setup.
reddit_pipeline/        — PRAW link-submission of uploaded videos to a
                          subreddit, with markdown reproduction comment
                          and dedupe log. Reads `recordings/upload_log.jsonl`.
```

> Credentials live under `*/credentials/` (gitignored). Discovery files,
> recordings, and personal batch-search shell scripts are also gitignored
> and not part of the repository.

All compute shaders are embedded in `simulator.py` as GLSL source strings, compiled at runtime via moderngl. The rendering pipeline uses a separate ray marching fragment shader with emission-absorption volume integration.

### Compute Shader Design

Each rule is a GLSL compute shader dispatched over an 8×8×8 workgroup grid. A shared `COMPUTE_HEADER` provides:

- Two pairs of ping-pong textures: primary `u_src`/`u_dst` (bindings 0/1, four channels) and optional auxiliary `u_src2`/`u_dst2` (bindings 2/3) for fields that need their own state — pressure, fear, magnetic flux density, particle velocity, broken-bond integrity, etc.
- Resolution-independent scaling via `h_sq` and `h_inv` (referenced to 128³)
- Boundary handling (toroidal wrap, clamp, or mirror/Neumann)
- Utility functions: `fetch()`, `fetch2()`, `fetch_interp()`, `hash_temporal()`, `lap19_v4()` (19-point isotropic Laplacian)

### Multi-pass presets

A preset can declare a list of passes; each pass picks a shader, a write target (`p1` or `p2` — or both for fused updates), and an optional per-pass parameter remap so the same global slider can mean different things to different passes. This lets a single logical step run, for example, eight Jacobi sweeps over a pressure field and then one transport step that reads the converged pressure.

Shaders that use large neighbourhood radii (SmoothLife, Lenia) have an optimised shared-memory tiling path with automatic fallback for drivers that don't support it. Most reaction-diffusion shaders use a 19-point Patra–Karttunen stencil whose leading error ∝ ∇²(∇²f) is rotationally symmetric — the cheaper 6-point stencil's anisotropic Σ ∂⁴/∂xi⁴ error visibly distorts spirals and droplets toward grid axes.

## Quality Assurance

The `ca_debug/` package provides a 19-probe correctness suite that audits every preset against a different physical-correctness criterion. All probes are headless and run end-to-end against the real simulator pipeline (the `HeadlessRunner` is the same one `test_harness.py` uses), so the verdict reflects the production code path.

```bash
# Original physics-correctness probes
python -m ca_debug.shader_lint    # static-analysis pass over GLSL source
python -m ca_debug.symmetry       # spatial-equivariance check (rotate / reflect / translate)
python -m ca_debug.coupling       # parameter-sensitivity matrix
python -m ca_debug.conservation   # mass / charge / probability drift
python -m ca_debug.limits         # determinism + empty-IC + damping-zero edge cases

# Numerical-correctness + coverage probes (added during the fcc-lattice bug-hunt)
python -m ca_debug.determinism            # same seed → bit-identical output across 32..256
python -m ca_debug.aligned_sizes          # behaviour at non-canonical grid sizes (65, 100, 127)
python -m ca_debug.dt_convergence         # halve dt, double steps — verifies per-time scaling
python -m ca_debug.boundary_honour        # toroidal/clamped/mirror actually differ at the edge
python -m ca_debug.init_variants          # every declared init variant constructs and evolves
python -m ca_debug.long_run               # 5000-step drift / NaN audit
python -m ca_debug.vis_channels           # each declared vis-channel actually carries signal
python -m ca_debug.recording_roundtrip    # recordings/*.json still reproducible under current engine
python -m ca_debug.init_density_scaling   # alive-fraction invariant across grid sizes
python -m ca_debug.param_coherence        # static GLSL u_paramK reads ↔ preset slot bindings
python -m ca_debug.discoveries_replay     # discoveries.json catalogue (28k entries) replayable
python -m ca_debug.param_endpoints        # every slider's full declared range runs without crash/NaN
python -m ca_debug.dt_endpoints           # dt_range endpoints stay finite (catches over-generous dt_max)
python -m ca_debug.visibility_audit       # would-the-user-see-anything (renderer mask on vis_default)
python -m ca_debug.golden_snapshots       # bit-exact + tol-stat regression vs blessed engine state
```

| Probe | What it catches |
|-------|-----------------|
| **shader_lint** | Static GLSL source review: undeclared uniforms, missing bounds checks, unscaled Laplacians, hard-coded grid-size literals, single-axis hash idioms, etc. Pure regex/AST inspection — no GPU required. |
| **symmetry** | Re-runs each preset on a mirrored / 90°-rotated / translated initial condition and compares against the un-transformed run. Catches accidental axis preferences (only-Y gravity in a horizontally isotropic rule, lab-frame stencil bias, etc.). Auto-detects 25+ shaders that *intentionally* break equivariance via lab-frame quenched noise (crystal twin nucleation, defect FBM, erosion rain cells, galaxy chirality) and skips them. |
| **coupling** | Per-parameter sensitivity matrix: perturbs each slider ±10 % and measures the per-channel response. Flags `DEAD` (no effect), `EXP` (≥10× explosion), `ASYM` (one-sided response), and `SAT` (param is responsive at extreme values but the design ±10 % range is inside a shader-internal clamp). Distinguishes mode/init-time/normal params so categorical knobs don't false-positive. |
| **conservation** | Tracks per-channel L¹ drift over a fixed number of steps for rules with explicit conservation laws (probability density for Schrödinger, mass for Cahn–Hilliard / Gray-Scott, integer particle count for sandpile). Per-voxel absolute-drift floor prevents near-zero-baseline rules from blowing up the relative-error metric. |
| **limits** | Three edge cases per rule: (1) determinism — same seed twice must produce identical output; (2) empty IC must stay empty (catches "thermal" sources that bootstrap the sim from vacuum); (3) zeroing the damping coefficient must NOT cause an `inf`/`NaN` blow-up within the design step horizon. |
| **determinism** | Multi-trial bit-identical replay at sizes 32, 64, 128, 192, 256. Caught a missing GL memory barrier in the texture-pool reuse path that produced silent corruption on the first reused allocation. |
| **aligned_sizes** | Replays each rule at non-canonical grid sizes (65, 100, 127, 129) that exercise off-by-one paths in stencils, dispatch ceil-divisions, and init density math. Caught two density-discontinuity bugs in 6 binary-mask init functions whose `scipy.zoom(order=1)` smoothing collapsed occupancy 33× at fractional sizes. |
| **dt_convergence** | Halves dt while doubling step count and compares the two trajectories; per-time-evolving rules should converge. Caught a Poisson nucleation term in `bz_excitable` whose probability was per-frame rather than per-time (ratio = 2.00 across dt halving), now scaled by `u_dt/0.05`. |
| **boundary_honour** | Runs each rule under all three boundary modes (toroidal / clamped / mirror) from a common seed and verifies the outer slab actually differs between modes. Catches shaders that silently bake one boundary policy regardless of `u_boundary`. |
| **init_variants** | Every declared init variant must construct without crash, produce a finite initial state, and evolve (not stay frozen at t=0). Caught an `rng.integers()` call in `init_nca_random_specks` that crashed under the legacy `RandomState` API. |
| **long_run** | 5000-step replay at moderate size with checkpointed state, growth-ratio tracking, and NaN/Inf surveillance. Verifies integrator stability beyond the short-horizon probes' default windows. |
| **vis_channels** | For rules using direct-mapping vis modes (density / bipolar / signed), each named channel must show signal within the preset's audit-step horizon. Catches dead-output or mis-wired vis_default settings. |
| **recording_roundtrip** | Reconstructs every `.json` recording sidecar's `(rule, size, seed, params, dt)` and replays it; flags rule renames, schema drift, non-determinism, or NaN. Verifies the historical recording catalogue stays reproducible as the engine evolves. |
| **init_density_scaling** | Measures `alive_frac` of channel-0 across grid sizes 32, 64, 96, 128, 192 per (rule, init variant). Two-regime grader (dense max/min ratio vs sparse log-log slope) catches density that should be size-invariant but isn't. Caught two "N seeds scattered in 3D but N was sized as if 2D" bugs in `nca_3d` and `greenberg_hastings_3d` init paths. |
| **param_coherence** | Static GLSL parse: for every preset's pass, verifies the shader's `u_paramK` reads are bound to actual slots (no stale-zero sliders) and every declared preset param is read by at least one pass (no dead GUI sliders). Skips rules whose binding paths it doesn't model (viewport raymarchers, particle-SSBO compute, entity-arena shaders). |
| **discoveries_replay** | Samples `discoveries.json` (28k entries, 82 rules, 170 (rule, init-variant) pairs) and re-runs each via `HeadlessRunner` with its recorded params/dt/seed; catches rule renames, param-schema drift, init-variant removal, and replay crashes. |
| **param_endpoints** | For each rule × slider × endpoint (low/high from `preset['param_ranges']`), sets just that one param to the endpoint and replays for `--cap` steps; flags crash or NaN/Inf. Verifies every advertised slider range is actually runnable. |
| **dt_endpoints** | Like `param_endpoints` but for the `dt_range` slider: tests dt at (dt_min, dt_max) with default params; flags CFL-violation blow-ups. Caught `compressible_euler_3d` advertising `dt_range=(0.005, 0.3)` while NaN-overflowing for dt ≥ 0.12 (CFL > 0.6). |
| **visibility_audit** | Closes the render-side blind spot: replays each rule to its audit horizon and computes the same visibility mask the GPU renderer applies (`val > voxel_threshold` in voxel mode, `val > iso_threshold` in iso mode, normalised value above accumulation floor in volumetric mode) on the `vis_default` channel; grades by visible-voxel fraction. Catches the Bug J class — simulation evolves fine but renderer culls everything (e.g. unset `voxel_threshold` smaller than the channel's natural range). Bonus diagnostic: flags channels whose (p1,p99) percentile falls outside `vis_range` (dim / saturated / narrow). |
| **golden_snapshots** | Level-3 regression guard: stores bit-exact byte hashes plus per-channel summary stats (mean/std/min/max/alive_count) of each rule's grid at fixed checkpoint steps in `ca_debug/golden/<rule>.json`. `--check` (default) grades divergence vs blessed state — `ok` for hash-identical, `high` for hash-differs/stats-within-tol (FP noise), `crit` for stats-diverged beyond `--rtol` (default 1%). Converts the user's act of visual approval into a permanent regression guard. Skips agent/particle/SSBO/viewport rules whose output isn't reproducibly hashable. |

Probes share a common headless harness in `ca_debug/recorder.py` and emit terminal-readable summaries plus optional JSON for CI:

```
Summary: crit=0 high=0 med=2 ok=84 n/a=0 err=0
```

### Investigation scratchpad

`ca_debug/scratch/` collects the informal scripts written while hunting
specific bugs — they are not part of CI and may break when their target
rule's preset changes, but they document the methodology used to validate
the 100+ presets. Highlights:

- **`density_audit.py`** — mirrors the GPU view shader's density math on
  the CPU and flags rules whose voxels saturate or vanish under their
  declared `vis_mode`. Found five solid-cube / uniform-haze rendering
  bugs in a single audit pass.
- **`particle_debug.py`** — reads the particle SSBO and deposit texture
  each frame. Caught an uninitialised-deposit-buffer bug that made every
  particle-coupled CA preset read $1.88 \times 10^{31}$ + NaN garbage on
  its first step.
- **`validate_*.py`** — per-rule analytic / lattice-solution checks
  (Ising, Langton's ant, Margolus block CA, fluid projection,
  predator-prey lattice, sparse dispatch).
- **`audit_*.py` / `probe_*.py` / `sweep_*.py`** — per-rule
  investigations, mostly crystal-growth dendritic-tip kinetics.

### Multi-channel field layout

Every rule's primary state lives in an RGBA `image3D` (4 channels per voxel), so even single-state automata like Game of Life have three "free" channels. Many discrete rules now compute auxiliary diagnostic fields and write them into ch1–ch3 every step, at near-zero cost (the values were already in registers from the main update). The renderer picks them up automatically through the `vis_mode` system.

The dispatcher writes back the *same* texture format, so:
- **Voxel mode** still reads only `.r` (state) — auxiliary channels are ignored, no behaviour change
- **Volumetric mode** lets the per-preset `vis_mode` choose how to reduce ch0–ch3 into a colour
- **Discovery scoring / GPU metrics** still operate on `.r`; auxiliaries are visualisation-onlyA condensed reference for what the auxiliary channels mean per rule (see source for full list):

| Rule | ch0 | ch1 | ch2 | ch3 | default `vis_mode` |
|------|-----|-----|-----|-----|--------------------|
| `game_of_life_3d` | alive | live-neighbour density | \|Δ\| (just-changed) | — | rgb_channels |
| `smoothlife_3d` | state | outer-annulus mean *n* | growth | \|Δ\| | rgb_channels |
| `lenia_3d` | state | potential *U* | growth | \|Δ\| | rgb_channels |
| `larger_than_life_3d` | alive | neighbour density | \|Δ\| | — | rgb_channels |
| `eden_3d` | solid | growth pressure | newly grown | — | rgb_channels |
| `ising_3d` | spin | alignment (Σnn+6)/12 | flipped | accept | rgb_channels |
| `sandpile_3d` | grains | supercritical (≥T) | slope | height/T | rgb_channels |
| `hodgepodge_3d` | state | infected/26 | ill/26 | \|Δ\| | rgba_blend |
| `prisoners_dilemma_3d` | strategy | local score | coop fraction | changed | rgba_blend |
| `xy_spin_3d` | θ/2π | ½+½cos θ | ½+½sin θ | \|Δθ\|/π | rgb_channels |
| `forest_fire_3d` | state | was burning | was tree | just ignited | rgba_blend |
| `greenberg_hastings_3d` | state | just-excited | excited n/6 | refractory | rgba_blend |
| `wireworld_3d` | state/3 | head | tail | conductor | rgba_blend |
| `margolus_3d` | particle | block hash | rotation axis | rotated | rgb_channels |
| `gray_scott` | U substrate | V catalyst | — | — | rgb_channels |
| `bz_cgl_3d` | Re(A) | Im(A) | phase | — | hsv_phase |
| `bz_turbulence_3d` | u | v | filament S | phase φ | hsv_phase |
| `kuramoto_3d` | cos θ | sin θ | local coherence | — | hsv_phase |
| `wave_3d` | displacement | velocity | — | — | bipolar |
| `cahn_hilliard_3d` | order param ∈[-1,1] | chemical potential | — | — | bipolar |
| `em_wave_3d` | E_x | E_y | E_z | ε_r | bipolar |
| `fire_3d` | temperature | soot | embers | fuel | rgba_blend |
| Schrödinger family (×8) | ψ real | ψ imag | potential V | \|Ψ\|² | (default density on \|Ψ\|²) |

The `wireworld_3d` row carries one important invariant: `ch0` *must* remain `state/3` because the kernel re-reads it via `ww_state()` to recover the discrete state on the next step. Promoting `ch1–ch3` is free; ch0 is load-bearing.

## Reproducibility & Audit

The discovery corpus and the producer/consumer code are kept honest by a small
schema + auditor stack rather than convention. `discoveries.json` accumulates
entries from every search run and is the main long-lived artefact in the
project; it has to survive code edits, refactors, and the inevitable drift
between "the rule that scored 0.87" and "the rule as it exists today".

**Discovery schema v1** (`schema.py`). Every entry written by current
producers carries:

```
schema_version, rule, params, score, seed,   # identity
size, steps, rule_code_hash                   # reproducibility
```

`rule_code_hash` is a digest of the rule's GLSL source at the moment the
entry was written. `schema.get_field(entry, name)` is strict on v1+ entries
and raises if a required field is missing, so silent producer bugs surface
the next time the entry is touched.

**Legacy backfill.** Pre-v1 entries (the bulk of the historical corpus)
were backfilled in place with the audit's documented historical replay
defaults (`size=48`, `steps=200`) plus a `_legacy_backfill` provenance
dict, so consumers can read `size`/`steps` uniformly across the corpus.
`schema_version` stays absent on these entries — they have no
`rule_code_hash`, so bit-exact replay is **not** guaranteed; the audit
flags them explicitly.

**Audit passes** (`python audit.py [--replay K]`):

| Pass | Checks |
|------|--------|
| **1 — schema** | Field coverage per entry, per-rule field shape clusters, value-range sanity (score ∈ [0, 1], no NaN/Inf), v1-required-field presence on v1+ entries. |
| **2 — cross-reference** | `derived_from` links resolve to existing entries; refinement blocks point at on-disk `report.json`. |
| **3 — replay (opt-in, GPU)** | Re-runs a random sample of entries at their recorded `size/steps/seed/params/dt`, compares fresh score against recorded. Bit-exact match expected for v1+ entries whose `rule_code_hash` still matches current source (drift is reported); legacy entries are replayed under the backfilled defaults and treated as approximate. |
| **4 — code surface** | Bare-`except` triage across the whole repo: every `except Exception:` must either narrow to a typed exception or carry a `# noqa: BLE001  <reason>` marker. Counts and per-file breakdown go into `audit_report.md`. |

**Lint enforcement.** `ruff.toml` selects `BLE001` so an unannotated
`except Exception:` is a hard error in CI:

```bash
ruff check .       # fails on any bare exception without a noqa: BLE001 marker
```

This puts a one-line audit trail on every defensive catch in the codebase
(225 sites at last count) — "why does this function swallow exceptions?"
is answered next to the `except`, not in commit history.

## Mathematical Reference

Every rule below runs as a GLSL compute shader on a 3D grid. Spatial derivatives use either the 7-point stencil Laplacian $\nabla^2 f = \sum_{\text{nn}} f - 6f$ scaled by $h^2 = (128/N)^2$, or the isotropic 19-point Patra-Karttunen stencil $\nabla^2 f \approx (\tfrac{1}{3}\Sigma_{\text{face}} + \tfrac{1}{6}\Sigma_{\text{edge}} - 4f)/h^2$ where rotational invariance matters. Gradients use central differences scaled by $h^{-1} = N/128$.

---

### Classical Automata

**Game of Life 3D** — Discrete totalistic rule on the 26-cell Moore neighborhood:

$$s^{t+1} = \begin{cases} 1 & \text{if } s^t = 0 \;\wedge\; b_1 \le N_{26} \le b_2 \\\ 1 & \text{if } s^t > 0 \;\wedge\; s_1 \le N_{26} \le s_2 \\\ 0 & \text{otherwise} \end{cases}$$

**SmoothLife 3D** (Rafler) — Continuous generalization with concentric spherical shells at radii $r_i = 1.5 h^{-1}$, $r_o = 2.5 h^{-1}$. Inner mean $m$ and outer mean $n$ drive a smooth sigmoid transition:

$$s^{t+1} = s^t + \Delta t \left(2\,\sigma\!\left(n,\; \ell(m),\; h(m),\; 0.03\right) - 1\right)$$

where $\sigma(x, a, b, w) = S(x,a,w)(1 - S(x,b,w))$ with $S(x,c,w) = (1 + e^{-(x-c)/w})^{-1}$, and $\ell, h$ interpolate between birth and survival intervals based on $m$.

---

### Reaction-Diffusion

**Gray-Scott** — Two-species feed-kill system producing spots, worms, and mitotic patterns:

$$\partial_t U = D_U \nabla^2 U - UV^2 + F(1-U)$$
$$\partial_t V = D_V \nabla^2 V + UV^2 - (F+k)V$$

**BZ (Complex Ginzburg-Landau)** — Normal form of the Belousov-Zhabotinsky oscillating reaction, writing $A = u + iv$:

$$\partial_t A = \mu A + (1 + i\alpha)D\nabla^2 A - (1 + i\beta)|A|^2 A$$

**Barkley** — Fast-slow excitable medium with stochastic nucleation:

$$\partial_t u = \varepsilon^{-1}\,\xi(v)\,u(1-u)\!\left(u - \tfrac{v+b}{a}\right) + D_u \nabla^2 u, \qquad \partial_t v = u - v$$

**BZ Turbulence (3D scroll waves)** — Barkley kinetics with **cubic recovery** so the inhibitor can drop back below threshold and the medium can re-excite, sustaining true 3D scroll waves rather than burning out after one rotation:

$$\partial_t u = \varepsilon^{-1}\,u(1-u)\!\left(u - \tfrac{v+b}{a}\right) + D \nabla^2 u, \qquad \partial_t v = u^3 - v$$

Filament strength is tracked as $S = |\nabla u \times \nabla v|$ windowed by $4u(1-u)\exp(-((v-0.3)\cdot 4)^2)\cdot 50$ (only the wavefront contributes), and the Winfree phase coordinate $\phi = \tfrac{1}{2\pi}\operatorname{atan2}(v-\tfrac{1}{2},\, u-\tfrac{1}{2}) + \tfrac{1}{2}$ is stored alongside for visualisation. Initial condition is one or more *broken wavefronts* (front + tail terminating inside the box) so each scroll core is anchored to a true phase-singularity line.

**Morphogen (Gierer-Meinhardt)** — Turing instability with activator saturation:

$$\partial_t a = D_a \nabla^2 a + \rho\!\left(\frac{a^2}{h(1 + \kappa a^2)} - a\right) + \sigma_a, \qquad \partial_t h = D_h \nabla^2 h + \rho(a^2 - h)$$

---

### Continuous CA (Lenia Family)

**Lenia 3D** — Gaussian ring kernel convolution with Gaussian growth function:

$$K(r) = \exp\!\left(-\tfrac{1}{2}\left(\tfrac{r/R - \beta}{0.15}\right)^2\right), \qquad U = \frac{\sum_j s_j K(|\mathbf{x}_j - \mathbf{x}|)}{\sum_j K(|\mathbf{x}_j - \mathbf{x}|)}$$

$$s^{t+1} = \text{clamp}\!\left(s^t + \Delta t\left(2e^{-\frac{(U-\mu)^2}{2\sigma^2}} - 1\right),\; 0,\; 1\right)$$

**Multi-Channel Lenia** — Three species with cyclic cross-coupling. Two kernels at ring positions $0.3$ (inner) and $0.7$ (outer) with cross-channel potentials:

$$P_a = \bar{S}^{\text{inner}}_a + \chi\,\tfrac{1}{2}(\bar{S}^{\text{outer}}_b + \bar{S}^{\text{outer}}_c)$$

Each channel has a Gaussian growth function with slightly shifted $\mu$ for symmetry breaking.

---

### Wave / Field Equations

**Wave 3D** — Symplectic Euler integration of the damped wave equation with an optional sinusoidal source:

$$\partial_t v = c^2 \nabla^2 u - \gamma v + A_d \sin(\omega_d t)\,\mathbf{1}_{|\mathbf{x}-\mathbf{x}_0| < r_s}, \qquad \partial_t u = v$$

**EM Wave (Yee FDTD)** — Full vector Maxwell update on a Yee-staggered lattice with leapfrog time-stepping (second-order accurate, energy-conserving). Pair 1 stores $(E_x, E_y, E_z, \text{medium})$, pair 2 stores $(B_x, B_y, B_z)$. Each step updates **E first** using current B, then B using fresh E:

$$\partial_t \mathbf{E} = c^2 \nabla \times \mathbf{B} - \sigma \mathbf{E}, \qquad \partial_t \mathbf{B} = -\nabla \times \mathbf{E}$$

A Hertzian dipole sources $E_z = \sin(2\pi f t)$ at the box centre. CFL: $c\,dt < h/\sqrt{3}$.

---

### Quantum Mechanics

**Schrödinger 3D** — Time-dependent Schrödinger equation via Yee leapfrog (symplectic, norm-conserving):

$$i\hbar\,\partial_t \Psi = -\frac{\hbar^2}{2m}\nabla^2\Psi + V(\mathbf{r})\Psi$$

Split into real/imaginary parts with staggered updates. 6 presets: hydrogen atom, orbital, wavepacket, harmonic oscillator, tunneling, double-slit.

**Schrödinger-Poisson** — Adds self-consistent Hartree mean-field: $\nabla^2 V = -\alpha|\Psi|^2$, solved via SOR Jacobi relaxation each frame.

**Molecular Schrödinger** — Two-center softened Coulomb potential for bonding/antibonding orbitals:

$$V(\mathbf{r}) = -\frac{Z}{\sqrt{|\mathbf{r}-\mathbf{R}_1|^2 + r_s^2}} - \frac{Z}{\sqrt{|\mathbf{r}-\mathbf{R}_2|^2 + r_s^2}}$$

---

### Ecology / Population Dynamics

**Predator-Prey (Rosenzweig-MacArthur)** — Holling type II functional response with prey logistic growth:

$$\partial_t u = ru(1 - u) - \frac{auv}{1+ahu} + D_u\nabla^2 u, \qquad \partial_t v = \frac{eauv}{1+ahu} - dv + D_v\nabla^2 v$$

**Lichen** — Three-species Lotka-Volterra competition for space and a shared resource, with asymmetric growth rates and competition coefficients.

**Flocking (Reynolds boids + predator field)** — Density $\rho$ and velocity $\mathbf{v}$ on a grid (pair 1) coupled to a scalar fear field $F$ (pair 2). Velocity updates apply alignment, cohesion, separation, and a flee force $-\nabla F$, with a soft drift toward unit speed $v_0$:

$$\mathbf{v}^{t+1} = \mathbf{v}^t + \Delta t\!\left(\alpha(\langle\mathbf{v}\rangle - \mathbf{v}) + \kappa(\langle\mathbf{x}\rangle - \mathbf{x}) - \sigma \nabla\rho - \beta \nabla F\right) + \text{drift}(v_0)$$

Fear grows where $\rho$ exceeds a threshold and diffuses with decay, producing a chase-and-scatter limit cycle.

**Physarum (Tero adaptive flux network)** — Quasi-steady pressure field driven by food sources (multi-pass Jacobi relaxation), feeding a Murray-law conductivity adaptation:

$$\nabla\!\cdot(\rho\,\nabla p) = q_{\text{food}}, \qquad \partial_t \rho = \alpha\,\rho\,|\nabla p| - \beta\,\rho^2 + \gamma$$

The quadratic pruning term $\beta\rho^2$ is the *correct* Murray balance — linear pruning collapses to a uniform sheet — and the equilibrium $\rho_{\text{eq}} = \alpha|\nabla p|/\beta$ scales linearly with local flux, sculpting dendritic transport channels. A semi-implicit Euler step prevents conductivity overshoot.

**Mycelium (anisotropic + anastomosing)** — Active tip orientation $(d_x, d_y, d_z, \text{age})$ stored in pair 2. Empty cells are colonised with weight $\max(0.2,\, \hat{\mathbf{r}}_{\text{toward}}\cdot\hat{\mathbf{d}}_{\text{parent}})$ so growth is forward-biased. New tips inherit parent direction plus a branching jitter. **Anastomosis**: tips meeting from opposing directions ($\hat{\mathbf{d}}_i\cdot\hat{\mathbf{d}}_j < -0.2$) fuse, producing real network loops rather than pure trees.

---

### Oscillator / Synchronization

**Kuramoto 3D** — Coupled phase oscillators on a 3D lattice with Hebbian frequency adaptation:

$$\dot\theta_i = \omega_i\Omega + \frac{K}{26}\sum_{j\in\mathcal{N}_{26}}\sin(\theta_j - \theta_i) + \eta_i, \qquad \dot\omega_i = \lambda R_i(\langle\omega\rangle_\mathcal{N} - \omega_i)$$

---

### Phase-Field / Materials

**Crystal Growth (Kobayashi)** — Anisotropic phase-field with cubic harmonics on the interface normal:

$$\partial_t \phi = \beta(\hat{n})^2\nabla^2\phi + 30\beta^2\phi(1-\phi)\!\left(\phi - \tfrac{1}{2} + \Delta + \tfrac{u}{2}\right)$$
$$\partial_t u = D\nabla^2 u - \tfrac{1}{2}\partial_t\phi$$

The shape mode parameter selects the anisotropy form via the cubic harmonic $K_4 = n_x^4 + n_y^4 + n_z^4$:

$$\beta(\hat{n}) = 1 + \varepsilon\,a(\hat{n}), \qquad a(\hat{n}) = \begin{cases} +(K_4 - 3/5) & \text{compact / octahedral / dendritic (} \beta \text{ max at } \langle 100 \rangle \text{ axes)} \\\\ -(K_4 - 3/5) & \text{cubic (} \beta \text{ max at } \langle 111 \rangle \text{ corners)} \end{cases}$$

Five presets cover the regime: **Compact** ($\varepsilon \approx 0$, rounded), **Octahedral** ($\varepsilon{=}2$, axial bulges), **Cubic** (inverted anisotropy, corner bulges), **Dendritic** (octahedral + temporal interface noise drives Mullins–Sekerka tip-splitting), and **Snowflake** (strong $\varepsilon$, low $D$ → fragile filaments).

**Cahn-Hilliard** — Spinodal decomposition via fourth-order diffusion of the chemical potential:

$$\mu = c^3 - c + \alpha_{\text{asym}} - \varepsilon^2\nabla^2 c, \qquad \partial_t c = M\nabla^2\mu$$

**Fracture (bond-based peridynamics)** — Each voxel carries displacement $\mathbf{u}$ and integrity $\chi\in\{0,1\}$ (pair 1) plus velocity $\mathbf{v}$ and max-stretch history (pair 2). The force on a voxel sums over its 26 bonds to neighbours:

$$\mathbf{F}_i = \sum_{j \in \mathcal{N}_{26}} c\,\chi_i \chi_j \, s_{ij}\,\hat{\boldsymbol{\xi}}_{ij}/|\boldsymbol{\xi}_{ij}|, \qquad s_{ij} = (\mathbf{u}_j - \mathbf{u}_i)\cdot\hat{\boldsymbol{\xi}}_{ij}/|\boldsymbol{\xi}_{ij}|$$

Integrity is irreversibly broken once any of the 26 bond stretches exceeds the critical value $s_c$. Loading is **quasi-static**: the initial condition pre-loads a linear strain field matching the displacement-controlled grip BCs, so no elastic shock is launched. A V-notch on the $-X$ face provides the geometric stress concentration that nucleates a clean Mode-I crack.

**Erosion (channelised hydraulic)** — Water flows down the surface-height gradient $H = \text{solid} + \text{fluid}$ rather than diffusing isotropically, and sediment transport uses a Hjulström-style carry capacity:

$$C = K_c v^2; \qquad \dot{s} = \begin{cases} K_e(C - s) & C > s\ \text{(erode)} \\\\ K_d(C - s) & C < s\ \text{(deposit)} \end{cases}$$

The positive deposition feedback (slow water drops sediment, raising the bed and slowing flow further) is what carves persistent channels rather than smoothing terrain into a single flat plain.

---

### Fluid / Transport

**Viscous Fingers (Saffman-Taylor)** — Two-phase porous flow with viscosity contrast $M = \mu_d/\mu_i$. Pressure is solved by multi-pass Jacobi with **harmonic-mean** face mobilities $\bar\lambda_f = 2\lambda\lambda_{\text{nb}}/(\lambda + \lambda_{\text{nb}})$ (preserves flux continuity at a viscosity jump), then transport uses donor-cell upwinding on the fractional-flow function:

$$\nabla\!\cdot(\bar\lambda \nabla p) = 0, \qquad f(s) = \frac{s/\mu_i}{s/\mu_i + (1-s)/\mu_d}, \qquad \partial_t s + \nabla\!\cdot(f(s)\,\mathbf{v}_{\text{Darcy}}) = 0$$

Geometry is a planar injector slab → planar drain slab, with sub-voxel interface roughness and heterogeneous permeability seeding the Saffman–Taylor instability into branching fingers.

**Fire (combustion fluid)** — Two-pass shader: chemistry pass writes $(T, \text{soot}, \text{ember}, \text{fuel})$, flow pass writes $(v_x, v_y, v_z, O_2)$. Chemistry uses Arrhenius-gated combustion, thermal conduction, $T^4$ radiative cooling, and ember spawn/decay. Flow uses semi-Lagrangian advection, buoyancy, viscous diffusion, and Steinhoff–Underhill **vorticity confinement** to keep flames from numerically diffusing into a uniform plume:

$$\mathbf{f}_{\text{vc}} = \varepsilon\,h\,\hat{\mathbf{N}} \times \boldsymbol{\omega}, \qquad \hat{\mathbf{N}} = \nabla|\boldsymbol{\omega}| / \|\nabla|\boldsymbol{\omega}|\|$$

---

### Astrophysics

**Galaxy Formation** — Self-gravity from a Jacobi-relaxed Poisson solve $\nabla^2 \Phi = 4\pi G(\rho - \bar\rho)$ (mean-subtracted source for periodic boundaries), feeding semi-Lagrangian advection of density and momentum:

$$\partial_t \mathbf{v} = -\nabla\Phi - P\nabla\rho/\rho + \nu\nabla^2 \mathbf{v}, \qquad \partial_t \rho + \nabla\!\cdot(\rho\mathbf{v}) = 0$$

The potential $\Phi$ is warm-started across timesteps in pair 2, so only a handful of Jacobi sweeps per logical step are needed for convergence. Initial filament-like density perturbations collapse into a cosmic web.

---

### Chemistry

**Element CA** — Particle-based system with all 118 elements, reaction rules, phase transitions, and electronegativity-driven bonding (separate shader and data file).

---

## Screenshots

| 64³ grid | 512³ grid |
|----------|-----------|
| ![3D Game of Life at 64³](screenshots/3d-ca-sim-voxel-3dgol-01.png) | ![3D Game of Life at 512³](screenshots/3d-ca-sim-voxel-3dgol-02.png) |

*3D Game of Life (B6-7/S5-7) — voxel renderer with surface neighbor count coloring. 11.5M visible voxels at 512³.*

## License

MIT
