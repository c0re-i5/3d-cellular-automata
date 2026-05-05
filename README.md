# 3D Cellular Automata

A GPU-accelerated 3D cellular automata simulator with real-time volumetric ray marching, ~40 distinct compute shaders, and 59 built-in presets — from classic Game of Life to quantum mechanics, peridynamic fracture, and Saffman–Taylor viscous fingering.

![OpenGL 4.3](https://img.shields.io/badge/OpenGL-4.3%20Compute-blue)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow)

> **Work in progress** — experimental voxel and volumetric CAs. Expect rough edges.

## Features

### Simulation
- **~40 GPU compute shaders** powering 59 built-in presets
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
- **MP4 video capture** at 2560×1440 60fps via ffmpeg (H.264)
- **Text overlays** burned into video — title, description, parameters, seed, grid size, discovery score
- **JSON metadata sidecar** — full parameter snapshot alongside each recording
- **Background writer thread** — 30-frame queue to avoid blocking the main loop
- **Blinking REC indicator** in the UI during capture
- **Opt-in only** — recording requires `CA_RECORDING_ENABLED=1` in the environment; F5 is a no-op without it

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
| **Physics** | Wave, EM Wave (Yee FDTD), Schrödinger (×6) | Wave equations, full-vector electromagnetics, quantum mechanics |
| **Materials** | Crystal Growth (×5: Compact, Octahedral, Cubic, Dendritic, Snowflake), Cahn-Hilliard, Fracture (peridynamic), Erosion (hydraulic) | Phase separation, anisotropic solidification, brittle fracture, channelised erosion |
| **Biology** | Predator-Prey, Flocking (Reynolds + predator), Physarum (adaptive flux), Mycelium (anastomosing), Lichen | Population dynamics, swarm behaviour, biological transport networks |
| **Astrophysics** | Galaxy Formation | Self-gravity (Jacobi-Poisson) + semi-Lagrangian gas dynamics |
| **Chemistry** | Element CA, Viscous Fingers (Saffman–Taylor), Fire (combustion fluid) | 118-element particle chemistry, two-phase porous flow, reactive flow with vorticity confinement |

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

## Architecture

```
simulator.py         — Main simulator: GLSL shaders, rendering, UI (~26 700 lines)
element_data.py      — Periodic table data for the Element Chemistry rule
test_harness.py      — Headless parameter sweep and discovery engine (~4 200 lines)
snapshot_3d.py       — Headless renderer + multi-channel auditor (PNG strips, channel-utilisation reports)
```

> `discoveries.json` and the batch search scripts are personal workflow tooling and are gitignored; they are not part of the repository.

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

### Multi-channel field layout

Every rule's primary state lives in an RGBA `image3D` (4 channels per voxel), so even single-state automata like Game of Life have three "free" channels. Many discrete rules now compute auxiliary diagnostic fields and write them into ch1–ch3 every step, at near-zero cost (the values were already in registers from the main update). The renderer picks them up automatically through the `vis_mode` system.

The dispatcher writes back the *same* texture format, so:
- **Voxel mode** still reads only `.r` (state) — auxiliary channels are ignored, no behaviour change
- **Volumetric mode** lets the per-preset `vis_mode` choose how to reduce ch0–ch3 into a colour
- **Discovery scoring / GPU metrics** still operate on `.r`; auxiliaries are visualisation-only

A condensed reference for what the auxiliary channels mean per rule (see source for full list):

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
