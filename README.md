# 3D Cellular Automata

A GPU-accelerated 3D cellular automata simulator with real-time volumetric ray marching, 27 compute shader rules, and 49 presets — from classic Game of Life to quantum mechanics.

![OpenGL 4.3](https://img.shields.io/badge/OpenGL-4.3%20Compute-blue)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow)

> **Work in progress** — experimental voxel and volumetric CAs. Expect rough edges.

## Features

### Simulation
- **27 GPU compute shader rules** with 49 built-in presets
- **Grid sizes 32³ to 512³** with dynamic resizing (auto-switches rgba32f → rgba16f at large sizes)
- **Resolution-independent physics** — h² Laplacian scaling and h⁻¹ gradient scaling keeps behavior consistent across grid sizes
- **62 initialization patterns** — sparse/dense random, centered blobs, crystal seeds, orbital wavefunctions, terrain, and more
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
- **Multi-channel visualization** — select which field to display (e.g., prey vs predator, real vs imaginary)
- **Half-resolution mode** — 2–4× faster volumetric rendering with catmull-rom upsampling
- **Compute raymarcher** — optional shared-memory cached ray-marching shader with occupancy-grid brick DDA (matches fragment path)
- **Adaptive brick DDA** — occupancy grid block edge is 4³ (small grids) or 8³ (≥192³) for fewer DDA steps and lower cache pressure
- **R16F view texture** — channel-select and abs are baked into a single half-precision texture; ray-marcher reads 2 B/voxel instead of 16 B, halving ray-march bandwidth at large grid sizes
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
| **Reaction-Diffusion** | Gray-Scott, BZ, Barkley, Morphogen | Chemical pattern formation — spots, spirals, waves |
| **Continuous** | Lenia, Multi-channel Lenia | Kernel-based continuous CAs with lifelike organisms |
| **Physics** | Wave, EM Wave, Schrödinger (×6) | Wave equations, quantum mechanics, tunneling, orbitals |
| **Materials** | Crystal Growth (×5: Compact, Octahedral, Cubic, Dendritic, Snowflake), Cahn-Hilliard, Fracture, Erosion | Phase separation, nucleation, elastic fracture |
| **Biology** | Predator-Prey, Flocking, Physarum, Mycelium, Lichen | Population dynamics, swarm behavior, fungal networks |
| **Astrophysics** | Galaxy Formation | N-body gravity with gas dynamics |
| **Chemistry** | Element CA, Viscous Fingers, Fire | 118-element particle chemistry, fluid instabilities, combustion |

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
simulator.py         — Main simulator: GLSL shaders, rendering, UI (~10 300 lines)
element_data.py      — Periodic table data for the Element Chemistry rule
test_harness.py      — Headless parameter sweep and discovery engine
```

> `discoveries.json` and the batch search scripts are personal workflow tooling and are gitignored; they are not part of the repository.

All 27 compute shaders are embedded in `simulator.py` as GLSL source strings, compiled at runtime via moderngl. The rendering pipeline uses a separate ray marching fragment shader with emission-absorption volume integration.

### Compute Shader Design

Each rule is a GLSL compute shader dispatched over an 8×8×8 workgroup grid. A shared `COMPUTE_HEADER` provides:

- Ping-pong textures (`u_src` / `u_dst`) as 3D `image3D` bindings
- Resolution-independent scaling via `h_sq` and `h_inv` (referenced to 128³)
- Boundary handling (toroidal wrap, clamp, or mirror/Neumann)
- Utility functions: `fetch()`, `fetch_interp()`, `hash_temporal()`, `lap19_v4()` (19-point isotropic Laplacian)

Shaders that use large neighborhood radii (SmoothLife, Lenia) have an optimized shared memory tiling path with automatic fallback for drivers that don't support it. Most reaction-diffusion shaders (Gray-Scott, BZ, Barkley, Gierer-Meinhardt, Cahn-Hilliard, predator-prey, lichen, fracture, wave) use a 19-point Patra-Karttunen stencil whose leading error ∝ ∇²(∇²f) is rotationally symmetric — the cheaper 6-point stencil's anisotropic Σ ∂⁴/∂xi⁴ error visibly distorts spirals and droplets toward grid axes.

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

**EM Wave** — TE-like Maxwell FDTD: $E_z$ driven by curl of $(B_x, B_y)$, magnetic fields updated by curl of $E_z$, with conductor absorption.

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

**Flocking (Vicsek)** — Active matter with velocity alignment, pressure repulsion, and semi-Lagrangian density advection:

$$\mathbf{v}^{t+1} = (1-\alpha\Delta t)\mathbf{v}^t + \alpha\Delta t\,v_0(1+2\bar\rho)\hat{\mathbf{v}}_{\text{avg}} - \kappa(\bar\rho - 0.3)\nabla\rho\;\Delta t$$

**Physarum** — Slime mold chemotaxis: agents follow trail gradients via semi-Lagrangian advection, depositing and evaporating pheromone.

**Mycelium** — Agent-based fungal network: tip extension with nutrient-gradient-biased branching, anastomosis, and starvation death.

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

**Fracture** — Elastic wave propagation with irreversible integrity loss when $|\sigma| > \sigma_c(1 - 0.03 N_{\text{broken}})$.

**Erosion** — Hydraulic erosion with gravity-driven fluid, shear-rate erosion, and velocity-dependent deposition.

---

### Fluid / Transport

**Viscous Fingers (Saffman-Taylor)** — Pressure-driven invasion with viscosity-dependent mobility:

$$p_{ijk} \leftarrow (1-\omega)p_{ijk} + \omega\frac{\sum_{\text{nn}}\bar\lambda\,p_{\text{nn}}}{\sum_{\text{nn}}\bar\lambda}$$

$$\partial_t S = \lambda\xi\sum_{\text{nn}}\max(0, p_{\text{nn}} - p)(S_{\text{nn}} - S) + \gamma\nabla^2 S$$

**Fire** — Combustion front with temperature diffusion, fuel consumption, oxygen transport, and rising embers.

---

### Astrophysics

**Galaxy Formation** — Multi-scale self-gravity with semi-Lagrangian advection:

$$\mathbf{F}_g = G\rho\!\left(\nabla\rho\big|_1 + \tfrac{1}{2}\nabla\rho\big|_2 + \tfrac{1}{4}\nabla\rho\big|_4\right)$$

$$\partial_t\mathbf{v} = \mathbf{F}_g - P\nabla\rho/\rho + \tfrac{\nu}{2}\nabla^2\mathbf{v} + \Lambda\hat{\mathbf{r}}\cdot 0.01$$

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
