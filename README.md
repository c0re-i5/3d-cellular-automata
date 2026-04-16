# 3D Cellular Automata

A GPU-accelerated 3D cellular automata simulator with real-time volumetric ray marching, 26 compute shader rules, and 46 presets — from classic Game of Life to quantum mechanics.

![OpenGL 4.3](https://img.shields.io/badge/OpenGL-4.3%20Compute-blue)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow)

> **Work in progress** — experimental voxel and volumetric CAs. Expect rough edges.

## Features

- **26 GPU compute shader rules** running on OpenGL 4.3 compute shaders
- **Real-time volumetric rendering** via ray marching with emission-absorption model
- **46 built-in presets** with tunable parameters
- **Interactive UI** with imgui — adjust parameters, camera, rendering in real-time
- **Grid sizes from 32³ to 512³** with automatic format switching (rgba32f → rgba16f)
- **Resolution-independent physics** — h² Laplacian scaling keeps behavior consistent across grid sizes
- **Recording system** — capture parameter snapshots + MP4 video
- **Headless test harness** for automated parameter sweeps and discovery search

### Rule Categories

| Category | Rules | Description |
|----------|-------|-------------|
| **Classic** | Game of Life, SmoothLife | Discrete and continuous life-like automata |
| **Reaction-Diffusion** | Gray-Scott, BZ, Barkley, Morphogen | Chemical pattern formation — spots, spirals, waves |
| **Continuous** | Lenia, Multi-channel Lenia | Kernel-based continuous CAs with lifelike organisms |
| **Physics** | Wave, EM Wave, Schrödinger (×6) | Wave equations, quantum mechanics, tunneling, orbitals |
| **Materials** | Crystal Growth, Cahn-Hilliard, Fracture, Erosion | Phase separation, nucleation, elastic fracture |
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
| **F** | Toggle fullscreen |
| **Tab** | Cycle through presets |
| **1-4** | Switch rendering mode |
| **Mouse drag** | Rotate camera |
| **Scroll** | Zoom |

### Headless Testing

```bash
python test_harness.py
```

Runs automated parameter sweeps, scores results by interestingness metrics, and saves discoveries to `discoveries.json`.

## Architecture

```
simulator.py       — Main simulator: GLSL shaders, rendering, UI (~8800 lines)
element_data.py    — Periodic table data for the Element Chemistry rule
test_harness.py    — Headless parameter sweep and discovery engine
discoveries.json   — Saved interesting parameter configurations
```

All 26 compute shaders are embedded in `simulator.py` as GLSL source strings, compiled at runtime via moderngl. The rendering pipeline uses a separate ray marching fragment shader with emission-absorption volume integration.

### Compute Shader Design

Each rule is a GLSL compute shader dispatched over an 8×8×8 workgroup grid. A shared `COMPUTE_HEADER` provides:

- Ping-pong textures (`u_src` / `u_dst`) as 3D `image3D` bindings
- Resolution-independent scaling via `h_sq` and `h_inv` (referenced to 128³)
- Boundary handling (toroidal wrap or clamp)
- Utility functions: `fetch()`, `fetch_interp()`, `hash_temporal()`

Shaders that use large neighborhood radii (SmoothLife, Lenia) have an optimized shared memory tiling path with automatic fallback for drivers that don't support it.

## Screenshots

*Coming soon — contributions welcome!*

## License

MIT

## Tools

- `explorer.html` — interactive multi-state 1D CA explorer (zero dependencies)
- `validator.html` — extended PRNG validation battery (15+ statistical tests)
- `ca_lab.py` — shared continuous CA engine: 7 rule families, simulation, 10+ measurements, Wolfram classification, interestingness scoring
- `phase_diagram.py` — 2D parameter sweeps, phase boundary detection, sweet spot finding, class/metric heatmaps
- `continuous_ca.py` — bifurcation diagrams, attractor type analysis (fixed/periodic/quasiperiodic/chaotic), cell trajectories
- `rule_space.py` — PCA of behavioral fingerprints, clustering, cross-family comparison, outlier detection
- `coevolution.py` — open-ended evolution: cells carry state + genome defining local rule, fitness-driven competition, speciation tracking
- `temporal_ca.py` — temporal-derivative CAs: each cell has state + velocity, 5 physics modes (spring/resonant/damped/threshold/wave)
- `adaptive_neighborhood.py` — adaptive neighborhoods: radius = f(cell state), 4 radius modes, emergent domain formation
- `causal_ca.py` — self-referential CAs: run CA → extract perturbation-based influence graph → run CA on graph → iterate
- `chemical_ca.py` — multi-species chemistry: concentration vectors with Brusselator, Gray-Scott, autocatalytic RPS, random chemistry search
- `async_ca.py` — asynchronous deterministic CAs: variable clock speed from energy, 4 energy modes, relativistic effects
- `number_ca.py` — number-theoretic lattices: neighborhoods from divisors, prime offsets, shared factors, modular arithmetic, Collatz

## Rule Families

| Family | Parameters | Description |
|--------|-----------|-------------|
| `weighted_threshold` | weight, theta | Weighted neighbor average + hard threshold |
| `logistic_coupling` | r, w | Logistic map with spatial coupling (chaos at r≈3.57) |
| `gaussian_kernel` | mu, sigma | Gaussian response peaked at target neighbor average |
| `reaction_diffusion` | D, R | Bistable reaction-diffusion with cubic reaction term |
| `asymmetric_wave` | a, decay | Left-biased signal propagation with decay |
| `fuzzy_totalistic` | birth, survive | Continuous analog of birth/survival rules |
| `sin_interference` | freq, phase | Sinusoidal interference patterns |

## Discoveries

### Phase transitions in reaction-diffusion CAs
The `reaction_diffusion` family has the richest phase structure of all 7 families:
- **491 phase boundary points** at 30×30 resolution (900 parameter combinations)
- All 4 Wolfram classes present: Fixed 40.9%, Periodic 23.3%, Chaotic 19.4%, **Complex 16.3%**
- Clear diagonal boundary in (D, R) space — low diffusion + low reaction = chaotic, high diffusion = periodic/fixed, the complex class (edge of chaos) lives at the hinge
- Top sweet spot: D≈0.48, R=5.0 — Complex class with entropy=0.422, Lyapunov=-0.070

### Period-doubling cascade in logistic coupling
The `logistic_coupling` family reproduces the classic period-doubling route to chaos:
- Bifurcation diagram shows fixed → period-2 → period-4 → chaos as r increases
- Edge of chaos confirmed at r≈3.57 (Lyapunov crosses zero)
- 41 distinct behavioral clusters in PCA space

### Families are genuinely different
Cross-family PCA scatter shows each rule family occupies a distinct region of behavioral space:
- `asymmetric_wave`: highest average entropy (0.903), consistently high structure
- `logistic_coupling`: highest interestingness (0.634), richest bifurcation behavior
- `weighted_threshold`: lowest activity (0.016), mostly dead/fixed
- Gaps between family clusters indicate **unexplored behavioral regimes** — potential targets for novel rule design

### Most unique rules discovered
Behaviorally isolated rules (far from everything else in measurement space):
- `gaussian_kernel(mu=0.00, sigma=0.21)` — nn_dist=0.756, Complex class
- `logistic_coupling(r=2.50, w=1.00)` — nn_dist=0.548, Periodic
- `logistic_coupling(r=3.40, w=1.00)` — nn_dist=0.541, Complex
- `sin_interference(freq=0.50, phase=0.40)` — nn_dist=0.511, Complex

### Coevolutionary rule ecology
Open-ended evolution with cells carrying both state and a 4-parameter genome (weight, threshold, steepness, asymmetry):
- `edge` fitness mode produces the richest ecologies (43 species at equilibrium)
- Long runs show **423 speciation events** in 1000 steps, species count oscillating 34–106
- Clear diversity-stability tradeoff: mutation rate 0.01→15 species, 0.50→88 species
- Species count fluctuations suggest the system operates at self-organized criticality

### Temporal-derivative CAs — wave physics from rate dependence
Rules that depend on velocity (rate of change) rather than just current state:
- **Spring physics** produces expanding pulse rings at ~0.38 cells/step with memory=6
- **Threshold mode** shows energy AMPLIFICATION (9.4× initial energy) — counter-intuitive
- **Damped mode** has longest temporal memory (41 steps autocorrelation)
- **Wave mode** preserves ~25% of energy with ballistic propagation

### Adaptive neighborhoods — emergent geometry
Neighborhood radius determined by cell state creates self-reinforcing spatial structure:
- **Bistable rule** creates strong domain formation: cells snap to 0 or 1 with clear gap boundaries
- High-state cells have large radius (more influence) → domains self-reinforce
- Domain formation completes in ~3 steps — extremely fast phase separation

### Self-referential causal CAs
Run CA → extract who-influences-whom → run CA on that influence graph → repeat:
- Rule 30 influence graph: avg_degree=7.5, clustering=0.346, **71.8% long-range connections**
- Graph structure measurably changes across iterations (clustering drifts 0.346→0.321)
- Activity drops sharply on first graph iteration (0.491→0.063) — the causal structure acts as a filter

### Multi-species chemical CAs
Concentration vectors with explicit reaction kinetics:
- **Brusselator**: settles to steady state (X=0.388, Y=4.056) in 1D
- **Gray-Scott**: forms localized activator spot (U=0.891, V=0.045)
- **Autocatalytic RPS**: all 3 species coexist with spatial concentration waves, near-perfect symmetry (A=0.307, B=0.319, C=0.323)

### Asynchronous deterministic CAs
Variable clock speed with no randomness — purely deterministic timing:
- **Activity-driven energy** creates self-reinforcing fast zones: active regions gain energy, update more, stay active
- **Conservative energy** produces energy waves — computation "sloshes" between regions
- Critical recharge rate threshold: below it, system freezes; above it, synchronous behavior returns
- At recharge=1.0, activity=0.473 (synchronous regime); at recharge=0.1, activity=0.001 (frozen)

### Number-theoretic lattice CAs — CAs that discover primes
Neighborhoods defined by number theory rather than spatial proximity:
- **Factor neighborhoods achieve 0.4355 prime-composite separation** — the CA genuinely distinguishes primes from composites
- Primes are isolated (few divisor-neighbors), highly composite numbers are hubs
- Divisor correlation of −0.67 shows final state strongly anti-correlates with divisor count
- Modular neighborhoods (mod 3): 0.088 separation; separation decreases with larger moduli
- Collatz neighborhoods create unique tree-like topology encoding trajectory structure
