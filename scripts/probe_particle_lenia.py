"""Headless probe for Particle Lenia (force_mode==6).

Drives the REAL GPU pipeline (Simulator._step_sim) headless, reads the
particle SSBO back, and computes structure metrics so we can tell
"noise / gas" apart from "metastable clusters / crystal" without
needing to watch it render.

Metrics per snapshot:
  speed      mean particle speed (|v|)            -> frozen vs active
  spread     std of positions / box               -> collapsed vs filling
  nn1        median nearest-neighbour distance     -> lattice spacing
  nn_cv      CV of nn1 distances (std/mean)        -> LOW = regular (crystal)
  g1         height of first RDF peak (vs ideal gas)-> >1.5 = real shell
  shells     # of RDF peaks above 1.2              -> crystal has several
  nclust     # connected components @1.5*nn1       -> fragmentation

Usage:
  python scripts/probe_particle_lenia.py                 # baseline
  python scripts/probe_particle_lenia.py --sweep growth  # sweep Growth mu
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from simulator import Simulator  # noqa: E402


def min_image_pdist(P, L):
    """All pairwise min-image distances (toroidal box L). Returns the
    flat upper-triangle vector and the (N,N) matrix nn-sorted per row."""
    d = P[:, None, :] - P[None, :, :]
    d -= L * np.round(d / L)
    r = np.sqrt((d * d).sum(-1))
    return r


def metrics(P, V, L):
    N = len(P)
    speed = float(np.linalg.norm(V, axis=1).mean())
    spread = float(P.std(0).mean() / L)
    r = min_image_pdist(P, L)
    np.fill_diagonal(r, np.inf)
    nn1 = r.min(1)
    nn1_med = float(np.median(nn1))
    nn_cv = float(nn1.std() / (nn1.mean() + 1e-9))
    # RDF up to L/2
    rmax = L / 2.0
    iu = np.triu_indices(N, 1)
    rr = r[iu]
    rr = rr[rr < rmax]
    nb = 80
    hist, edges = np.histogram(rr, bins=nb, range=(0.0, rmax))
    centres = 0.5 * (edges[1:] + edges[:-1])
    shell_vol = 4.0 * np.pi * centres**2 * (edges[1] - edges[0])
    rho = N / (L**3)
    ideal = 0.5 * N * rho * shell_vol  # expected pairs per shell (ideal gas)
    g = hist / np.maximum(ideal, 1e-9)
    # first peak height (skip r< small noise)
    valid = centres > max(nn1_med * 0.4, 0.5)
    g1 = float(g[valid][:20].max()) if valid.any() else 0.0
    shells = int((g[valid] > 1.2).sum())
    # connected components at 1.5*nn1
    thr = 1.5 * nn1_med
    adj = r < thr
    nclust = _count_components(adj)
    return dict(speed=speed, spread=spread, nn1=nn1_med, nn_cv=nn_cv,
                g1=g1, shells=shells, nclust=nclust)


def _count_components(adj):
    N = len(adj)
    seen = np.zeros(N, bool)
    comps = 0
    for s in range(N):
        if seen[s]:
            continue
        comps += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            nb = np.nonzero(adj[u] & ~seen)[0]
            seen[nb] = True
            stack.extend(nb.tolist())
    return comps


def read_particles(sim):
    raw = np.frombuffer(sim.particle_ssbo.read(), dtype=np.float32)
    rec = raw.reshape(-1, 8)
    return rec[:, 0:3].copy(), rec[:, 4:7].copy()


def run(sim, params, steps=600, snaps=(50, 150, 300, 600), seed=1, init=None):
    # Fresh particle field each run (one GL context, reused).
    sim.seed = seed
    if init is not None:
        sim.preset['particle_init'] = init
    sim._release_particles()
    sim._alloc_particles()
    sim.step_count = 0
    for k, v in params.items():
        sim.params[k] = v
    L = float(sim.size)
    out = {}
    snapset = set(snaps)
    for n in range(1, steps + 1):
        sim._step_sim()
        if n in snapset:
            P, V = read_particles(sim)
            out[n] = metrics(P, V, L)
    return out


def fmt(m):
    return (f"speed={m['speed']:6.3f} spread={m['spread']:.3f} "
            f"nn1={m['nn1']:5.2f} nn_cv={m['nn_cv']:.3f} "
            f"g1={m['g1']:5.2f} shells={m['shells']} nclust={m['nclust']}")


DEFAULTS = {"Kernel μ": 4.0, "Kernel σ": 1.0, "Kernel w": 0.1,
            "Growth μ": 1.5, "Growth σ": 1.0, "Repulsion": 1.0,
            "Repulse R": 1.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["none", "growth", "kernel", "rep", "wk"],
                    default="none")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--init", default=None,
                    help="particle_init override (e.g. random_volume)")
    ap.add_argument("--wk", type=float, default=None)
    ap.add_argument("--growth", type=float, default=None)
    args = ap.parse_args()

    sim = Simulator(size=64, rule="particle_lenia", headless=True)

    if args.sweep == "none":
        d = dict(DEFAULTS)
        if args.wk is not None:
            d["Kernel w"] = args.wk
        if args.growth is not None:
            d["Growth μ"] = args.growth
        print("BASELINE", d, "init=", args.init or "centered_blob")
        snaps = (50, 100, 200, 400, 600, 900)
        res = run(sim, d, steps=args.steps, snaps=snaps, init=args.init)
        for n, m in res.items():
            print(f"  step {n:4d}: {fmt(m)}")
        return

    grids = {
        "growth": ("Growth μ", [0.5, 1.0, 2.0, 4.0, 8.0]),
        "kernel": ("Kernel μ", [2.0, 3.0, 4.0, 6.0]),
        "rep": ("Repulsion", [0.3, 1.0, 2.0, 4.0]),
        "wk": ("Kernel w", [0.02, 0.05, 0.1, 0.2, 0.4]),
    }
    key, vals = grids[args.sweep]
    print(f"SWEEP {key}: {vals}  (final step {args.steps})")
    for v in vals:
        p = dict(DEFAULTS)
        p[key] = v
        res = run(sim, p, steps=args.steps, snaps=(args.steps,))
        m = res[args.steps]
        print(f"  {key}={v:6.2f}: {fmt(m)}")


if __name__ == "__main__":
    main()
