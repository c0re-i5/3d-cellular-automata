"""Batch-debug audit: rich per-snapshot stats across many seeds + parameter sweeps.

Equivalent to capturing the GUI's debug-stats history (channel histograms, COM,
rg, NaN/Inf counts, mass-conservation drift, growth rate, etc.) but headless
and parallel across (rule, seed, parameter sweep value) cells.

Two output products:
 1. Per-trial JSON dumps in `debug_runs/batch_<timestamp>/<rule>__seed<S>__U<v>.json`
    — same schema as the live GUI's debug_runs/*.json so the existing
    `compare_growth_dendritic.py` / explorer.html viewers work unchanged.
 2. Roll-up summary printed to stdout: per-trial trajectories of the
    most discriminating metrics, plus seed-variance analysis to show
    which patterns are robust vs. seed-dependent.

Usage:
    .venv/bin/python scripts/batch_debug_audit.py \
        --rules crystal_cubic,crystal_octahedral,crystal_polycrystal \
        --seeds 42,7,123,256,9999 \
        --undercoolings 0.3,0.5,0.7,0.9 \
        --size 64 --steps 1500 --interval 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from test_harness import create_headless_context, HeadlessRunner  # noqa: E402
from simulator import RULE_PRESETS  # noqa: E402

CRYSTAL_RULES = [r for r in RULE_PRESETS if r.startswith('crystal_')]


# ── Per-snapshot stats (numpy mirror of GPU DEBUG_STATS_SHADER) ───────────────

def snapshot_stats(grid: np.ndarray, step: int, t_wall: float,
                   prev_alive_count: int | None = None,
                   active_threshold: float = 0.5) -> dict:
    """Compute the same per-snapshot dict the GUI's _harvest_debug_stats writes,
    plus a few extras (per-channel quartiles, growth rate, mass)."""
    sz = grid.shape[0]
    N = grid.size // 4

    # Per-channel scalars
    finite = []
    nan_count = []
    inf_count = []
    mins, maxs, means, stds, vars_ = [], [], [], [], []
    quartiles = []  # [p10, p50, p90] per channel
    for c in range(4):
        v = grid[..., c]
        nan_mask = np.isnan(v)
        inf_mask = np.isinf(v)
        n_nan = int(nan_mask.sum())
        n_inf = int(inf_mask.sum())
        fin = v[~nan_mask & ~inf_mask]
        finite.append(int(fin.size))
        nan_count.append(n_nan)
        inf_count.append(n_inf)
        if fin.size > 0:
            mn, mx = float(fin.min()), float(fin.max())
            mu = float(fin.mean())
            va = float(fin.var())
            mins.append(mn); maxs.append(mx); means.append(mu)
            vars_.append(va); stds.append(float(math.sqrt(va)))
            qs = np.quantile(fin, [0.1, 0.5, 0.9])
            quartiles.append([float(qs[0]), float(qs[1]), float(qs[2])])
        else:
            mins.append(float('nan')); maxs.append(float('nan'))
            means.append(float('nan')); vars_.append(float('nan'))
            stds.append(float('nan'))
            quartiles.append([float('nan')] * 3)

    # Histogram per channel (64 bins, range = current min..max)
    hist = []
    hist_min = []
    hist_max = []
    for c in range(4):
        v = grid[..., c]
        fin = v[np.isfinite(v)]
        if fin.size == 0 or maxs[c] - mins[c] < 1e-9:
            hist.append([0] * 64)
            hist_min.append(mins[c])
            hist_max.append(maxs[c] if not math.isnan(maxs[c]) else 0.0)
            continue
        h, edges = np.histogram(fin, bins=64, range=(mins[c], maxs[c]))
        hist.append([int(x) for x in h])
        hist_min.append(float(edges[0]))
        hist_max.append(float(edges[-1]))

    # Spatial: alive mask on channel 0
    alive_mask = grid[..., 0] > active_threshold
    active_count = int(alive_mask.sum())
    out_spatial = {}
    envelope_octa = float('nan')
    if active_count > 0:
        z, y, x = np.nonzero(alive_mask)
        cx, cy, cz = float(x.mean()), float(y.mean()), float(z.mean())
        rg_vox = float(np.sqrt(((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2).mean()))
        bbmin = [int(x.min()), int(y.min()), int(z.min())]
        bbmax = [int(x.max()), int(y.max()), int(z.max())]
        com = [cx / sz, cy / sz, cz / sz]
        rg = rg_vox / sz
        # boundary count: alive cells within 4 of any wall
        shell = 4
        bd = (
            (x < shell) | (x >= sz - shell) |
            (y < shell) | (y >= sz - shell) |
            (z < shell) | (z >= sz - shell)
        )
        boundary_count = int(bd.sum())
        # Envelope shape: ratio of max axis-reach to max diagonal-reach.
        # axis-reach = furthest alive cell along ±x, ±y, or ±z from COM.
        # diag-reach = furthest alive cell along the 4 body-diagonals.
        # Octahedron tips along axes ⇒ ratio > 1.
        # Cube corners along diagonals ⇒ ratio < 1.
        # Sphere ⇒ ratio ≈ 1.
        dx = x - cx; dy = y - cy; dz = z - cz
        axis_reach = max(
            float(np.abs(dx).max()),
            float(np.abs(dy).max()),
            float(np.abs(dz).max()),
        )
        # Project onto the 4 body-diagonal unit vectors
        inv_sqrt3 = 1.0 / math.sqrt(3.0)
        diag_reach = max(
            float(np.abs((dx + dy + dz) * inv_sqrt3).max()),
            float(np.abs((dx + dy - dz) * inv_sqrt3).max()),
            float(np.abs((dx - dy + dz) * inv_sqrt3).max()),
            float(np.abs((-dx + dy + dz) * inv_sqrt3).max()),
        )
        if diag_reach > 0:
            envelope_octa = axis_reach / diag_reach

        # BRANCHINESS metric: coefficient of variation of distance-from-COM
        # across alive cells. For a sphere, all alive cells are at roughly
        # the same distance from the COM (= radius), so std/mean is near 0.
        # For a branched shape, alive cells span everything from COM
        # (interior) out to the dendrite tips, so std/mean is large.
        # Voxel-staircase artefacts that contaminated the surface-area
        # version cancel out here because we measure radial spread, not
        # surface count.
        #   solid sphere (r constant inside? no — uniform fill from 0..r):
        #     dist uniform on [0, r] → std/mean = (r/√12)/(r/2) = 1/√3 ≈ 0.577 ... actually
        #   for a 3D ball with uniform density, mean dist = (3/4)r,
        #     E[d²] = (3/5)r², var = 3r²/80, std ≈ 0.194 r,
        #     std/mean ≈ 0.194/0.75 ≈ 0.258
        #   true 6-arm dendrite (most mass at tips): std/mean ≳ 0.5
        #   thin shell (only at radius r): std/mean ≈ 0
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        mean_dist = float(dist.mean())
        std_dist = float(dist.std())
        branchiness = std_dist / mean_dist if mean_dist > 0 else float('nan')
    else:
        bbmin = bbmax = None
        com = [float('nan')] * 3
        rg = float('nan')
        boundary_count = 0
        branchiness = float('nan')

    # Growth rate (delta active per step since last snapshot)
    growth_rate = (
        float('nan') if prev_alive_count is None
        else (active_count - prev_alive_count)
    )

    return {
        'step': step,
        't_wall': t_wall,
        'finite': finite,
        'nan': nan_count,
        'inf': inf_count,
        'min': mins,
        'max': maxs,
        'mean': means,
        'var': vars_,
        'std': stds,
        'quartiles': quartiles,
        'active_count': active_count,
        'active_frac': active_count / float(N),
        'bbox_min': bbmin,
        'bbox_max': bbmax,
        'com': com,
        'rg': rg,
        'boundary_count': boundary_count,
        'boundary_frac': (boundary_count / active_count) if active_count > 0 else 0.0,
        'envelope_octa': envelope_octa,
        'branchiness': branchiness,
        'hist': hist,
        'hist_min': hist_min,
        'hist_max': hist_max,
        'growth_rate': growth_rate,
        # Mass: total of phi (channel 0) and total u (channel 1) — for
        # checking conservation in Karma-Rappel. The latent-heat coupling is
        # du = -0.5 * dphi/dt, so sum(phi) + 2*sum(u) is the conserved
        # invariant (each unit of phi solidified consumes 0.5 of u, so
        # adding 1 to phi total must drop u total by 0.5 -> +2x weight).
        'mass_phi': float(grid[..., 0].sum()),
        'mass_u': float(grid[..., 1].sum()),
        'mass_conserved_proxy': float(grid[..., 0].sum() + 2.0 * grid[..., 1].sum()),
    }


# ── Trial runner ─────────────────────────────────────────────────────────────

@dataclass
class Trial:
    rule: str
    seed: int
    size: int
    overrides: dict = field(default_factory=dict)
    snapshots: list[dict] = field(default_factory=list)
    elapsed: float = 0.0


def run_trial(ctx, rule: str, seed: int, size: int, steps: int, interval: int,
              param_overrides: dict | None = None) -> Trial:
    """Run one trial and capture snapshots every `interval` steps."""
    runner = HeadlessRunner(ctx, rule, size=size, seed=seed)
    if param_overrides:
        runner.params.update(param_overrides)

    trial = Trial(rule=rule, seed=seed, size=size,
                  overrides=dict(param_overrides or {}))

    t0 = time.perf_counter()
    prev_alive = None

    # Snapshot at step 0 (initial state)
    g = runner.read_grid()
    snap = snapshot_stats(g, 0, time.time(), prev_alive)
    trial.snapshots.append(snap)
    prev_alive = snap['active_count']

    for i in range(1, steps + 1):
        runner.step()
        if i % interval == 0 or i == steps:
            g = runner.read_grid()
            snap = snapshot_stats(g, i, time.time(), prev_alive)
            trial.snapshots.append(snap)
            prev_alive = snap['active_count']

    trial.elapsed = time.perf_counter() - t0
    runner.release()
    return trial


# ── JSON dump (compatible with debug_runs/ schema) ───────────────────────────

def dump_trial(trial: Trial, out_dir: Path) -> Path:
    safe_overrides = "_".join(f"{k}{v:g}" for k, v in trial.overrides.items()) or "default"
    fname = f"{trial.rule}__seed{trial.seed}__{safe_overrides}.json"
    path = out_dir / fname
    with open(path, 'w') as f:
        json.dump({
            'rule': trial.rule,
            'seed': trial.seed,
            'size': trial.size,
            'overrides': trial.overrides,
            'elapsed_s': trial.elapsed,
            'history': trial.snapshots,
        }, f)
    return path


# ── Analysis ─────────────────────────────────────────────────────────────────

def trajectory_summary(trial: Trial) -> dict:
    """Compute summary statistics over a trial's trajectory."""
    snaps = trial.snapshots
    if len(snaps) < 2:
        return {}

    af = [s['active_frac'] for s in snaps]
    rg = [s['rg'] for s in snaps if math.isfinite(s.get('rg', float('nan')))]
    mass = [s['mass_conserved_proxy'] for s in snaps]
    nans = [sum(s['nan']) for s in snaps]
    infs = [sum(s['inf']) for s in snaps]

    # Mass drift: relative change of (phi + 0.5*u) total. Should be small.
    mass_drift = (mass[-1] - mass[0]) / max(abs(mass[0]), 1e-9) if mass else float('nan')

    # Time to reach af=0.10, 0.25, halt detection
    def time_to_af(target):
        for s in snaps:
            if s['active_frac'] >= target:
                return s['step']
        return None

    # Halt: when growth_rate drops below 1% of peak for 3 consecutive snaps
    growth_rates = [s['growth_rate'] for s in snaps if math.isfinite(s.get('growth_rate', float('nan')))]
    peak_gr = max(growth_rates) if growth_rates else 0
    halt_step = None
    if peak_gr > 0:
        consec = 0
        for s in snaps:
            gr = s.get('growth_rate', float('nan'))
            if math.isfinite(gr) and gr < 0.01 * peak_gr:
                consec += 1
                if consec >= 3:
                    halt_step = s['step']
                    break
            else:
                consec = 0

    # Final-state envelope_octa (axis-reach / diagonal-reach):
    #   sphere ≈ 1.0, octahedron > 1, cube < 1.
    final = snaps[-1]
    envelope_octa = final.get('envelope_octa', float('nan'))

    return {
        'final_af': af[-1],
        'peak_af': max(af),
        'final_rg': rg[-1] if rg else float('nan'),
        'mass_drift_rel': mass_drift,
        'final_nan_count': nans[-1] if nans else 0,
        'final_inf_count': infs[-1] if infs else 0,
        'peak_growth_rate': peak_gr,
        't_af10': time_to_af(0.10),
        't_af25': time_to_af(0.25),
        'halt_step': halt_step,
        'final_envelope_octa': envelope_octa,
        'n_snapshots': len(snaps),
    }


def print_per_trial(trials: list[Trial]):
    print("\n" + "=" * 110)
    print("PER-TRIAL TRAJECTORY (rule, seed, overrides → summary)")
    print("=" * 110)
    hdr = (
        f"  {'rule':<22} {'seed':>5} {'overrides':<22} "
        f"{'final_af':>9} {'peak_gr':>8} {'halt@':>6} "
        f"{'mass_drift':>11} {'NaN':>4} {'oct':>5}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for t in trials:
        s = trajectory_summary(t)
        ovr = "_".join(f"{k}={v:g}" for k, v in t.overrides.items()) or "default"
        if len(ovr) > 22:
            ovr = ovr[:19] + "..."
        halt = f"{s['halt_step']}" if s.get('halt_step') else "—"
        print(
            f"  {t.rule:<22} {t.seed:>5} {ovr:<22} "
            f"{s.get('final_af', float('nan')):>9.3f} "
            f"{s.get('peak_growth_rate', 0):>8.0f} {halt:>6} "
            f"{s.get('mass_drift_rel', 0):>+11.4f} "
            f"{s.get('final_nan_count', 0):>4} "
            f"{s.get('final_envelope_octa', float('nan')):>5.2f}"
        )


def print_seed_variance(trials: list[Trial]):
    """For each (rule, override) cell, compute variance of final-state metrics
    across seeds. High variance ⇒ seed-dependent / stochastic. Low ⇒ deterministic."""
    cells = {}  # (rule, override_str) → list of summaries
    for t in trials:
        ovr = "_".join(f"{k}={v:g}" for k, v in sorted(t.overrides.items())) or "default"
        cells.setdefault((t.rule, ovr), []).append(trajectory_summary(t))

    print("\n" + "=" * 110)
    print("SEED VARIANCE per (rule, overrides) — std across seeds, > tells you which patterns are stochastic")
    print("=" * 110)
    hdr = (
        f"  {'rule':<22} {'overrides':<22} {'n':>3} "
        f"{'final_af σ':>11} {'oct σ':>7} {'halt σ':>8} {'verdict':<14}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for (rule, ovr), summaries in sorted(cells.items()):
        if len(summaries) < 2:
            continue
        af_vals = [s['final_af'] for s in summaries if math.isfinite(s.get('final_af', float('nan')))]
        oct_vals = [s['final_envelope_octa'] for s in summaries
                    if math.isfinite(s.get('final_envelope_octa', float('nan')))]
        halt_vals = [s['halt_step'] for s in summaries if s.get('halt_step') is not None]
        af_std = float(np.std(af_vals)) if len(af_vals) > 1 else float('nan')
        oct_std = float(np.std(oct_vals)) if len(oct_vals) > 1 else float('nan')
        halt_std = float(np.std(halt_vals)) if len(halt_vals) > 1 else float('nan')
        verdict = ("stochastic" if af_std > 0.02 or oct_std > 0.05 else "deterministic")
        ovr_show = ovr if len(ovr) <= 22 else ovr[:19] + "..."
        print(
            f"  {rule:<22} {ovr_show:<22} {len(summaries):>3} "
            f"{af_std:>11.4f} {oct_std:>7.3f} {halt_std:>8.1f} {verdict:<14}"
        )


def print_param_sweep(trials: list[Trial]):
    """For trials sharing rule but different overrides, show how a metric
    changes monotonically with the parameter. Detects parameter sensitivity."""
    by_rule_param = {}  # (rule, param_name) → {param_val → [summary, ...]}
    for t in trials:
        if not t.overrides:
            continue
        for k, v in t.overrides.items():
            by_rule_param.setdefault((t.rule, k), {}).setdefault(v, []).append(
                trajectory_summary(t)
            )

    if not by_rule_param:
        return

    print("\n" + "=" * 110)
    print("PARAMETER SWEEPS — final-state metrics vs swept parameter")
    print("=" * 110)
    for (rule, pname), val_map in sorted(by_rule_param.items()):
        if len(val_map) < 2:
            continue
        print(f"\n  {rule}  parameter={pname}")
        print(f"    {'value':>10}  {'mean(af)':>10}  {'mean(oct)':>10}  "
              f"{'mean(rg)':>10}  {'mean(halt)':>11}  {'n':>3}")
        for v in sorted(val_map.keys()):
            ss = val_map[v]
            af = np.nanmean([s['final_af'] for s in ss])
            oc = np.nanmean([s['final_envelope_octa'] for s in ss])
            rg = np.nanmean([s['final_rg'] for s in ss])
            ht = np.nanmean([s['halt_step'] for s in ss if s.get('halt_step') is not None]
                            or [float('nan')])
            print(f"    {v:>10g}  {af:>10.3f}  {oc:>10.2f}  {rg:>10.3f}  {ht:>11.0f}  {len(ss):>3}")


def print_temporal_patterns(trials: list[Trial]):
    """Summarize temporal trajectory patterns: monotonic growth, oscillation,
    plateau, divergence. Detects subtle dynamical behaviors not visible in
    final-state stats."""
    print("\n" + "=" * 110)
    print("TEMPORAL PATTERN DETECTION — looks for oscillation, plateaus, runaway")
    print("=" * 110)
    print(f"  {'rule':<22} {'seed':>5} {'overrides':<20} "
          f"{'pattern':<14} {'detail':<40}")
    print("  " + "-" * 108)
    for t in trials:
        snaps = t.snapshots
        if len(snaps) < 5:
            continue
        af = np.array([s['active_frac'] for s in snaps])
        if np.all(af == 0):
            pattern, detail = "no-growth", "active_frac stayed 0"
        else:
            # Differentiate: monotonic growth, plateau, oscillation, runaway.
            # The "oscillation" verdict is reserved for FRONT INSTABILITY
            # (Mullins-Sekerka): meaningful negative growth bursts that
            # are at least a small fraction of peak forward growth.
            # Sub-1% jitter near a halted state is just numerical noise,
            # not a real instability — we filter it out by requiring the
            # mean shrink magnitude to exceed 2% of the peak forward rate.
            d_af = np.diff(af)
            grows = (d_af > 1e-4).sum()
            shrinks = (d_af < -1e-4).sum()
            flat = len(d_af) - grows - shrinks
            peak_grow = float(d_af.max()) if d_af.size else 0.0
            shrink_vals = -d_af[d_af < -1e-4]
            mean_shrink = float(shrink_vals.mean()) if shrink_vals.size else 0.0
            real_instability = (
                shrinks > 3 and peak_grow > 1e-3 and
                mean_shrink > 0.02 * peak_grow
            )
            if af[-1] >= 0.99:
                pattern, detail = "saturated", f"af reached {af[-1]:.3f}"
            elif real_instability:
                pattern, detail = "oscillation", (
                    f"{grows} grows, {shrinks} shrinks, mean shrink "
                    f"{mean_shrink:.4f} = {100*mean_shrink/peak_grow:.0f}% of peak"
                )
            elif flat > len(d_af) * 0.6:
                pattern, detail = "plateau", (
                    f"{flat}/{len(d_af)} intervals flat — halted at af={af[-1]:.3f}"
                )
            elif grows > 0 and d_af[-3:].mean() < 1e-4:
                pattern, detail = "halted", (
                    f"af={af[-1]:.3f}, last 3 deltas tiny"
                )
            else:
                pattern, detail = "growing", (
                    f"af={af[-1]:.3f}, peak rate {d_af.max():.5f}"
                )
        ovr = "_".join(f"{k}{v:g}" for k, v in t.overrides.items()) or "default"
        if len(ovr) > 20:
            ovr = ovr[:17] + "..."
        print(f"  {t.rule:<22} {t.seed:>5} {ovr:<20} {pattern:<14} {detail:<40}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_csv(s: str, cast):
    return [cast(x.strip()) for x in s.split(',') if x.strip()]


def print_matched_af_shapes(trials: list[Trial],
                             af_targets=(0.02, 0.05, 0.10, 0.20)):
    """Print envelope_octa at matched-af snapshots across the trajectory.

    The final-state envelope is dominated by box-fill saturation — every
    rule reads ~0.58 (= 1/√3, the bbox-fill ratio) once the crystal
    spans most of the grid. To see the actual kernel-driven morphology,
    we need to sample shape metrics WHILE the crystal is still small.

    Octahedron should give axis/diag > 1.5 at low af.
    Cube           ≈ 0.58.
    Sphere         ≈ 1.0.
    """
    print("\n" + "=" * 110)
    print("MATCHED-AF SHAPE ANALYSIS — envelope_octa (axis-reach / diag-reach)")
    print("        sphere ≈ 1.0   octahedron > 1.5   cube ≈ 0.58")
    print("        branchiness (std/mean of radial distance from COM):")
    print("          uniform 3D ball ≈ 0.26   thin shell ≈ 0   6-arm dendrite ≳ 0.5")
    print("=" * 110)
    cells = {}  # (rule, override) → list of trials
    for t in trials:
        ovr = "_".join(f"{k}={v:g}" for k, v in sorted(t.overrides.items())) or "default"
        cells.setdefault((t.rule, ovr), []).append(t)

    hdr = f"  {'rule':<22} {'overrides':<22} " + " ".join(
        f"{f'oct@{a:.2f}':>10}" for a in af_targets
    ) + " " + " ".join(
        f"{f'br@{a:.2f}':>9}" for a in af_targets
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for (rule, ovr), trial_group in sorted(cells.items()):
        oct_cols = []
        br_cols = []
        for af_target in af_targets:
            oct_vals = []
            br_vals = []
            for t in trial_group:
                hit = None
                for s in t.snapshots:
                    if s['active_frac'] >= af_target:
                        hit = s
                        break
                if hit is None:
                    continue
                env = hit.get('envelope_octa', float('nan'))
                br = hit.get('branchiness', float('nan'))
                if math.isfinite(env):
                    oct_vals.append(env)
                if math.isfinite(br):
                    br_vals.append(br)
            oct_cols.append(f"{np.mean(oct_vals):>10.2f}" if oct_vals else f"{'—':>10}")
            br_cols.append(f"{np.mean(br_vals):>9.1f}" if br_vals else f"{'—':>9}")
        ovr_show = ovr if len(ovr) <= 22 else ovr[:19] + "..."
        print(f"  {rule:<22} {ovr_show:<22} " + " ".join(oct_cols) + " " + " ".join(br_cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', default=','.join(CRYSTAL_RULES),
                    help="Comma-separated rule names, default: all crystals")
    ap.add_argument('--seeds', default='42,7,123',
                    help="Comma-separated RNG seeds")
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--interval', type=int, default=30,
                    help="Snapshot every N steps")
    ap.add_argument('--undercoolings', default='',
                    help="Comma-separated Undercooling values to sweep "
                         "(empty = use preset default)")
    ap.add_argument('--anisotropies', default='',
                    help="Comma-separated 'Anisotropy strength' values to sweep")
    ap.add_argument('--no-dump', action='store_true',
                    help="Skip per-trial JSON dumps (just print summaries)")
    ap.add_argument('--out-dir', default='',
                    help="Override output directory (default: debug_runs/batch_<ts>/)")
    args = ap.parse_args()

    rules = parse_csv(args.rules, str)
    seeds = parse_csv(args.seeds, int)
    undercoolings = parse_csv(args.undercoolings, float) if args.undercoolings else [None]
    anisotropies = parse_csv(args.anisotropies, float) if args.anisotropies else [None]

    out_dir = None
    if not args.no_dump:
        if args.out_dir:
            out_dir = Path(args.out_dir)
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = Path(ROOT) / "debug_runs" / f"batch_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"# Output dir: {out_dir}")

    n_trials = len(rules) * len(seeds) * len(undercoolings) * len(anisotropies)
    print(f"# Running {n_trials} trials "
          f"({len(rules)} rules × {len(seeds)} seeds × "
          f"{len(undercoolings)} U × {len(anisotropies)} A) "
          f"@ size={args.size}, steps={args.steps}, interval={args.interval}")

    ctx = create_headless_context()
    if isinstance(ctx, tuple):
        _win, ctx = ctx
    trials = []
    t_start = time.perf_counter()
    done = 0

    for rule in rules:
        for seed in seeds:
            for U in undercoolings:
                for A in anisotropies:
                    overrides = {}
                    if U is not None:
                        overrides['Undercooling'] = U
                    if A is not None:
                        overrides['Anisotropy strength'] = A
                    try:
                        t = run_trial(ctx, rule, seed, args.size, args.steps,
                                      args.interval, overrides)
                        trials.append(t)
                        if out_dir:
                            dump_trial(t, out_dir)
                        done += 1
                        elapsed = time.perf_counter() - t_start
                        eta = elapsed / done * (n_trials - done)
                        print(
                            f"  [{done:>3}/{n_trials}] {rule} seed={seed} "
                            f"{overrides}  elapsed={t.elapsed:.1f}s  ETA={eta:.0f}s"
                        )
                    except Exception as e:
                        print(f"  [FAIL] {rule} seed={seed} {overrides}: {e}")

    print(f"\n# All trials done in {time.perf_counter() - t_start:.1f}s")

    # Analyses
    print_per_trial(trials)
    print_matched_af_shapes(trials)
    print_temporal_patterns(trials)
    print_seed_variance(trials)
    print_param_sweep(trials)


if __name__ == '__main__':
    main()
