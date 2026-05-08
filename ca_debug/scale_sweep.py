"""Scale-sweep probe: run the same rule at a ladder of grid sizes
and print per-step alive-ratio so size-dependent regressions show up
as a single sweep table.

Triggered by the report that 3D Game of Life behaves correctly at
size<=96 but collapses to a static block at size>=128 — exactly the
threshold at which `_sparse_supported()` flips on. This probe lets us
A/B the sparse-dispatch path by toggling
``RULE_PRESETS[rule]['sparse_dispatch']`` between runs.

Usage::

    python -m ca_debug.scale_sweep --rule game_of_life_3d \
        --sizes 64,96,128,160,192 --steps 60
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _alive_ratio(grid: np.ndarray, threshold: float = 0.5) -> float:
    return float((grid[..., 0] > threshold).mean())


def _run_one(ctx, rule: str, *, size: int, steps: int, seed: int,
             sparse: bool | None = None,
             samples: int | None = None,
             ) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Run rule at size for steps; return (alive trajectory, g0, gN).

    If `sparse` is not None, force-set the preset flag for this run.
    If `samples` is not None, downsample the per-step trajectory to
    that many evenly-spaced points (plus the initial sample). This is
    used by the size-scaled `--all-rules` scan to make trajectories
    from different `steps` counts directly comparable: a 100-step run
    and a 300-step run both produce signatures over the same number
    of measurements, just spaced 1× vs 3× further apart in sim time.
    """
    from simulator import RULE_PRESETS
    from test_harness import HeadlessRunner

    saved = RULE_PRESETS[rule].get('sparse_dispatch', False)
    if sparse is not None:
        RULE_PRESETS[rule]['sparse_dispatch'] = sparse
    try:
        r = HeadlessRunner(ctx, rule, size=size, seed=seed)
        traj: list[float] = []
        g0 = r.read_grid().astype(np.float32).copy()
        traj.append(_alive_ratio(g0))
        if samples is None or samples >= steps:
            sample_at = set(range(1, steps + 1))
        else:
            # Evenly-spaced sample indices in [1, steps], inclusive of last.
            sample_at = {int(round(i * steps / samples))
                         for i in range(1, samples + 1)}
        for s in range(1, steps + 1):
            r.step()
            if s in sample_at:
                traj.append(_alive_ratio(r.read_grid()))
        gN = r.read_grid().astype(np.float32).copy()
        if hasattr(r, 'release'):
            try: r.release()
            except Exception: pass
        return traj, g0, gN
    finally:
        RULE_PRESETS[rule]['sparse_dispatch'] = saved


def _solid_block_signature(g: np.ndarray, threshold: float = 0.5) -> dict:
    """Quantify how 'solid block'-like the final state is.
    A clean B6-7/S5-7 chaos has alive~5-15% and high boundary entropy.
    A pathological 'cube' state has alive~50%+ in a contiguous region
    and near-zero per-step delta.
    """
    alive = g[..., 0] > threshold
    frac = float(alive.mean())
    # Largest filled axis-aligned box: cheap proxy via per-axis bounds
    if not alive.any():
        return {'alive_frac': 0.0, 'bbox_fill': 0.0, 'bbox_dims': (0, 0, 0)}
    nz = np.argwhere(alive)
    lo = nz.min(0); hi = nz.max(0) + 1
    box_dims = tuple(int(x) for x in (hi - lo))
    box_vol = int(np.prod(box_dims))
    bbox_fill = float(alive.sum()) / max(box_vol, 1)
    return {'alive_frac': frac, 'bbox_fill': bbox_fill, 'bbox_dims': box_dims}


def _health_signature(g: np.ndarray) -> dict:
    """Cheap sanity check on the final grid (channel 0 only).

    Distinguishes physically-healthy size-dependent richness from
    classic explicit-scheme failure modes:

    * `clip_hi` / `clip_lo` — fraction at extreme values (NaN, ±1.0,
      ±max-of-range). High clip is the signature of biharmonic /
      reaction-diffusion blow-up.
    * `frozen` — std < 1e-6: the field has collapsed to a uniform
      value (typical of solid-block pathology, dead-out).
    * `nan_frac` — fraction of non-finite values (numerical disaster).

    A 'bug' is anything with clip > 30%, nan > 0%, or frozen=True at a
    size where the smaller grid was healthy.
    """
    c = g[..., 0]
    nan_frac = float((~np.isfinite(c)).mean())
    finite = c[np.isfinite(c)]
    if finite.size == 0:
        return {'nan_frac': nan_frac, 'clip_hi': 0.0, 'clip_lo': 0.0,
                'frozen': True, 'std': 0.0,
                'min': float('nan'), 'max': float('nan'),
                'mean': float('nan')}
    cmin, cmax = float(finite.min()), float(finite.max())
    span = max(cmax - cmin, 1e-9)
    # Per-cell distance from each end of the dynamic range.
    clip_hi = float((finite >= cmax - 1e-4 * span).mean())
    clip_lo = float((finite <= cmin + 1e-4 * span).mean())
    std = float(finite.std())
    return {'nan_frac': nan_frac, 'clip_hi': clip_hi, 'clip_lo': clip_lo,
            'frozen': std < 1e-6, 'std': std,
            'min': cmin, 'max': cmax, 'mean': float(finite.mean())}


def _health_verdict(h_small: dict, h_large: dict) -> str:
    """Compare per-size health snapshots; return one of:
        'healthy', 'blowup', 'frozen', 'nan', 'feature'
    A divergence is only a 'bug' if the larger grid is in a clearly
    pathological state that the smaller grid is not.
    """
    if h_large['nan_frac'] > 0:
        return 'nan'
    # Saturated to one extreme — classic explicit-scheme blow-up.
    # A *real* blow-up has TWO signatures, both required:
    #   1. clip-fraction grew significantly at the larger grid, AND
    #   2. std SHRANK — the field lost dynamic range (collapsed onto
    #      the saturated value).
    # Without (2) we get false-positives on rules whose dynamics are
    # intrinsically sparse — most cells at 0 background, e.g.
    # `causal_ca`, `bz_excitable` — where rising large-grid background
    # fraction looks like clip-growth but is actually just more empty
    # volume *with richer wavefronts living in it*. In those cases std
    # grows along with clip_lo because the wavefronts have more space
    # to fragment.
    std_collapsed = h_large['std'] < 0.5 * max(h_small['std'], 1e-6)
    for side in ('clip_hi', 'clip_lo'):
        if (h_large[side] > 0.30
                and h_large[side] > h_small[side] + 0.15
                and h_small[side] < 0.50
                and std_collapsed):
            return 'blowup'
    # Field died / froze where smaller did not.
    if h_large['frozen'] and not h_small['frozen']:
        return 'frozen'
    return 'feature'


def _trajectory_signature(traj: list[float]) -> dict:
    """Compact summary used to A/B trajectories at different sizes.

    Two trajectories are ~equivalent if all four numbers are close;
    they diverge in physically meaningful ways otherwise."""
    import statistics
    if not traj:
        return {'mean': 0.0, 'std': 0.0, 'tail': 0.0, 'p2p_tail': 0.0}
    tail = traj[len(traj) // 2:]   # second half = post-transient
    return {
        'mean': float(statistics.fmean(traj)),
        'std': float(statistics.pstdev(traj)) if len(traj) > 1 else 0.0,
        'tail': float(statistics.fmean(tail)),
        'p2p_tail': float(max(tail) - min(tail)),
    }


def _divergence(sig_small: dict, sig_large: dict) -> float:
    """L1 distance between two trajectory signatures, normalised so a
    perfectly-matching pair scores ~0 and a 'collapsed to cube' vs
    'period-2 blink' pair scores >> 1."""
    keys = ('tail', 'p2p_tail', 'mean', 'std')
    return sum(abs(sig_small[k] - sig_large[k]) for k in keys)


def _scan_all_rules(ctx, sizes: list[int], steps: int, seed: int,
                    scale_steps: bool = True,
                    ) -> list[dict]:
    """For every preset, compare the trajectory signature at the
    smallest size against every larger size. Returns rows sorted by
    max divergence, descending. Skips presets that fail to run.

    When `scale_steps` is true (default), each size runs for
    `steps × size / base_size` iterations and the trajectory is then
    downsampled to `steps` points. This eliminates the false-positive
    flood from rules whose dynamics are voxel-paced (Eden, crystal
    growth, fire, erosion, BZ excitable fronts): a unit-speed front
    covers the same fraction of the cube at every size, so the
    trajectory signature is genuinely size-invariant. Without this,
    measuring at fixed step count flags every front-propagation rule
    as 'divergent' simply because the larger cubes are still in the
    transient regime.

    Genuine size-dependent BUGS (sparse-dispatch fallthrough,
    biharmonic instability, kernel-radius mis-scaling) still surface
    because they alter the *steady-state* signature, not just the
    transient timing.
    """
    from simulator import RULE_PRESETS
    rows: list[dict] = []
    rule_names = sorted(RULE_PRESETS.keys())
    base_size = sizes[0]
    for i, name in enumerate(rule_names):
        # Skip composed/agent/entity rules — they often have
        # default_size constraints that auto-bump and confound the comparison.
        p = RULE_PRESETS[name]
        if p.get('compose') or p.get('passes'):
            # Multi-pass: still try, but tag.
            pass
        try:
            sigs: dict[int, dict] = {}
            healths: dict[int, dict] = {}
            for size in sizes:
                if scale_steps:
                    run_steps = max(steps,
                                    int(round(steps * size / base_size)))
                    samples = steps
                else:
                    run_steps = steps
                    samples = None
                traj, _, gN = _run_one(ctx, name, size=size,
                                       steps=run_steps,
                                       seed=seed, samples=samples)
                sigs[size] = _trajectory_signature(traj)
                healths[size] = _health_signature(gN)
        except Exception as e:
            rows.append({'rule': name, 'status': f'ERR: {type(e).__name__}',
                         'max_div': float('nan'), 'divs': {},
                         'verdict': 'error'})
            continue
        # Compare every larger size against base_size.
        divs = {s: _divergence(sigs[base_size], sigs[s]) for s in sizes[1:]}
        max_div = max(divs.values()) if divs else 0.0
        # Health verdict: worst classification across sizes.
        verdicts = [_health_verdict(healths[base_size], healths[s])
                    for s in sizes[1:]]
        priority = {'nan': 4, 'blowup': 3, 'frozen': 2, 'feature': 1, 'healthy': 0}
        verdict = max(verdicts, key=lambda v: priority.get(v, 0)) if verdicts else 'healthy'
        rows.append({'rule': name, 'status': 'ok', 'max_div': max_div,
                     'divs': divs, 'sigs': sigs, 'healths': healths,
                     'verdict': verdict})
        # Lightweight progress
        if (i + 1) % 5 == 0 or i + 1 == len(rule_names):
            print(f"  ... {i+1}/{len(rule_names)} scanned",
                  file=__import__('sys').stderr)
    rows.sort(key=lambda r: (-r['max_div']
                             if r['max_div'] == r['max_div']
                             else 0))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rule', default='game_of_life_3d')
    p.add_argument('--sizes', default='64,96,128,160,192',
                   help='comma-separated grid edges')
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ab-sparse', action='store_true',
                   help='at each size, run with sparse=on AND sparse=off and '
                        'diff the trajectories')
    p.add_argument('--print-traj', action='store_true',
                   help='print full per-step alive ratio')
    p.add_argument('--all-rules', action='store_true',
                   help='scan every preset; print divergence-ranked table')
    p.add_argument('--top', type=int, default=20,
                   help='--all-rules: how many top-divergence rows to print')
    p.add_argument('--threshold', type=float, default=0.15,
                   help='--all-rules: highlight rows with max_div > threshold')
    p.add_argument('--no-scale-steps', action='store_true',
                   help='--all-rules: disable proportional step-count scaling. '
                        'By default each size runs steps×(size/base_size) '
                        'iterations so voxel-paced fronts are measured at '
                        'equivalent simulation time, not equivalent step '
                        'count. Pass this to revert to fixed step count '
                        '(noisier — flags front-propagation rules as false '
                        'positives).')
    args = p.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(',')]

    import moderngl
    ctx = moderngl.create_standalone_context(require=430)

    if args.all_rules:
        scale_steps = not args.no_scale_steps
        mode_tag = ('size-scaled steps' if scale_steps
                    else 'fixed steps (front-propagation rules will false-positive)')
        print(f"\n=== scale_sweep --all-rules sizes={sizes} "
              f"steps={args.steps} [{mode_tag}] ===\n")
        rows = _scan_all_rules(ctx, sizes, args.steps, args.seed,
                                scale_steps=scale_steps)
        sz_cols = '  '.join(f'div@{s}' for s in sizes[1:])
        print(f"\n{'rule':<32}  {'max_div':>8}  {sz_cols}  {'verdict':<8}  notes")
        print('-' * (60 + 9 * len(sizes)))
        bug_verdicts = {'nan', 'blowup', 'frozen', 'error'}
        # Sort: bugs first (by verdict severity), then by max_div desc.
        priority = {'nan': 4, 'blowup': 3, 'frozen': 2, 'error': 5,
                    'feature': 1, 'healthy': 0}
        rows_sorted = sorted(
            rows,
            key=lambda r: (-priority.get(r.get('verdict', 'healthy'), 0),
                           -(r['max_div'] if r['max_div'] == r['max_div'] else 0)))
        printed = 0
        for r in rows_sorted:
            if printed >= args.top:
                break
            divs_str = '  '.join(
                f"{r['divs'].get(s, float('nan')):>6.3f}" for s in sizes[1:])
            verdict = r.get('verdict', 'healthy')
            note = ''
            if verdict in bug_verdicts:
                note = '<<< BUG'
                # Annotate with size at which it first appears bad.
                hs = r.get('healths', {})
                for s in sizes[1:]:
                    h = hs.get(s, {})
                    if (h.get('nan_frac', 0) > 0
                            or max(h.get('clip_hi', 0), h.get('clip_lo', 0)) > 0.30
                            or h.get('frozen', False)):
                        note += f' @ size {s}'
                        break
            elif (r['max_div'] == r['max_div']
                  and r['max_div'] > args.threshold):
                note = '(feature: richer dynamics at large size)'
            print(f"{r['rule']:<32}  {r['max_div']:>8.3f}  {divs_str}  "
                  f"{verdict:<8}  {note}")
            printed += 1
        n_bugs = sum(1 for r in rows if r.get('verdict') in bug_verdicts)
        n_div = sum(1 for r in rows
                    if r['max_div'] == r['max_div']
                    and r['max_div'] > args.threshold)
        print(f"\n{n_bugs} likely BUG(s); {n_div} rules diverge by > "
              f"{args.threshold} (most are size-dependent features)")
        return 0

    print(f"\n=== scale_sweep: {args.rule} seed={args.seed} steps={args.steps} ===\n")
    fmt_hdr = (f"{'size':>5}  {'mode':<8}  {'alive[0]':>9}  {'alive[mid]':>10}  "
               f"{'alive[N]':>9}  {'bbox_fill':>9}  {'bbox_dims':<14}  notes")
    print(fmt_hdr)
    print("-" * len(fmt_hdr))

    modes: list[bool | None]
    if args.ab_sparse:
        modes = [False, True]   # dense first, then sparse
    else:
        modes = [None]          # leave preset default

    for size in sizes:
        for mode in modes:
            label = ('dense' if mode is False
                     else 'sparse' if mode is True
                     else 'default')
            try:
                traj, g0, gN = _run_one(
                    ctx, args.rule, size=size, steps=args.steps,
                    seed=args.seed, sparse=mode)
            except Exception as e:
                print(f"{size:>5}  {label:<8}  ERROR: {e!r}")
                continue
            sig = _solid_block_signature(gN)
            mid = traj[len(traj) // 2]
            note = ''
            if sig['bbox_fill'] > 0.85 and sig['alive_frac'] > 0.4:
                note = '<<< SOLID-BLOCK PATHOLOGY'
            elif traj[-1] < 1e-4:
                note = '<<< died out'
            print(f"{size:>5}  {label:<8}  {traj[0]:>9.4f}  {mid:>10.4f}  "
                  f"{traj[-1]:>9.4f}  {sig['bbox_fill']:>9.4f}  "
                  f"{str(sig['bbox_dims']):<14}  {note}")
            if args.print_traj:
                line = ' '.join(f"{v:.3f}" for v in traj)
                print(f"        traj: {line}")
        if args.ab_sparse:
            print()  # blank line between sizes for readability

    return 0


if __name__ == '__main__':
    sys.exit(main())
