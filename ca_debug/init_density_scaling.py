"""init density-scaling audit.

Probe #9 — many init functions look correct in isolation but accidentally
encode a *size-dependent* density.  The canonical example caught earlier
in this bug-hunt was the ``_canonical_noise(size, rng) < threshold``
pattern (Bug D): the upsampled noise field has its variance crushed by
the linear interpolator, so the threshold catches a different fraction
of voxels at non-canonical sizes.  Other commonly seen offenders:

  - "n seeds scattered through the cube" where n scales as ``size**2``
    rather than ``size**3``, so volumetric density drops linearly with
    grid resolution and the rule starves at large sizes.
  - "n blobs of radius size//k" where n is constant — total covered
    volume is ``n * (size//k)**3``, so blob coverage is constant only
    if the rule scales the same way (often the rule kernel is
    canonical-size, so the same coverage doesn't mean the same per-
    kernel density).
  - hard-coded canonical-resolution helpers that fall through to
    ``rng.random((size, size, size))`` at non-canonical sizes,
    breaking seed→pattern reproducibility (a different *kind* of bug
    but surfaced by the same measurement).

For each rule × declared-init variant, construct the runner at several
sizes and measure two scale-invariant statistics of the initial state:

  - ``alive_frac``   fraction of voxels with |value| > eps in channel 0.
  - ``mean_abs``     mean |value| of channel 0.

A *legitimate* size-dependent init exists (single-voxel seed has
``alive_frac = 1/size**3`` by design).  We don't flag those — instead
we look at the RATIO of the largest-size measurement to the smallest-
size measurement.  Init kinds we expect to be size-invariant:

  - thresholded noise fields (alive_frac should be constant)
  - "n random specks where n scales with volume"
  - blob inits where (n_blobs, blob_radius) both scale appropriately

A ratio > 4x for ``alive_frac`` across sizes 32..192 is flagged as
``high`` — almost certainly a density bug.  2x..4x is ``med``
(possibly intentional, e.g. boundary effects in a tiny cube).  Single-
seed and "blob count constant, blob radius fixed in voxels" inits are
EXPECTED to scale poorly and are filtered by the ``is_seed_init``
heuristic (alive_frac < 0.001 at the largest tested size).

Usage::

    python -m ca_debug.init_density_scaling
    python -m ca_debug.init_density_scaling --rules nca_3d,gray_scott_3d
    python -m ca_debug.init_density_scaling --sizes 32,64,128 --severity med
    python -m ca_debug.init_density_scaling --json /tmp/idens.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback

import numpy as np


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'ok': 4, 'n/a': 5}

_EPS = 1e-6
# Ratio of max/min alive_frac across sizes above this → high severity
# (this metric is only used for non-sparse inits — see grading below).
_HIGH_RATIO = 4.0
_MED_RATIO = 2.0
# Sparse-init regime: alive_frac at largest size below this triggers the
# *count-scaling* check instead of the frac-stability check.
_SPARSE_INIT_FRAC = 5e-3
# For sparse inits, alive_count is expected to scale ∝ volume (= size**3).
# Fit log(count) vs log(size); slope ~3 is volume-scaling (correct),
# slope ~2 is area-scaling (the canonical density bug), slope ~0 is
# size-independent count (also a density bug for sparse inits).
# Tolerance window around the volume-scaling exponent.
_SPARSE_SLOPE_OK = (2.5, 3.5)
_SPARSE_SLOPE_MED = (2.0, 4.0)


def _construct(ctx, rule: str, size: int, seed: int,
               init_override: str | None):
    from test_harness import HeadlessRunner
    with contextlib.redirect_stdout(io.StringIO()):
        return HeadlessRunner(ctx, rule, size=size, seed=seed,
                              init_override=init_override)


def _measure_init(grid: np.ndarray) -> dict:
    """Return scale-invariant statistics of channel 0 of the initial grid."""
    # Use the primary channel (ch0); other channels often encode auxiliary
    # fields (energy, phase, velocity) whose density semantics differ.
    if grid.ndim == 4:
        ch0 = grid[..., 0]
    else:
        ch0 = grid
    finite = np.isfinite(ch0)
    if not finite.any():
        return {'alive_frac': float('nan'), 'mean_abs': float('nan'),
                'std': float('nan'), 'nonfinite': True}
    v = ch0[finite].astype(np.float64, copy=False)
    alive = (np.abs(v) > _EPS).sum()
    return {
        'alive_frac': float(alive) / float(v.size),
        'mean_abs':   float(np.abs(v).mean()),
        'std':        float(v.std()),
        'nonfinite':  False,
    }


def _probe_variant(ctx, rule: str, variant: str | None,
                   sizes: list[int], seed: int) -> dict:
    """Measure one variant across sizes.  Returns a dict ready to grade."""
    per_size: dict[int, dict] = {}
    errors: list[str] = []
    for sz in sizes:
        try:
            r = _construct(ctx, rule, sz, seed, variant)
        except Exception as e:  # noqa: BLE001  construction crash captured
            errors.append(f'size={sz}: construct {type(e).__name__}: {e}')
            continue
        try:
            grid = np.asarray(r.read_grid()).copy()
            per_size[sz] = _measure_init(grid)
        except Exception as e:  # noqa: BLE001  read crash captured
            errors.append(f'size={sz}: read {type(e).__name__}: {e}')
        finally:
            try: r.release()
            except Exception: pass  # noqa: BLE001

    out: dict = {
        'variant': variant,
        'per_size': per_size,
        'errors':   errors,
    }
    if errors and not per_size:
        out['grade'] = 'err'
        out['reason'] = errors[0]
        return out

    # Look at alive_frac variation across sizes.
    fracs = [(sz, m['alive_frac']) for sz, m in per_size.items()
             if not m.get('nonfinite')]
    if not fracs:
        out['grade'] = 'crit'
        out['reason'] = 'init non-finite at every probed size'
        return out
    fracs.sort()
    vals = [f for _, f in fracs]
    largest_sz, largest_frac = fracs[-1]

    out['fracs'] = fracs

    # Skip blank inits (all zero everywhere — by design).
    if all(v < _EPS for v in vals):
        out['grade'] = 'n/a'
        out['reason'] = 'blank init (all zero)'
        return out

    # Two-regime grading:
    #
    #   1. Dense init  (alive_frac at largest size >= _SPARSE_INIT_FRAC):
    #      alive_frac should be roughly size-independent.  Flag if the
    #      ratio max/min exceeds the thresholds.
    #
    #   2. Sparse init (alive_frac < _SPARSE_INIT_FRAC at largest size):
    #      alive_count is expected to scale ∝ volume.  Fit
    #      log(count) = α + β·log(size); β ≈ 3 is volume-scaling (ok),
    #      β ≈ 2 is area-scaling (the canonical density bug), β ≈ 0 is a
    #      fixed-N seed (ok — single-seed and small-N blob inits live
    #      here).  We can only distinguish the bug from a legitimate
    #      fixed-N init by looking at the slope: slope ~2 with a
    #      monotonically-growing count is the bug fingerprint, slope ~0
    #      (count plateaued) is intentional.
    counts = [(sz, frac * (sz ** 3)) for sz, frac in fracs]
    if largest_frac >= _SPARSE_INIT_FRAC:
        # Dense regime: alive_frac should be size-invariant.
        nz_vals = [v for v in vals if v > _EPS]
        if not nz_vals:
            out['grade'] = 'n/a'
            out['reason'] = 'all-zero at every size'
            return out
        fmin = min(nz_vals)
        fmax = max(nz_vals)
        ratio = fmax / fmin if fmin > 0 else float('inf')
        out['ratio_max_min'] = ratio
        out['regime'] = 'dense'
        if ratio >= _HIGH_RATIO:
            out['grade'] = 'high'
            out['reason'] = (
                f'dense init: alive_frac varies {ratio:.1f}x across '
                f'sizes ({fmin:.3g}..{fmax:.3g})')
        elif ratio >= _MED_RATIO:
            out['grade'] = 'med'
            out['reason'] = (
                f'dense init: alive_frac varies {ratio:.1f}x across '
                f'sizes ({fmin:.3g}..{fmax:.3g})')
        else:
            out['grade'] = 'ok'
        return out

    # Sparse regime.  Need at least three sizes for a meaningful fit;
    # with fewer points fall back to a simple slope between endpoints.
    out['regime'] = 'sparse'
    nz_counts = [(sz, c) for sz, c in counts if c > 0.5]
    if len(nz_counts) < 2:
        out['grade'] = 'ok'
        out['reason'] = 'sparse seed init, too few non-empty sizes to grade'
        return out
    sz_arr = np.log(np.array([sz for sz, _ in nz_counts], dtype=np.float64))
    cn_arr = np.log(np.array([c for _, c in nz_counts],  dtype=np.float64))
    # Linear fit β = slope of log(count) vs log(size).
    if len(nz_counts) >= 3:
        slope, _intercept = np.polyfit(sz_arr, cn_arr, 1)
    else:
        slope = float((cn_arr[-1] - cn_arr[0]) / (sz_arr[-1] - sz_arr[0]))
    out['count_slope'] = float(slope)
    out['counts'] = nz_counts

    # Asymptotic-density check: if the alive_frac at the *upper half* of
    # tested sizes is roughly constant (max/min < 1.3), the init is
    # behaving correctly at production sizes; a slope-fit anomaly from
    # the size=32 "floor regime" (max(8, ...) clamps) shouldn't be
    # flagged.  Use the upper half of the sizes for the asymptotic
    # check.  This catches the common "max(min_count, scaled_count)"
    # idiom which is intentional at tiny grids and exact at large.
    upper = nz_counts[len(nz_counts) // 2:]
    asymptotic_ok = False
    if len(upper) >= 2:
        upper_fracs = [c / (sz ** 3) for sz, c in upper]
        u_min = min(upper_fracs)
        u_max = max(upper_fracs)
        if u_min > 0 and (u_max / u_min) < 1.3:
            asymptotic_ok = True
            out['asymptotic_frac'] = float(np.median(upper_fracs))

    # Slope ~0 = constant N (intentional fixed-seed init, ok).
    # Slope ~3 = volume-scaling (correct sparse init).
    # Slope ~2 = area-scaling (BUG: density drops with size).
    # Slope ~1 = linear (also BUG).
    # Tolerate a wide band around 0 (fixed-N) AND a band around 3 (volume).
    fixed_n_band = 0.5
    vol_lo, vol_hi = _SPARSE_SLOPE_OK
    vol_lo_med, vol_hi_med = _SPARSE_SLOPE_MED
    if abs(slope) <= fixed_n_band:
        out['grade'] = 'ok'
        out['reason'] = (
            f'sparse init: alive_count ≈ constant '
            f'(slope={slope:+.2f}, fixed-N seed)')
    elif vol_lo <= slope <= vol_hi:
        out['grade'] = 'ok'
        out['reason'] = (
            f'sparse init: alive_count ∝ size^{slope:.2f} (volume-scaling)')
    elif asymptotic_ok:
        # Slope is off, but density is stable at production sizes —
        # this is the floor-regime knee, not a real density bug.
        out['grade'] = 'ok'
        out['reason'] = (
            f'sparse init: slope={slope:+.2f} skewed by small-grid '
            f'floor; asymptotic density stable at upper sizes')
    elif vol_lo_med <= slope <= vol_hi_med:
        out['grade'] = 'med'
        out['reason'] = (
            f'sparse init: alive_count slope {slope:+.2f} '
            f'(off from volume-scaling slope=3)')
    else:
        # Slope between fixed_n_band and 2.0 is the bug fingerprint:
        # count grows with size but slower than volume — density falls.
        out['grade'] = 'high'
        out['reason'] = (
            f'sparse init: alive_count ∝ size^{slope:.2f} '
            f'(expected size^3 = volume, or size^0 = fixed-N; '
            f'this slope means density falls with grid size)')
    return out


def _variants_for_rule(preset: dict) -> list[str]:
    seen: list[str] = []
    primary = preset.get('init')
    if primary:
        seen.append(primary)
    for v in (preset.get('init_variants') or []):
        if v and v not in seen:
            seen.append(v)
    return seen


def _select_rules(args) -> list[str]:
    from simulator import RULE_PRESETS, _resolve_composed_preset
    if args.rules:
        return [r.strip() for r in args.rules.split(',') if r.strip()]
    rules = []
    for r in sorted(RULE_PRESETS.keys()):
        try:
            preset = _resolve_composed_preset(r)
        except Exception:  # noqa: BLE001
            continue
        if preset.get('kind') == 'viewport':
            continue
        if preset.get('agent_count') or 'entity_arena' in preset:
            continue
        if (preset.get('passes') or [{}])[0].get('kind') == 'particle':
            continue
        if preset.get('particle_count'):
            continue
        if preset.get('audit_skip'):
            continue
        rules.append(r)
    if args.skip_flagship:
        rules = [r for r in rules if not r.startswith('flagship_')]
    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(',') if s.strip()}
        rules = [r for r in rules if r not in skip_set]
    return rules


def _probe_rule(ctx, rule: str, sizes: list[int], seed: int) -> dict:
    from simulator import _resolve_composed_preset
    try:
        preset = _resolve_composed_preset(rule)
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'worst_grade': 'err',
                'error': f'preset resolve: {type(e).__name__}: {e}',
                'variants': []}
    variants = _variants_for_rule(preset)
    if not variants:
        return {'rule': rule, 'worst_grade': 'n/a',
                'reason': 'no init declared', 'variants': []}
    per_variant: list[dict] = []
    for v in variants:
        per_variant.append(_probe_variant(ctx, rule, v, sizes, seed))
    worst = min((p['grade'] for p in per_variant),
                key=lambda g: _SEV_ORDER.get(g, 9))
    return {'rule': rule, 'worst_grade': worst,
            'n_variants': len(variants),
            'variants': per_variant}


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names (default: all).')
    ap.add_argument('--sizes', default='32,64,96,128,192',
                    help='Comma-separated sizes to probe (default: 32,64,96,128,192).')
    ap.add_argument('--seed', type=int, default=1001)
    ap.add_argument('--skip-flagship', action='store_true')
    ap.add_argument('--skip', help='Comma-separated rules to omit.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='med',
                    help='Min severity to print (default: med).')
    ap.add_argument('--json', help='Write per-rule report JSON to this path.')
    args = ap.parse_args(argv)

    sizes = [int(s.strip()) for s in args.sizes.split(',') if s.strip()]

    from test_harness import create_headless_context
    window, ctx = create_headless_context()

    rules = _select_rules(args)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, rule in enumerate(rules, 1):
        sys.stdout.write(f"\r[{i:>3}/{len(rules)}] {rule:<42}")
        sys.stdout.flush()
        rows.append(_probe_rule(ctx, rule, sizes, args.seed))
    sys.stdout.write('\r' + ' ' * 70 + '\r')
    elapsed = time.perf_counter() - t0

    rule_counts = {k: 0 for k in _SEV_ORDER}
    variant_counts = {k: 0 for k in _SEV_ORDER}
    for row in rows:
        rule_counts[row['worst_grade']] = rule_counts.get(row['worst_grade'], 0) + 1
        for v in row.get('variants') or []:
            variant_counts[v.get('grade', 'n/a')] = (
                variant_counts.get(v.get('grade', 'n/a'), 0) + 1)

    print(f'\ninit-density-scaling probe — {len(rules)} rules in {elapsed:.1f}s')
    print('  sizes probed:', sizes)
    print('  by rule (worst grade per rule):')
    for g in ('err', 'crit', 'high', 'med', 'ok', 'n/a'):
        print(f'    {g:<5}  {rule_counts.get(g, 0):>4}')
    print('  by variant:')
    for g in ('err', 'crit', 'high', 'med', 'ok', 'n/a'):
        print(f'    {g:<5}  {variant_counts.get(g, 0):>4}')

    sev_cap = _SEV_ORDER[args.severity]
    flagged = [r for r in rows if _SEV_ORDER.get(r['worst_grade'], 9) <= sev_cap]
    if flagged:
        print(f'\nflagged ({args.severity}+):')
        for row in flagged:
            print(f'  [{row["worst_grade"]:<4}] {row["rule"]}')
            for v in row.get('variants') or []:
                if _SEV_ORDER.get(v.get('grade', 'n/a'), 9) > sev_cap:
                    continue
                vn = v.get('variant') or '(default)'
                print(f'      [{v["grade"]:<4}] {vn:<32}  {v.get("reason", "")}')
                fracs = v.get('fracs') or []
                if fracs:
                    pretty = ', '.join(f'{sz}:{f:.3g}' for sz, f in fracs)
                    print(f'           alive_frac per size: {pretty}')
                counts = v.get('counts') or []
                if counts:
                    pretty = ', '.join(f'{sz}:{int(c)}' for sz, c in counts)
                    print(f'           alive_count per size: {pretty}')
                for e in (v.get('errors') or [])[:3]:
                    print(f'           ! {e}')

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({
                'sizes': sizes, 'seed': args.seed,
                'elapsed_s': elapsed,
                'rule_counts': rule_counts,
                'variant_counts': variant_counts,
                'rows': rows,
            }, fh, indent=2, default=float)
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()
