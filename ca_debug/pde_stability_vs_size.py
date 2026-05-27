"""Probe #22 — PDE stability across grid sizes.

For every preset, run at a ladder of grid sizes with DEFAULT dt and
DEFAULT params (no slider tweaks), then inspect the final state for
numerical-instability signatures that are masked at the canonical
REF_SIZE=128 grid but emerge as the user enlarges the cube.

Detected signatures (per channel, per size):

    nan        any non-finite value
    clamp_hi   |val|_max ∈ [990, 1010] -- the schrodinger_3d ±1e3
               safety clamp documented in `simulator.py:5284` is the
               canonical fingerprint of CFL violation in shaders that
               mask blowup rather than substep around it.
    blowup     |val|_max grew > 100× vs the baseline size AND
               baseline max stayed bounded < 10. Catches saturation
               at unit-bound clamps (e.g. Cahn-Hilliard clip(±1)) and
               raw fp32 explosion before NaN.
    saturated  channel std collapsed (>50× drop vs baseline) while
               |val|_max stayed comparable. Fingerprint of frozen
               attractors at large size: rule had rich spatial
               structure at REF_SIZE but locked to a uniform extreme
               (e.g. phase_separation pegged at ±1) when scaled up.
               Distinct from clamp_hi: bound may be sub-unit.

Grading (worst across sizes × channels):

    crit  -- nan at any size
    high  -- clamp_hi or blowup observed at any size, and baseline
             (smallest size) was clean
    med   -- value range grew between 10× and 100× vs baseline
    ok    -- bounded growth (<10×), no NaN, no clamp saturation
    n/a   -- rule has no u_dt uniform (pure CA, no PDE)
    err   -- crash during run

Skip filter mirrors `determinism.py`: viewport / agent / entity /
particle / element-CA presets are out of scope. The probe targets
shaders that integrate a PDE on a regular voxel grid via h_sq-scaled
Laplacians or explicit time-stepping.

Usage:
    python -m ca_debug.pde_stability_vs_size
    python -m ca_debug.pde_stability_vs_size --sizes 64,128,256,384 --steps 80
    python -m ca_debug.pde_stability_vs_size --rules quantum_hydrogen --json /tmp/p22.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time

import numpy as np


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'ok': 4, 'n/a': 5}

# Schrödinger family bakes a ±1e3 safety clamp into the shader. Detect
# saturation against that bound (allow tiny float slop on either side).
_CLAMP_BOUND = 1000.0
_CLAMP_TOL   = 10.0
# Anything growing this many times relative to the baseline size is
# almost certainly numerical, not physical. 100× was picked empirically
# from the quantum-family observations (size 96: max ≈ 1.0, size 256:
# max = 1000 -> 1000× jump).
_BLOWUP_RATIO   = 100.0
_BASELINE_OK_MAX = 10.0   # baseline must be physically bounded to flag
# Saturation/freeze detector: large size's spatial std collapses far
# below baseline's. Empirical thresholds: schnakenberg_3d at size 384
# gives 24x drop from baseline (real bug — dt_eff silent clamp); CH
# late-time coarsening gives <5x. 20x catches both bug families.
# Require baseline to have meaningful structure (std > 0.01).
_STD_COLLAPSE_RATIO = 20.0
_BASELINE_MIN_STD   = 0.01


def _gather_rules(args):
    from simulator import RULE_PRESETS, _resolve_composed_preset
    if args.rules:
        return [r.strip() for r in args.rules.split(',') if r.strip()]
    rules = []
    for r in sorted(RULE_PRESETS.keys()):
        try:
            preset = _resolve_composed_preset(r)
        except Exception:  # noqa: BLE001  -- bad presets caught later
            continue
        if preset.get('kind') == 'viewport':
            continue
        if preset.get('agent_count') or 'entity_arena' in preset:
            continue
        if (preset.get('passes') or [{}])[0].get('kind') == 'particle':
            continue
        # Pure-CA / element / fractal rules either have no u_dt or
        # legitimately saturate (binary states) -- skip.
        if preset.get('is_element_ca'):
            continue
        rules.append(r)
    return rules


def _channel_stats(g: np.ndarray) -> list[dict]:
    """One stats dict per channel."""
    out = []
    nch = g.shape[-1]
    for c in range(nch):
        a = g[..., c]
        nan = int(np.isnan(a).sum()) + int(np.isinf(a).sum())
        if nan == g.size // nch:
            out.append({'nan': nan, 'max_abs': float('nan'),
                        'min': float('nan'), 'max': float('nan'),
                        'std': float('nan')})
            continue
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            out.append({'nan': nan, 'max_abs': float('nan'),
                        'min': float('nan'), 'max': float('nan'),
                        'std': float('nan')})
            continue
        out.append({'nan': nan,
                    'max_abs': float(np.abs(finite).max()),
                    'min': float(finite.min()),
                    'max': float(finite.max()),
                    'std': float(finite.std())})
    return out


def _is_clamp_saturated(stat: dict) -> bool:
    m = stat['max_abs']
    if not (m == m):  # NaN
        return False
    return abs(m - _CLAMP_BOUND) <= _CLAMP_TOL


def _run_one(ctx, rule: str, *, size: int, steps: int, seed: int):
    """Returns (channel_stats list, error_str or None)."""
    from test_harness import HeadlessRunner
    try:
        r = HeadlessRunner(ctx, rule, size=size, seed=seed)
    except Exception as e:  # noqa: BLE001  -- per-rule failure recorded
        return None, f'{type(e).__name__}: {e}'
    try:
        for _ in range(steps):
            r.step()
        g = np.asarray(r.read_grid()).copy()
        return _channel_stats(g), None
    except Exception as e:  # noqa: BLE001  -- crashes/NaN caught here
        return None, f'{type(e).__name__}: {e}'
    finally:
        if hasattr(r, 'release'):
            try: r.release()
            except Exception: pass  # noqa: BLE001 -- cleanup best-effort


def _classify(rule: str, per_size: dict, sizes: list[int]) -> dict:
    """Walk per-size channel stats; pick worst signature."""
    base_size = sizes[0]
    base = per_size.get(base_size)
    if base is None or isinstance(base, str):
        return {'status': 'err', 'error': str(base or 'no baseline'),
                'sig': 'err', 'worst_size': base_size}
    base_max = max((c['max_abs'] for c in base
                    if c['max_abs'] == c['max_abs']), default=0.0)

    worst = {'status': 'ok', 'sig': 'ok', 'worst_size': base_size,
             'worst_channel': 0, 'worst_max': base_max,
             'baseline_max': base_max}
    sev_rank = {'ok': 0, 'med': 1, 'high': 2, 'crit': 3, 'err': 4}

    for s in sizes:
        stats = per_size.get(s)
        if isinstance(stats, str):
            cur = {'status': 'err', 'sig': 'err', 'worst_size': s,
                   'worst_channel': 0, 'worst_max': float('nan'),
                   'baseline_max': base_max, 'error': stats}
        else:
            cur = None
            for ci, st in enumerate(stats):
                base_st = base[ci] if ci < len(base) else None
                if st['nan'] > 0:
                    cand = {'status': 'crit', 'sig': 'nan',
                            'worst_size': s, 'worst_channel': ci,
                            'worst_max': st['max_abs'],
                            'baseline_max': base_max, 'nan': st['nan']}
                elif _is_clamp_saturated(st):
                    base_ok = base_max < _BASELINE_OK_MAX
                    cand = {'status': 'high' if base_ok else 'med',
                            'sig': 'clamp_hi', 'worst_size': s,
                            'worst_channel': ci,
                            'worst_max': st['max_abs'],
                            'baseline_max': base_max}
                else:
                    m = st['max_abs']
                    if m != m:
                        cand = None
                        continue
                    ratio = m / max(base_max, 1e-9)
                    if (ratio > _BLOWUP_RATIO
                            and base_max < _BASELINE_OK_MAX
                            and m > 10.0):
                        cand = {'status': 'high', 'sig': 'blowup',
                                'worst_size': s, 'worst_channel': ci,
                                'worst_max': m, 'baseline_max': base_max,
                                'ratio': ratio}
                    elif ratio > 10.0 and base_max < _BASELINE_OK_MAX:
                        cand = {'status': 'med', 'sig': 'grew',
                                'worst_size': s, 'worst_channel': ci,
                                'worst_max': m, 'baseline_max': base_max,
                                'ratio': ratio}
                    else:
                        cand = None
                # Independently check std-collapse on top of any other
                # signature: if rule pegged to a uniform extreme at
                # large size (std ≪ baseline std) and baseline was
                # structured, that's an attractor lock-up bug.
                if (cand is None and base_st is not None
                        and s != base_size):
                    bs = base_st.get('std', 0.0)
                    cs = st.get('std', 0.0)
                    if (bs == bs and cs == cs  # not NaN
                            and bs > _BASELINE_MIN_STD
                            and cs > 0
                            and bs / max(cs, 1e-12) > _STD_COLLAPSE_RATIO):
                        cand = {'status': 'high', 'sig': 'saturated',
                                'worst_size': s, 'worst_channel': ci,
                                'worst_max': st['max_abs'],
                                'baseline_max': base_max,
                                'std_ratio': bs / max(cs, 1e-12),
                                'baseline_std': bs, 'worst_std': cs}
                if cand is None:
                    continue
                if cur is None or sev_rank[cand['status']] > sev_rank[cur['status']]:
                    cur = cand
            if cur is None:
                cur = {'status': 'ok', 'sig': 'ok', 'worst_size': s,
                       'worst_channel': 0, 'worst_max': 0.0,
                       'baseline_max': base_max}
        if sev_rank[cur['status']] > sev_rank[worst['status']]:
            worst = cur
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sizes', default='64,96,128,192,256,384',
                    help='Comma-separated grid edges.')
    ap.add_argument('--steps', type=int, default=80,
                    help='Steps per size (uniform — instability shows '
                         'before 100 steps in every observed case).')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rules', help='Comma-separated rule subset.')
    ap.add_argument('--json', help='Write full per-(rule,size,channel) data.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER), default='med')
    ap.add_argument('--quiet', action='store_true', help='Suppress per-rule progress.')
    args = ap.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(',')]
    sizes.sort()

    import moderngl
    ctx = moderngl.create_standalone_context(require=430)

    rules = _gather_rules(args)
    print(f"\n=== pde_stability_vs_size — sizes={sizes} steps={args.steps} "
          f"rules={len(rules)} ===\n", file=sys.stderr)

    t0 = time.time()
    all_results = []
    for i, rule in enumerate(rules):
        per_size: dict = {}
        # Per-rule print buffer suppresses chatty preset bootstrap chatter.
        with contextlib.redirect_stdout(io.StringIO()):
            for s in sizes:
                stats, err = _run_one(ctx, rule, size=s,
                                      steps=args.steps, seed=args.seed)
                per_size[s] = err if err is not None else stats
        verdict = _classify(rule, per_size, sizes)
        verdict['rule'] = rule
        verdict['per_size'] = {s: ('err: ' + per_size[s])
                                  if isinstance(per_size[s], str)
                                  else per_size[s]
                               for s in sizes}
        all_results.append(verdict)
        if not args.quiet and (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(rules)} ({time.time()-t0:.1f}s)",
                  file=sys.stderr)

    # Print summary table sorted by severity then by rule name.
    all_results.sort(key=lambda r: (_SEV_ORDER[r['status']], r['rule']))
    threshold = _SEV_ORDER[args.severity]
    print(f"\n{'rule':<32}  {'sev':<5}  {'sig':<10}  {'worst@size':>11}  "
          f"{'worst_max':>11}  base_max")
    print('-' * 90)
    counts = {k: 0 for k in _SEV_ORDER}
    for r in all_results:
        counts[r['status']] += 1
        if _SEV_ORDER[r['status']] > threshold:
            continue
        wm = r.get('worst_max', float('nan'))
        bm = r.get('baseline_max', float('nan'))
        print(f"{r['rule']:<32}  {r['status']:<5}  {r['sig']:<10}  "
              f"{r['worst_size']:>11d}  {wm:>11.3e}  {bm:.3e}")
    print(f"\n[{time.time()-t0:.1f}s] crit={counts['crit']} "
          f"high={counts['high']} med={counts['med']} "
          f"ok={counts['ok']} err={counts['err']}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'sizes': sizes, 'steps': args.steps,
                       'results': all_results}, f, indent=2, default=str)
        print(f"wrote {args.json}")

    return 0 if counts['err'] + counts['crit'] + counts['high'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
