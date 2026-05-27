"""Probe #23 — Seed sensitivity at fixed grid size.

Run every PDE/continuous preset at REF_SIZE (=128 by default) with
default dt and default params, varying only the random seed across a
small ladder. For each rule, compare per-seed outcomes and flag any
rule where the choice of seed flips the numerical fate of the run.

This is orthogonal to Probe #22 (which varies grid size at fixed seed).
Probe #22 catches CFL-vs-h_sq bugs; Probe #23 catches initial-condition
fragility bugs:

    * NaN or crash on some seeds but not others (boundary IC edge cases)
    * ±1e3 schrodinger clamp_hi triggered on some seeds (IC amplitude
      pushes the wavepacket over the implicit stability threshold)
    * dynamic range varies by >100× across seeds with bounded baseline
      (initial perturbation amplitude controls whether the rule
      saturates or stays linear)

Detected signatures (per seed):

    nan        any non-finite value at end of run
    clamp_hi   |val|_max ∈ [990, 1010] for any channel
    blowup     |val|_max > 100× the median across seeds, while median
               itself stayed bounded < 10. (Median-relative because
               there is no canonical "baseline seed".)

Grading rule (per-rule worst):

    crit  -- NaN at any seed
    high  -- some seed clamp_hi or blowup AND at least one other seed
             stayed clean. Pure-seed-dependent failure.
    med   -- max_abs ratio (max over seeds / min over seeds) > 10
             while min stayed bounded < 10. Suggests fragile IC scaling.
    ok    -- all seeds within 10× of each other (any deterministic
             chaotic divergence is fine — the AGGREGATE statistics
             across seeds should be comparable for a healthy rule).
    n/a   -- rule skipped (viewport/agent/entity/element-CA/particle).
    err   -- crash during preset bootstrap.

Skip filter mirrors Probe #22.

Usage:
    python -m ca_debug.seed_sensitivity
    python -m ca_debug.seed_sensitivity --seeds 0,1,42,8675309 --steps 80
    python -m ca_debug.seed_sensitivity --rules quantum_hydrogen --json /tmp/p23.json
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

_CLAMP_BOUND = 1000.0
_CLAMP_TOL   = 10.0
_BLOWUP_RATIO   = 100.0
_GREW_RATIO     = 10.0
_BASELINE_OK_MAX = 10.0

# Default seed ladder: deliberately diverse — small ints, Numerical-
# Recipes classic, Knuth multiplicative constant, well-known cultural.
_DEFAULT_SEEDS = [0, 1, 42, 8675309, 2654435761]


def _gather_rules(args):
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
        if preset.get('is_element_ca'):
            continue
        rules.append(r)
    return rules


def _channel_stats(g: np.ndarray) -> list[dict]:
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
    if not (m == m):
        return False
    return abs(m - _CLAMP_BOUND) <= _CLAMP_TOL


def _run_one(ctx, rule: str, *, size: int, steps: int, seed: int):
    from test_harness import HeadlessRunner
    try:
        r = HeadlessRunner(ctx, rule, size=size, seed=seed)
    except Exception as e:  # noqa: BLE001
        return None, f'{type(e).__name__}: {e}'
    try:
        for _ in range(steps):
            r.step()
        g = np.asarray(r.read_grid()).copy()
        return _channel_stats(g), None
    except Exception as e:  # noqa: BLE001
        return None, f'{type(e).__name__}: {e}'
    finally:
        if hasattr(r, 'release'):
            try: r.release()
            except Exception: pass  # noqa: BLE001


def _classify(rule: str, per_seed: dict, seeds: list[int]) -> dict:
    """Compare seed-to-seed outcomes; pick worst signature."""
    # Collect per-seed max_abs (across all channels), nan-count, clamp_hi-fired
    seed_max  = {}
    seed_nan  = {}
    seed_clmp = {}
    seed_err  = {}
    for s in seeds:
        v = per_seed.get(s)
        if isinstance(v, str):
            seed_err[s] = v
            continue
        if v is None:
            seed_err[s] = 'empty'
            continue
        max_per_ch = [c['max_abs'] for c in v if c['max_abs'] == c['max_abs']]
        seed_max[s]  = max(max_per_ch) if max_per_ch else float('nan')
        seed_nan[s]  = sum(c['nan'] for c in v)
        seed_clmp[s] = any(_is_clamp_saturated(c) for c in v)

    # If every seed errored, that's an 'err' (rule-level breakage).
    if len(seed_err) == len(seeds):
        return {'status': 'err', 'sig': 'err',
                'error': next(iter(seed_err.values())),
                'n_err': len(seed_err), 'n_ok': 0}

    # Bucket seeds.
    bad_nan  = [s for s in seeds if seed_nan.get(s, 0) > 0]
    bad_clmp = [s for s in seeds if seed_clmp.get(s, False)]
    good     = [s for s in seeds
                if s in seed_max and not seed_nan.get(s, 0)
                and not seed_clmp.get(s, False)]
    n_err    = len(seed_err)
    n_total  = len(seeds)

    # Top signatures: differential failure (some seeds bad, others good).
    if bad_nan and (good or len(bad_nan) < n_total - n_err):
        return {'status': 'crit', 'sig': 'nan_some_seeds',
                'bad_seeds': bad_nan, 'good_seeds': good,
                'n_bad': len(bad_nan), 'n_ok': len(good),
                'n_err': n_err}
    if bad_nan:  # all (non-errored) seeds NaN'd — still a crit, but not differential
        return {'status': 'crit', 'sig': 'nan_all_seeds',
                'bad_seeds': bad_nan, 'good_seeds': good,
                'n_bad': len(bad_nan), 'n_ok': len(good),
                'n_err': n_err}
    if bad_clmp and good:
        # The interesting case: seed flips Schrödinger ±1e3 clamp.
        return {'status': 'high', 'sig': 'clamp_hi_some_seeds',
                'bad_seeds': bad_clmp, 'good_seeds': good,
                'n_bad': len(bad_clmp), 'n_ok': len(good),
                'n_err': n_err}

    # Differential blowup: ratio across seeds.
    if len(seed_max) >= 2:
        vals = list(seed_max.values())
        med  = float(np.median(vals))
        mx   = max(vals)
        mn   = min(vals)
        if med < _BASELINE_OK_MAX:
            ratio_blow = mx / max(med, 1e-9)
            if ratio_blow > _BLOWUP_RATIO and mx > 10.0:
                worst_seed = max(seed_max, key=seed_max.get)
                return {'status': 'high', 'sig': 'blowup_some_seeds',
                        'worst_seed': worst_seed,
                        'worst_max': mx, 'median_max': med,
                        'ratio': ratio_blow, 'n_err': n_err}
            ratio_grew = mx / max(mn, 1e-9) if mn > 0 else float('inf')
            if ratio_grew > _GREW_RATIO and mn < _BASELINE_OK_MAX:
                return {'status': 'med', 'sig': 'wide_range',
                        'worst_max': mx, 'min_max': mn,
                        'ratio': ratio_grew, 'n_err': n_err}

    if n_err > 0:
        # Some seeds errored but not all → robustness issue.
        return {'status': 'high', 'sig': 'err_some_seeds',
                'bad_seeds': list(seed_err),
                'n_bad': n_err, 'n_ok': len(good)}

    return {'status': 'ok', 'sig': 'ok',
            'n_ok': len(good), 'n_err': n_err}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--size', type=int, default=128,
                    help='Single grid size (probe varies seed instead).')
    ap.add_argument('--steps', type=int, default=80)
    ap.add_argument('--seeds', default=','.join(str(s) for s in _DEFAULT_SEEDS),
                    help='Comma-separated seed ladder.')
    ap.add_argument('--rules', help='Comma-separated rule subset.')
    ap.add_argument('--json', help='Write per-(rule,seed,channel) data.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER), default='med')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(',')]

    import moderngl
    ctx = moderngl.create_standalone_context(require=430)

    rules = _gather_rules(args)
    print(f"\n=== seed_sensitivity — size={args.size} steps={args.steps} "
          f"seeds={seeds} rules={len(rules)} ===\n", file=sys.stderr)

    t0 = time.time()
    all_results = []
    for i, rule in enumerate(rules):
        per_seed: dict = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for s in seeds:
                stats, err = _run_one(ctx, rule, size=args.size,
                                      steps=args.steps, seed=s)
                per_seed[s] = err if err is not None else stats
        verdict = _classify(rule, per_seed, seeds)
        verdict['rule'] = rule
        verdict['per_seed'] = {s: ('err: ' + per_seed[s])
                                   if isinstance(per_seed[s], str)
                                   else per_seed[s]
                               for s in seeds}
        all_results.append(verdict)
        if not args.quiet and (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(rules)} ({time.time()-t0:.1f}s)",
                  file=sys.stderr)

    all_results.sort(key=lambda r: (_SEV_ORDER[r['status']], r['rule']))
    threshold = _SEV_ORDER[args.severity]
    print(f"\n{'rule':<32}  {'sev':<5}  {'sig':<22}  n_bad/n_ok  detail")
    print('-' * 96)
    counts = {k: 0 for k in _SEV_ORDER}
    for r in all_results:
        counts[r['status']] += 1
        if _SEV_ORDER[r['status']] > threshold:
            continue
        detail = ''
        if 'ratio' in r:
            detail = f"ratio={r['ratio']:.1f}× worst_max={r.get('worst_max', r.get('worst_max', 0)):.2e}"
        elif 'bad_seeds' in r:
            detail = f"bad={r['bad_seeds']}"
        n_b = r.get('n_bad', 0)
        n_o = r.get('n_ok', 0)
        print(f"{r['rule']:<32}  {r['status']:<5}  {r['sig']:<22}  "
              f"{n_b:>4}/{n_o:<4}  {detail}")
    print(f"\n[{time.time()-t0:.1f}s] crit={counts['crit']} "
          f"high={counts['high']} med={counts['med']} "
          f"ok={counts['ok']} err={counts['err']}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'size': args.size, 'steps': args.steps,
                       'seeds': seeds, 'results': all_results},
                      f, indent=2, default=str)
        print(f"wrote {args.json}")

    return 0 if counts['err'] + counts['crit'] + counts['high'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
