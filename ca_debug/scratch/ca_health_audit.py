"""CA health audit — runs every CA at preset defaults for a moderately
long trajectory and classifies the result.

For each rule we run two trials:
  - Trial A: preset defaults (params=None, dt=None) — the "showroom" run.
  - Trial B: same params but seed perturbed — checks reproducibility/sensitivity.

We then classify the rule into one of:
  ok            — alive in (0.005, 0.95) at end AND mean_activity > 1e-4
  dead          — alive < 0.001 throughout
  born_dead     — alive > 0.05 early but < 0.001 by end (collapsed)
  saturated     — alive > 0.95 from early on AND activity ~0
  noise         — activity > 0.5 throughout AND alive ≈ 0.5 (incoherent)
  exploded      — has_nan or has_inf
  silent        — alive > 0.005 but activity < 1e-5 (frozen, possibly OK)
  weak          — alive in (0, 0.005] at end with no NaN — barely alive

Output a sorted Markdown table. Run with:
    .venv/bin/python ca_health_audit.py
    .venv/bin/python ca_health_audit.py --steps 300 --size 64
"""
import argparse
import json
import math
import sys
import time
from typing import Any


def classify(history: list[dict], final_alive: float, final_act: float,
             mean_act: float, has_nan: bool, has_inf: bool) -> tuple[str, str]:
    """Return (label, reason)."""
    if has_nan or has_inf:
        return 'exploded', 'NaN/Inf in field'
    if not history:
        return 'exploded', 'no metric samples'

    alive = [m['alive_ratio'] for m in history]
    act   = [m['activity']    for m in history]
    a_max = max(alive)
    a_end = alive[-1]
    act_mean = sum(act) / len(act) if act else 0.0
    act_end  = act[-1]

    # dead: never above 0.001
    if a_max < 0.001:
        return 'dead', f'max alive={a_max:.2e} over {len(history)} samples'

    # born-dead: peaked then collapsed
    if a_max > 0.05 and a_end < 0.001:
        return 'born_dead', f'peaked at {a_max:.3f}, ended at {a_end:.2e}'

    # saturated: full grid, no motion (true equilibrium / runaway fill)
    if a_end > 0.95 and act_mean < 1e-3:
        return 'saturated', f'alive={a_end:.3f} act={act_mean:.2e}'

    # noise: all-active, all-changing — but with no temporal coherence.
    # A working excitable medium (Greenberg–Hastings scroll waves) or
    # ordered phase (active nematic) also runs at ~50% alive with high
    # activity, but its alive_ratio cycles rather than sitting flat.
    # Discriminate by alive-ratio variance: true noise has near-constant
    # alive ratio, coherent dynamics swing it.
    if act_mean > 0.5 and 0.3 < a_end < 0.7:
        alive_std = (sum((a - sum(alive)/len(alive))**2 for a in alive) / len(alive)) ** 0.5
        # Empirically: GH cycles by ~0.10–0.15, true noise stays within
        # ~0.02 of its mean. 0.05 is well below GH and well above noise.
        if alive_std < 0.05:
            return 'noise', f'act_mean={act_mean:.3f} alive≈0.5 std={alive_std:.3f}'

    # silent: appreciable population, but completely frozen
    if a_end > 0.005 and act_mean < 1e-5:
        return 'silent', f'alive={a_end:.3f} act={act_mean:.2e} (frozen)'

    # weak: barely alive at end
    if 0.0 < a_end <= 0.005:
        return 'weak', f'alive={a_end:.2e} (very sparse)'

    return 'ok', f'alive={a_end:.3f} act={act_mean:.2e}'


def run_one(ctx, rule, *, size: int, steps: int, seed: int) -> dict[str, Any]:
    from test_harness import run_trial
    t0 = time.perf_counter()
    try:
        r = run_trial(ctx, rule, size=size, seed=seed, steps=steps,
                      sample_interval=max(5, steps // 12), verbose=False)
    except Exception as e:  # noqa: BLE001  trial may crash on bad params, score=0
        return {'ok': False, 'error': f'{type(e).__name__}: {e}',
                'dt_sec': time.perf_counter() - t0}
    history = r.get('history', [])
    label, reason = classify(
        history,
        final_alive=r.get('final_alive', float('nan')),
        final_act  =r.get('final_activity', float('nan')),
        mean_act   =r.get('mean_activity', float('nan')),
        has_nan    =r.get('has_nan', False),
        has_inf    =r.get('has_inf', False),
    )
    return {
        'ok': True,
        'label': label,
        'reason': reason,
        'final_alive':   float(r.get('final_alive', float('nan'))),
        'final_activity':float(r.get('final_activity', float('nan'))),
        'mean_activity': float(r.get('mean_activity', float('nan'))),
        'score':         float(r.get('score', float('nan'))),
        'has_nan':       bool(r.get('has_nan', False)),
        'has_inf':       bool(r.get('has_inf', False)),
        'effective_size':int(r.get('size', size)),
        'samples':       len(history),
        'dt_sec':        time.perf_counter() - t0,
    }


# Severity order for sorting (worst first)
SEVERITY = {
    'exploded':  0,
    'dead':      1,
    'born_dead': 2,
    'saturated': 3,
    'noise':     4,
    'silent':    5,
    'weak':      6,
    'ok':        9,
    '_error':   -1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=200,
                    help='trajectory length (default 200; long enough for '
                         'PDEs/crystals to develop)')
    ap.add_argument('--size', type=int, default=48,
                    help='grid size for cheap rules (preset default_size '
                         'auto-bumps quantum/lenia)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 137],
                    help='seed list (multiple = robustness check)')
    ap.add_argument('--out', default='ca_health_report.json')
    ap.add_argument('--rule', default=None,
                    help='run a single rule for debugging')
    args = ap.parse_args()

    sys.path.insert(0, '.')
    from simulator import RULE_PRESETS
    from test_harness import create_headless_context, destroy_context

    rules = sorted(RULE_PRESETS.keys()) if args.rule is None else [args.rule]
    print(f'[health] {len(rules)} rule(s), seeds={args.seeds}, '
          f'size={args.size}, steps={args.steps}')

    window, ctx = create_headless_context()
    results: dict[str, Any] = {}
    t_start = time.perf_counter()
    for i, name in enumerate(rules):
        preset = RULE_PRESETS[name]
        # Per-preset opt-out: presets like 'sandbox' are intentionally
        # empty for user brush-mode building and would always read 'dead'.
        if preset.get('audit_skip'):
            results[name] = {'label': 'skipped',
                             'reason': preset.get('audit_skip_reason',
                                                  'audit_skip=True in preset')}
            print(f'  [{i+1:>3}/{len(rules)}] {name:<32}   '
                  f'skipped    {results[name]["reason"]}')
            continue
        # Per-preset audit-step override: slow-growth rules (e.g.
        # crystal_growth at near-isotropic anisotropy) need many more
        # steps than 200 to develop visible structure.
        steps = int(preset.get('audit_steps', args.steps))
        per_seed = [run_one(ctx, name, size=args.size, steps=steps,
                            seed=seed) for seed in args.seeds]
        # If any seed errored, surface that. Otherwise pick the WORST
        # severity across seeds — a rule that's broken on some seeds and
        # fine on others is still suspicious.
        if not all(r.get('ok') for r in per_seed):
            err = next(r for r in per_seed if not r.get('ok'))
            results[name] = {'label': '_error', 'reason': err.get('error'),
                             'per_seed': per_seed}
            print(f'  [{i+1:>3}/{len(rules)}] {name:<32}  ERROR  '
                  f'{err.get("error")}')
            continue
        # Combine: take worst label, list per-seed labels
        worst = min(per_seed, key=lambda r: SEVERITY.get(r['label'], 99))
        labels = [r['label'] for r in per_seed]
        results[name] = {
            'label': worst['label'],
            'reason': worst['reason'],
            'all_labels': labels,
            'mean_alive': sum(r['final_alive'] for r in per_seed) / len(per_seed),
            'mean_activity': sum(r['mean_activity'] for r in per_seed) / len(per_seed),
            'mean_score': sum(r['score'] for r in per_seed) / len(per_seed),
            'effective_size': per_seed[0]['effective_size'],
            'mean_dt_sec': sum(r['dt_sec'] for r in per_seed) / len(per_seed),
            'per_seed': per_seed,
        }
        marker = '✗' if worst['label'] != 'ok' else ' '
        consistent = '' if len(set(labels)) == 1 else f' (varies: {labels})'
        print(f'  [{i+1:>3}/{len(rules)}] {name:<32} {marker} '
              f'{worst["label"]:<10} {worst["reason"]}{consistent}')

    destroy_context(window)
    elapsed = time.perf_counter() - t_start
    print(f'\n[health] {elapsed:.1f}s total')

    # Write JSON
    payload = {
        'meta': {'steps': args.steps, 'size': args.size, 'seeds': args.seeds,
                 'elapsed_sec': elapsed},
        'rules': results,
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f'[health] wrote {args.out}')

    # Markdown summary
    by_label: dict[str, list[str]] = {}
    for name, r in results.items():
        by_label.setdefault(r['label'], []).append(name)
    print('\n══════ SUMMARY ══════')
    for label in sorted(by_label, key=lambda l: SEVERITY.get(l, 99)):
        names = sorted(by_label[label])
        print(f'\n{label:>10}  ({len(names)} rule{"s" if len(names)!=1 else ""})')
        for n in names:
            r = results[n]
            extra = ''
            if r.get('all_labels') and len(set(r['all_labels'])) > 1:
                extra = f'  [varies: {r["all_labels"]}]'
            print(f'           - {n:<32} {r.get("reason", "")}{extra}')


if __name__ == '__main__':
    main()
