"""Generate preset_overrides.json from discoveries.

Strategy:
  1. For each rule, gather top-K discoveries by score.
  2. Compute median per param + median dt.
  3. Round to a "human" precision based on the param range.
  4. Decide whether to override:
       - Health-broken rules (dead/weak/noise/saturated/born_dead at defaults
         in ca_health_report.json) → ALWAYS override if they have ≥
         --min-discoveries.
       - Healthy rules → override only when max distance from current
         default exceeds --min-dist-ok (default 0.40, conservative; the
         existing default already works, so don't churn unless the search
         strongly disagrees).
  5. Sandbox is excluded (intentionally empty by design).

Output: preset_overrides.json (dry-run prints to stdout instead).

Usage:
    .venv/bin/python apply_preset_tuning.py --dry-run
    .venv/bin/python apply_preset_tuning.py --apply
    .venv/bin/python apply_preset_tuning.py --apply --include-ok
"""
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict

EXCLUDE_RULES = {'sandbox'}  # by-design empty (brush mode)

# Physics-derived rules whose params (damping, wave speed, ℏ/2m, viscosity,
# ...) carry physical meaning beyond "tune for an interesting pattern".
# The random-search auto-tuner picks parameter combinations that produce
# visually striking artefacts but routinely fight the underlying PDE
# (e.g. saving Damping=9.3 for em_wave, which kills the field by 90 %
# per step). Excluding them keeps hand-tuned defaults intact.
EXCLUDE_PHYSICS_RULES = {
    'em_wave',
    'wave_3d',
    'sine_gordon_3d',
    'dirac_3d',
    'compressible_euler_3d',
    # all quantum_* (Schrödinger leapfrog)
    'quantum_hydrogen', 'quantum_orbital', 'quantum_element',
    'quantum_wavepacket', 'quantum_harmonic', 'quantum_tunneling',
    'quantum_double_slit', 'quantum_molecule', 'quantum_antibonding',
    'quantum_selfinteract',
}
EXCLUDE_RULES |= EXCLUDE_PHYSICS_RULES


def round_smart(v: float, lo: float, hi: float) -> float:
    if not math.isfinite(v):
        return v
    span = max(abs(hi - lo), 1e-9)
    if span >= 100:    step = 1.0
    elif span >= 10:   step = 0.1
    elif span >= 1:    step = 0.01
    elif span >= 0.1:  step = 0.001
    else:              step = 0.0001
    return round(v / step) * step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--discoveries', default='discoveries.json')
    ap.add_argument('--health',      default='ca_health_report.json')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--min-discoveries', type=int, default=10,
                    help='need at least this many to consider override')
    ap.add_argument('--min-dist-broken', type=float, default=0.10,
                    help='for broken-default rules, override if any '
                         'param/dt distance exceeds this (default 0.10)')
    ap.add_argument('--min-dist-ok', type=float, default=0.40,
                    help='for healthy rules, override only if a param/dt '
                         'distance exceeds this (default 0.40)')
    ap.add_argument('--include-ok', action='store_true',
                    help='also patch healthy rules (subject to min-dist-ok)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--apply',   action='store_true')
    ap.add_argument('--out', default='preset_overrides.json')
    args = ap.parse_args()
    if not args.dry_run and not args.apply:
        ap.error('pass --dry-run or --apply')

    sys.path.insert(0, '.')
    # Load presets WITHOUT existing overrides so we patch against pristine
    # defaults — re-running the script must converge.
    import os
    os.environ['CA_DISABLE_PRESET_OVERRIDES'] = '1'
    from simulator import RULE_PRESETS

    discs = json.load(open(args.discoveries))
    if isinstance(discs, dict):
        discs = discs.get('discoveries', discs)
    by_rule = defaultdict(list)
    for d in discs:
        by_rule[d['rule']].append(d)

    health = {}
    try:
        h = json.load(open(args.health))
        health = h.get('rules', {})
    except FileNotFoundError:
        print(f'[tuning] no health report at {args.health} — '
              f'treating all rules as healthy')

    BROKEN = {'dead', 'weak', 'noise', 'saturated', 'born_dead', 'silent',
              'exploded'}

    overrides = {}
    notes = []
    for rule in sorted(RULE_PRESETS.keys()):
        if rule in EXCLUDE_RULES:
            continue
        hits = by_rule.get(rule, [])
        if len(hits) < args.min_discoveries:
            continue
        preset = RULE_PRESETS[rule]
        cur_params = preset.get('params', {})
        ranges     = preset.get('param_ranges', {})
        cur_dt     = preset.get('dt')
        dt_range   = preset.get('dt_range')

        is_broken = health.get(rule, {}).get('label') in BROKEN
        threshold = args.min_dist_broken if is_broken else args.min_dist_ok
        if not is_broken and not args.include_ok:
            continue

        hits_sorted = sorted(hits, key=lambda x: x.get('score', 0.0),
                             reverse=True)[:args.top]

        # dt
        dt_vals = [h.get('dt') for h in hits_sorted if h.get('dt') is not None]
        med_dt = statistics.median(dt_vals) if dt_vals else None
        dt_dist = 0.0
        new_dt = None
        if med_dt is not None and cur_dt is not None and dt_range:
            dlo, dhi = dt_range
            if dhi > dlo:
                dt_dist = abs(med_dt - cur_dt) / (dhi - dlo)
                if dt_dist >= threshold:
                    new_dt = round_smart(med_dt, dlo, dhi)
                    # safety: clamp into valid range
                    new_dt = max(dlo, min(dhi, new_dt))

        # params
        new_params = {}
        max_dist = dt_dist
        for pname, cur_v in cur_params.items():
            vals = [h['params'].get(pname) for h in hits_sorted
                    if isinstance(h.get('params'), dict)
                    and h['params'].get(pname) is not None
                    and isinstance(h['params'].get(pname), (int, float))]
            if not vals:
                continue
            med = statistics.median(vals)
            lo, hi = ranges.get(pname, (None, None))
            if lo is None or hi is None or hi <= lo:
                continue
            dist = abs(med - cur_v) / (hi - lo)
            max_dist = max(max_dist, dist)
            if dist >= threshold:
                v = round_smart(med, lo, hi)
                v = max(lo, min(hi, v))
                # Preserve int-typed params (e.g. crystal Shape)
                if isinstance(cur_v, int) and float(v).is_integer():
                    v = int(v)
                new_params[pname] = v

        patch = {}
        if new_params:
            patch['params'] = new_params
        if new_dt is not None:
            patch['dt'] = new_dt
        if patch:
            overrides[rule] = patch
            notes.append(
                f'  {rule:<32} {"BROKEN" if is_broken else "ok":<6} '
                f'n={len(hits):>4}  max_dist={max_dist:.2f}  '
                f'patches: {list(patch.keys())} '
                + (f'+{len(new_params)}p' if new_params else '')
            )

    print(f'[tuning] {len(overrides)} rule(s) will be patched:')
    for line in notes:
        print(line)

    payload = {
        'meta': {
            'top_k': args.top,
            'min_discoveries': args.min_discoveries,
            'min_dist_broken': args.min_dist_broken,
            'min_dist_ok': args.min_dist_ok,
            'include_ok': args.include_ok,
        },
        'overrides': overrides,
    }
    if args.dry_run:
        print('\n--- preset_overrides.json (dry run) ---')
        print(json.dumps(payload, indent=2))
    else:
        with open(args.out, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'\n[tuning] wrote {args.out}')


if __name__ == '__main__':
    main()
