"""Init-coverage & opportunity report: where is it worth adding inits?

``init_variants.py`` answers "is this init broken?".  ``init_bench.py``
answers "is this *candidate* good?".  This module answers the question
that comes before both: **which rules deserve attention at all?**

For every eligible rule it runs two cheap default-param trials (a short
and a long horizon) and cross-references the result against how many
init variants the preset already declares.  That combination surfaces
four distinct problems that look identical from a score table alone:

    saturated   the measure config counts almost every voxel as alive
                (median_alive > 0.90), so the score is pinned no matter
                what the rule does.  This is a *scorer* bug, not a rule
                bug -- it is exactly what floored schnakenberg_3d,
                brusselator_3d, sandpile_3d and hodgepodge_3d.  Fix the
                measure branch in test_harness.py before touching inits.

    empty       the init leaves an essentially blank field that never
                grows (median_alive < 0.01 at both horizons).  The rule
                never had a chance; the declared default init is wrong.

    slow-growth a sparse seed that *is* filling out, just slowly.  Looks
                identical to ``empty`` at a short horizon, which is why
                it is separated: fcc_dla reads as blank at 300 steps and
                scores 0.797 at 800.  Sweep further before judging.

    mis-horizon the rule scores well early and collapses by the long
                horizon.  Not a bad rule -- a bad step count.  Check
                STEPS_BY_RULE before concluding anything else.

    thin        the rule already scores well but declares <=1 init.
                This is the highest-return place to add variants: the
                kinetics are proven, only the seeding is unexplored.

Rules are ranked by opportunity, so the top of the output is the
shortlist of where new inits will actually pay off.

A note on ``--size``: several rule families are scale-sensitive, and
probing them below the canonical 64 manufactures phantom bugs.  At
size 40 the Lenia family and the dendrite growers read as ``empty``
with score 0.000 -- their kernels simply do not fit -- and ``eden_3d``
reads as ``floor``.  All are healthy at 64.  Do not lower the default
unless you are prepared to re-check every flag it produces.

Usage::

    python -m ca_debug.init_coverage
    python -m ca_debug.init_coverage --flag thin
    python -m ca_debug.init_coverage --rules flow_turing,schnakenberg_3d
    python -m ca_debug.init_coverage --json coverage.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

SHORT_STEPS = 60
LONG_STEPS = 300
# Canonical scale. Lower values produce false 'empty'/'floor' flags on
# scale-sensitive families (Lenia kernels, dendrite growth).
DEFAULT_SIZE = 64

SATURATION_ALIVE = 0.90
EMPTY_ALIVE = 0.01
# 'empty' means the rule is genuinely not working, so it must also be
# scoring badly -- some rules (volcanic, wave fields) legitimately run at a
# very low alive ratio while scoring fine.
EMPTY_MAX_SCORE = 0.20
# A sparse seed that multiplies its alive ratio by this much between the
# short and long horizon is growing, not dead -- it just needs more steps.
# fcc_dla manages 2.35x, hence the low bar.
GROWTH_FACTOR = 2.0
FLOOR_SCORE = 0.105          # MIN_QUALITY is 0.10; allow float slop.
HEALTHY_SCORE = 0.50
MIS_HORIZON_FACTOR = 2.0

_FLAG_ORDER = ['saturated', 'empty', 'slow-growth', 'floor', 'mis-horizon',
               'thin', 'ok']


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
    return rules


def _trial(ctx, rule: str, size: int, seed: int, steps: int) -> dict | None:
    from test_harness import run_trial
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_trial(ctx, rule, size=size, seed=seed, steps=steps)
    except Exception:  # noqa: BLE001  a broken rule must not stop the sweep
        return None
    return {'score': float(r['score']),
            'median_alive': float(r['median_alive']),
            'mean_activity': float(r['mean_activity']),
            'measure_mode': r.get('measure_mode')}


def _classify(short: dict | None, long_: dict | None,
              n_var: int, rule: str) -> tuple[str, str]:
    """Return (flag, note). Order matters: report the root cause, not
    the symptom it produces downstream."""
    if short is None and long_ is None:
        return 'err', 'both trials crashed'
    ref = short or long_
    s_short = short['score'] if short else 0.0
    s_long = long_['score'] if long_ else 0.0
    best = max(s_short, s_long)

    # Saturation and emptiness are measurement problems; they mask
    # everything else, so they are checked first.  Both are only
    # actionable when the score is *also* unhealthy: oscillatory field
    # rules (sine-Gordon, wave) legitimately sit at a low alive ratio
    # while scoring well, and flagging those is pure noise.
    alive = ref['median_alive']
    if alive > SATURATION_ALIVE and best < HEALTHY_SCORE:
        return 'saturated', (f'median_alive={alive:.3f} — measure "'
                             f'{ref["measure_mode"]}" counts ~everything '
                             f'as alive; fix the measure branch first')
    if alive < EMPTY_ALIVE and best < EMPTY_MAX_SCORE:
        a_s = short['median_alive'] if short else 0.0
        a_l = long_['median_alive'] if long_ else 0.0
        if a_l > max(a_s, 1e-9) * GROWTH_FACTOR:
            return 'slow-growth', (
                f'alive {a_s:.5f} -> {a_l:.5f} over {SHORT_STEPS}->'
                f'{LONG_STEPS}: a sparse seed still filling out, not dead. '
                f'Sweep wider before judging: python -m ca_debug.init_bench '
                f'{rule} --horizons 300,600,1200')
        return 'empty', (f'median_alive={alive:.5f} and no growth by '
                         f'{LONG_STEPS} steps — the declared default init or '
                         f'its default params are suspect')

    if s_short >= MIS_HORIZON_FACTOR * max(s_long, 1e-6) and s_short >= 0.3:
        return 'mis-horizon', (f'{s_short:.3f}@{SHORT_STEPS} vs '
                               f'{s_long:.3f}@{LONG_STEPS} — run this rule '
                               f'nearer {SHORT_STEPS} steps')
    if best <= FLOOR_SCORE:
        return 'floor', 'floors at both horizons — every search trial rejected'
    if best >= HEALTHY_SCORE and n_var <= 1:
        return 'thin', (f'scores {best:.3f} with only '
                        f'{n_var} variant(s) — proven kinetics, unexplored '
                        f'seeding')
    return 'ok', ''


def _opportunity(flag: str, best: float, n_var: int) -> float:
    """Higher = more worth working on. Used only for sort order."""
    weight = {'saturated': 100, 'empty': 90, 'mis-horizon': 80,
              'thin': 70, 'slow-growth': 50, 'floor': 40,
              'ok': 0, 'err': 95}.get(flag, 0)
    return weight + best * 10 - n_var


def survey(ctx, rules: list[str], *, size: int, seed: int,
           short_steps: int, long_steps: int, progress: bool = True) -> list[dict]:
    from simulator import _resolve_composed_preset
    rows = []
    for i, rule in enumerate(rules, 1):
        if progress:
            sys.stdout.write(f'\r[{i:>3}/{len(rules)}] {rule:<44}')
            sys.stdout.flush()
        try:
            preset = _resolve_composed_preset(rule)
        except Exception:  # noqa: BLE001
            continue
        n_var = len(preset.get('init_variants') or [])
        n_extra = len(preset.get('init_extras') or [])
        short = _trial(ctx, rule, size, seed, short_steps)
        long_ = _trial(ctx, rule, size, seed, long_steps)
        flag, note = _classify(short, long_, n_var, rule)
        best = max((t['score'] for t in (short, long_) if t), default=0.0)
        rows.append({
            'rule': rule,
            'lattice': preset.get('lattice', 'cubic'),
            'n_variants': n_var,
            'n_extras': n_extra,
            'score_short': short['score'] if short else None,
            'score_long': long_['score'] if long_ else None,
            'median_alive': (short or long_ or {}).get('median_alive'),
            'measure_mode': (short or long_ or {}).get('measure_mode'),
            'flag': flag,
            'note': note,
            'opportunity': _opportunity(flag, best, n_var),
        })
    if progress:
        sys.stdout.write('\r' + ' ' * 72 + '\r')
    return rows


def _print_report(rows: list[dict], only_flag: str | None) -> None:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r['flag']] = counts.get(r['flag'], 0) + 1
    print(f'init coverage — {len(rows)} rules')
    for f in _FLAG_ORDER + ['err']:
        if counts.get(f):
            print(f'    {f:<12} {counts[f]:>3}')
    print()

    shown = [r for r in rows if not only_flag or r['flag'] == only_flag]
    shown.sort(key=lambda r: -r['opportunity'])
    if only_flag:
        shown = [r for r in shown if r['flag'] == only_flag]
    else:
        shown = [r for r in shown if r['flag'] != 'ok']

    print(f"{'FLAG':<12} {'RULE':<32} {'VAR':>3} {'EXT':>3} "
          f"{'@' + str(SHORT_STEPS):>7} {'@' + str(LONG_STEPS):>7}  NOTE")
    print('-' * 130)
    for r in shown:
        ss = '  -- ' if r['score_short'] is None else f"{r['score_short']:7.3f}"
        sl = '  -- ' if r['score_long'] is None else f"{r['score_long']:7.3f}"
        print(f"{r['flag']:<12} {r['rule']:<32} {r['n_variants']:>3} "
              f"{r['n_extras']:>3} {ss} {sl}  {r['note']}")
    if not shown:
        print('  (nothing flagged)')


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rules', help='Comma-separated rules (default: all eligible).')
    ap.add_argument('--size', type=int, default=DEFAULT_SIZE)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--short-steps', type=int, default=SHORT_STEPS)
    ap.add_argument('--long-steps', type=int, default=LONG_STEPS)
    ap.add_argument('--skip-flagship', action='store_true')
    ap.add_argument('--flag', choices=_FLAG_ORDER + ['err'],
                    help='Show only rules with this flag.')
    ap.add_argument('--json', help='Write the full row list to this path.')
    args = ap.parse_args(argv)

    from test_harness import create_headless_context
    window, ctx = create_headless_context()  # noqa: F841  keep window alive

    rules = _select_rules(args)
    t0 = time.perf_counter()
    rows = survey(ctx, rules, size=args.size, seed=args.seed,
                  short_steps=args.short_steps, long_steps=args.long_steps)
    elapsed = time.perf_counter() - t0
    _print_report(rows, args.flag)
    print(f'\n  {elapsed:.1f}s')

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f'  wrote {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
