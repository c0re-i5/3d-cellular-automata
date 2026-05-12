"""Propose better default presets based on what discoveries reveal.

For each rule with at least N discoveries, look at the top-K by score and:
  - propose median dt and per-param median values (rounded sensibly)
  - report how far the current preset is from that median (in units of
    its param_range)

Outputs a markdown report. Does not modify simulator.py — just proposes.

Usage:
    .venv/bin/python preset_tuning_audit.py
    .venv/bin/python preset_tuning_audit.py --top 20 --min-discoveries 10
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from typing import Any


def round_smart(v: float, lo: float, hi: float) -> float:
    """Round to ~3 sig figs of the parameter's range."""
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
    ap.add_argument('--top', type=int, default=20,
                    help='use top-K discoveries by score (default 20)')
    ap.add_argument('--min-discoveries', type=int, default=5,
                    help='skip rules with fewer than this many discoveries')
    ap.add_argument('--out', default='preset_tuning_report.md')
    args = ap.parse_args()

    import sys
    sys.path.insert(0, '.')
    from simulator import RULE_PRESETS

    discs = json.load(open(args.discoveries))
    if isinstance(discs, dict):
        discs = discs.get('discoveries', discs)

    by_rule: dict[str, list[dict]] = defaultdict(list)
    for d in discs:
        by_rule[d['rule']].append(d)

    lines: list[str] = []
    lines.append('# Preset tuning audit')
    lines.append('')
    lines.append(f'Source: {args.discoveries} ({len(discs)} discoveries, '
                 f'{len(by_rule)} rules with hits)')
    lines.append(f'Using top-{args.top} discoveries by score per rule '
                 f'(min {args.min_discoveries} discoveries to consider).')
    lines.append('')
    lines.append('Each table shows: current default → median of top-K → '
                 'distance in units of param_range (|d| > 0.25 = far off).')
    lines.append('')

    summary_far = []   # rules where the default is far from what works
    summary_skip = []  # rules with too few discoveries

    for rule in sorted(RULE_PRESETS.keys()):
        preset = RULE_PRESETS[rule]
        hits = by_rule.get(rule, [])
        if len(hits) < args.min_discoveries:
            summary_skip.append((rule, len(hits)))
            continue

        # Top-K by score
        hits_sorted = sorted(hits, key=lambda x: x.get('score', 0.0),
                             reverse=True)[:args.top]

        cur_params: dict = preset.get('params', {})
        ranges: dict = preset.get('param_ranges', {})
        cur_dt = preset.get('dt')
        dt_range = preset.get('dt_range')

        # dt summary
        dt_vals = [h.get('dt') for h in hits_sorted
                   if h.get('dt') is not None]
        med_dt = statistics.median(dt_vals) if dt_vals else None

        # Per-param summary
        rows: list[str] = []
        max_dist = 0.0
        for pname, cur_v in cur_params.items():
            vals = [h['params'].get(pname) for h in hits_sorted
                    if isinstance(h.get('params'), dict)
                    and h['params'].get(pname) is not None]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if not vals:
                continue
            med = statistics.median(vals)
            mn  = min(vals); mx = max(vals)
            lo, hi = ranges.get(pname, (None, None))
            dist_str = ''
            if lo is not None and hi is not None and hi > lo:
                dist = abs(med - cur_v) / (hi - lo)
                max_dist = max(max_dist, dist)
                if dist > 0.25:    dist_str = f'  **|d|={dist:.2f}**'
                else:              dist_str = f'  |d|={dist:.2f}'
            rounded = round_smart(med, lo or med, hi or med) if lo is not None else med
            rows.append(f'  - `{pname}`: cur={cur_v}  →  med={rounded}  '
                        f'[range {mn:.3g}–{mx:.3g}]{dist_str}')

        # dt distance
        dt_dist_str = ''
        if med_dt is not None and cur_dt is not None and dt_range:
            dlo, dhi = dt_range
            if dhi > dlo:
                ddist = abs(med_dt - cur_dt) / (dhi - dlo)
                max_dist = max(max_dist, ddist)
                if ddist > 0.25:  dt_dist_str = f'  **|d|={ddist:.2f}**'
                else:             dt_dist_str = f'  |d|={ddist:.2f}'

        flag = ' ⚠️ FAR' if max_dist > 0.25 else ''
        lines.append(f'## {rule}{flag}')
        lines.append(f'- discoveries: {len(hits)} (using top {len(hits_sorted)})')
        if med_dt is not None and cur_dt is not None:
            lines.append(f'- dt: cur={cur_dt}  →  med={round_smart(med_dt, dt_range[0] if dt_range else 0, dt_range[1] if dt_range else 1)}{dt_dist_str}')
        lines.extend(rows)
        lines.append('')

        if max_dist > 0.25:
            summary_far.append((rule, len(hits), max_dist))

    # Top of report
    head = ['# Preset tuning audit', '']
    head.append(f'**{len(summary_far)} rules** have defaults > 0.25 of '
                f'param-range away from what the search finds:')
    head.append('')
    for r, n, d in sorted(summary_far, key=lambda x: -x[2]):
        head.append(f'  - `{r}`  (n={n}, max_dist={d:.2f})')
    head.append('')
    head.append(f'Skipped (< {args.min_discoveries} discoveries): '
                f'{len(summary_skip)}')
    for r, n in summary_skip:
        head.append(f'  - `{r}`  (n={n})')
    head.append('')
    head.append('---')
    head.append('')

    full = '\n'.join(head + lines[2:])
    with open(args.out, 'w') as f:
        f.write(full)
    print(f'[tuning] wrote {args.out}')
    print(f'[tuning] {len(summary_far)} rules far from optimal, '
          f'{len(summary_skip)} skipped')
    for r, n, d in sorted(summary_far, key=lambda x: -x[2])[:20]:
        print(f'  {r:<32} n={n:>4}  max_dist={d:.2f}')


if __name__ == '__main__':
    main()
