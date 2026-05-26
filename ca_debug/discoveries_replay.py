"""Discoveries replay fidelity probe.

For a sampled subset of entries in ``discoveries.json`` verify that the
engine can still reproduce each saved discovery's run from its recorded
``(rule, init_variant, size, seed, dt, params)`` metadata:

  err   the rule no longer exists, the runner construction fails,
        or stepping crashes.
  crit  the run produces NaN/Inf within ``--cap`` steps.
  high  the recorded ``params`` dict has keys the current preset
        no longer declares, or the preset has added params that the
        recording doesn't supply (replay still attempts with the
        intersection + new-param defaults).
  med   the recording uses a size we refuse to replay
        (>= 512 — Xid risk on this GPU per Probe #1).
  ok    construction + cap-step replay clean.

Sampling strategy (the catalogue is ~28k entries; we want broad
coverage of the (rule, init_variant) cross-product plus full coverage
of user-marked discoveries which carry the highest historical value):

  * EVERY entry with ``marked == True`` is replayed (curated favourites,
    most regression-sensitive).
  * For unmarked entries, up to ``--per-group`` entries per
    (rule, init_variant) pair are sampled at fixed RNG (seed=1001
    by default) for reproducibility.

This catches the same drift modes as Probe #8 (rename / param schema /
NaN regressions) but across the much broader discovery surface that
recordings don't reach.  Outcome comparison (alive_frac matching the
recorded ``final_alive``) is INTENTIONALLY NOT performed — engine
evolution legitimately changes outcomes and that's not a bug; this
probe just verifies "still runs cleanly".

Usage::

    python -m ca_debug.discoveries_replay
    python -m ca_debug.discoveries_replay --per-group 2 --cap 30
    python -m ca_debug.discoveries_replay --rules game_of_life_3d,lichen
    python -m ca_debug.discoveries_replay --marked-only
    python -m ca_debug.discoveries_replay --json /tmp/drep.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
import traceback

import numpy as np


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'ok': 4}

# Sizes we refuse to replay — earlier probes hit driver-level Xid faults.
_OVERSIZE_THRESHOLD = 512


def _replay(ctx, rule, size, seed, params, dt, init_variant, cap):
    from test_harness import HeadlessRunner
    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt,
                           init_override=init_variant)
        try:
            for _ in range(cap):
                r.step()
            g = np.asarray(r.read_grid())
            finite = bool(np.isfinite(g).all())
            return finite, None
        finally:
            try: r.release()
            except Exception: pass  # noqa: BLE001


def _probe_entry(ctx, entry: dict, cap: int,
                 preset_cache: dict) -> dict:
    from simulator import RULE_PRESETS, _resolve_composed_preset
    rule = entry.get('rule')
    out = {
        'rule': rule,
        'size': entry.get('size'),
        'seed': entry.get('seed'),
        'init_variant': entry.get('init_variant'),
        'marked': bool(entry.get('marked')),
    }
    if not rule:
        return {**out, 'grade': 'err', 'reason': 'entry has no rule field'}
    if rule not in RULE_PRESETS:
        return {**out, 'grade': 'err', 'reason': f'rule {rule!r} no longer exists'}

    if rule not in preset_cache:
        try:
            preset_cache[rule] = _resolve_composed_preset(rule)
        except Exception as e:  # noqa: BLE001
            return {**out, 'grade': 'err',
                    'reason': f'resolve {rule!r}: {type(e).__name__}: {e}'}
    preset = preset_cache[rule]
    expected = set((preset.get('params') or {}).keys())
    recorded_params = entry.get('params') or {}
    recorded = set(recorded_params.keys())
    missing = recorded - expected
    added = expected - recorded
    schema_drift = bool(missing or added)
    if schema_drift:
        out['schema_drift'] = {
            'recorded_not_in_preset': sorted(missing),
            'preset_new': sorted(added),
        }

    size = int(entry.get('size') or 0)
    if size <= 0:
        return {**out, 'grade': 'err', 'reason': 'invalid size'}
    if size >= _OVERSIZE_THRESHOLD:
        return {**out, 'grade': 'med',
                'reason': f'size={size} >= {_OVERSIZE_THRESHOLD} (Xid risk), skipped'}

    safe_params = {k: v for k, v in recorded_params.items() if k in expected}
    seed = int(entry.get('seed') or 0)
    dt = entry.get('dt')
    init_variant = entry.get('init_variant')

    try:
        finite, _ = _replay(ctx, rule, size, seed, safe_params, dt,
                            init_variant, cap)
    except Exception as e:  # noqa: BLE001
        return {**out, 'grade': 'err',
                'reason': f'{type(e).__name__}: {e}',
                'tb': traceback.format_exc().splitlines()[-3:]}

    if not finite:
        return {**out, 'grade': 'crit',
                'reason': f'NaN/Inf within {cap} steps'}

    if schema_drift:
        bits = []
        if missing: bits.append(f'recorded keys not in preset: {sorted(missing)}')
        if added:   bits.append(f'preset added since record: {sorted(added)}')
        return {**out, 'grade': 'high', 'reason': '; '.join(bits)}

    return {**out, 'grade': 'ok'}


def _sample_entries(path: str, per_group: int, rules_filter: set | None,
                    marked_only: bool, sample_seed: int) -> list[dict]:
    with open(path) as fh:
        data = json.load(fh)
    if rules_filter:
        data = [e for e in data if e.get('rule') in rules_filter]
    marked = [e for e in data if e.get('marked')]
    if marked_only:
        return marked
    groups: dict[tuple, list[dict]] = {}
    for e in data:
        if e.get('marked'):
            continue
        key = (e.get('rule'), e.get('init_variant'))
        groups.setdefault(key, []).append(e)
    rng = random.Random(sample_seed)
    unmarked_sample: list[dict] = []
    for key, items in groups.items():
        if len(items) <= per_group:
            unmarked_sample.extend(items)
        else:
            unmarked_sample.extend(rng.sample(items, per_group))
    # Deduplicate (marked entries take precedence; identity by id())
    seen = set()
    out: list[dict] = []
    for e in marked + unmarked_sample:
        eid = id(e)
        if eid in seen:
            continue
        seen.add(eid)
        out.append(e)
    return out


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='discoveries.json',
                    help='Discoveries JSON file (default: discoveries.json).')
    ap.add_argument('--per-group', type=int, default=1,
                    help='Max unmarked samples per (rule, init_variant) (default: 1).')
    ap.add_argument('--cap', type=int, default=20,
                    help='Max steps per replay (default: 20).')
    ap.add_argument('--sample-seed', type=int, default=1001,
                    help='RNG seed for the per-group sampling (default: 1001).')
    ap.add_argument('--rules', help='Comma-separated rule names to restrict to.')
    ap.add_argument('--marked-only', action='store_true',
                    help='Replay only marked entries.')
    ap.add_argument('--limit', type=int, help='Process at most N entries.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='med',
                    help='Min severity to print (default: med).')
    ap.add_argument('--json', help='Write per-entry report JSON.')
    args = ap.parse_args(argv)

    rules_filter = None
    if args.rules:
        rules_filter = {r.strip() for r in args.rules.split(',') if r.strip()}

    entries = _sample_entries(args.path, args.per_group, rules_filter,
                              args.marked_only, args.sample_seed)
    if args.limit:
        entries = entries[:args.limit]
    if not entries:
        print('no entries to replay')
        return 0

    from test_harness import create_headless_context
    _window, ctx = create_headless_context()

    preset_cache: dict = {}
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, entry in enumerate(entries, 1):
        sys.stdout.write(
            f"\r[{i:>4}/{len(entries)}] {entry.get('rule', '?'):<30}")
        sys.stdout.flush()
        rows.append(_probe_entry(ctx, entry, args.cap, preset_cache))
    sys.stdout.write('\r' + ' ' * 60 + '\r')
    elapsed = time.perf_counter() - t0

    counts: dict = {k: 0 for k in _SEV_ORDER}
    for r in rows:
        counts[r['grade']] = counts.get(r['grade'], 0) + 1

    # Aggregate err/crit/high by rule for compact output.
    by_rule: dict[str, dict] = {}
    for r in rows:
        rname = r.get('rule') or '?'
        d = by_rule.setdefault(rname, {k: 0 for k in _SEV_ORDER})
        d[r['grade']] += 1

    print(f"\ndiscoveries-replay probe -- {len(entries)} entries "
          f"({sum(1 for e in entries if e.get('marked'))} marked) "
          f"in {elapsed:.1f}s")
    print('  by entry:')
    for g in ('err', 'crit', 'high', 'med', 'ok'):
        print(f'    {g:<5}  {counts.get(g, 0):>5}')

    sev_cap = _SEV_ORDER[args.severity]
    flagged = [r for r in rows if _SEV_ORDER.get(r['grade'], 9) <= sev_cap]
    if flagged:
        print(f'\nflagged ({args.severity}+):  {len(flagged)} entries')
        # Group identical (rule, grade, reason) signatures to avoid spam.
        sig_groups: dict[tuple, list[dict]] = {}
        for r in flagged:
            sig = (r.get('rule'), r['grade'], r.get('reason'))
            sig_groups.setdefault(sig, []).append(r)
        for (rule, grade, reason), items in sorted(
                sig_groups.items(),
                key=lambda kv: _SEV_ORDER.get(kv[0][1], 9)):
            mk = sum(1 for it in items if it.get('marked'))
            print(f'  [{grade:<4}] {rule:<30}  '
                  f'n={len(items):>3} (marked={mk})  {reason}')

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'counts': counts, 'by_rule': by_rule,
                       'rows': rows, 'cap': args.cap,
                       'elapsed_s': elapsed}, fh, indent=2, default=str)
        print(f'\nwrote {args.json}')

    return 1 if (counts['err'] + counts['crit'] + counts['high']) else 0


if __name__ == '__main__':
    sys.exit(main())
