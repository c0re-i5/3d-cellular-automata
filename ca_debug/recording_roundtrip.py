"""Recording round-trip fidelity probe.

For every recording JSON under ``recordings/`` verify that the
engine can still reproduce its run from the recorded metadata:

  err   the recording's rule no longer exists, the runner
        construction fails, or stepping crashes.
  crit  the run produces NaN/Inf, or two replays from the same
        metadata diverge (non-deterministic).
  high  the recorded ``params`` dict has keys that do not match
        the current preset (renamed/removed/added parameter).
        Replay still attempts with the intersection; if it fails
        afterwards the higher-severity grade wins.
  med   the recording uses a size that's too large for safe
        replay (>=512 has triggered Xid in earlier probes) so
        the run is skipped — recorded for visibility, not a bug.
  ok    construction + cap-step replay clean and deterministic.

Run-cost is capped: we only step ``--cap`` frames per recording
(default 30) — full multi-thousand-frame replay at 384^3 is
ten-minute scale for the whole catalogue. The cap exercises the
construction + early-evolution path, which is where rule rename /
param-schema drift would surface.

Usage::

    python -m ca_debug.recording_roundtrip
    python -m ca_debug.recording_roundtrip --cap 60 --severity high
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import sys
import time
import traceback

import numpy as np


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'ok': 4}

# Sizes we refuse to replay because earlier probes hit driver-level
# faults (Xid) at this resolution on this hardware.
_OVERSIZE_THRESHOLD = 512


def _find_recordings(root: str) -> list[str]:
    paths = glob.glob(os.path.join(root, '**', '*.json'), recursive=True)
    return sorted(p for p in paths if 'upload_log' not in os.path.basename(p))


def _load_meta(path: str) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def _state_hash(runner) -> tuple[bytes, float, float]:
    g = np.asarray(runner.read_grid())
    g = np.ascontiguousarray(g)
    finite = np.isfinite(g)
    if not finite.all():
        return (b'NONFINITE', float('nan'), float('nan'))
    return (g.tobytes()[-4096:], float(g.mean()), float(np.abs(g).max()))


def _replay_once(ctx, rule, size, seed, params, dt, cap):
    from test_harness import HeadlessRunner
    with contextlib.redirect_stdout(io.StringIO()):
        r = HeadlessRunner(ctx, rule, size=size, seed=seed,
                           params=params, dt=dt)
        for _ in range(cap):
            r.step()
        h = _state_hash(r)
        try: r.release()
        except Exception: pass  # noqa: BLE001
    return h


def _probe_recording(ctx, path: str, cap: int) -> dict:
    meta = _load_meta(path)
    name = os.path.basename(path)
    if meta is None:
        return {'path': name, 'grade': 'err',
                'error': 'unreadable JSON'}

    rule = meta.get('rule')
    size = int(meta.get('size') or 0)
    seed = int(meta.get('seed') or 0)
    params = meta.get('params') or {}
    dt = meta.get('dt')

    out = {'path': name, 'rule': rule, 'size': size, 'seed': seed}

    if not rule:
        return {**out, 'grade': 'err', 'error': 'no rule in metadata'}

    try:
        from simulator import RULE_PRESETS, _resolve_composed_preset
    except Exception as e:  # noqa: BLE001
        return {**out, 'grade': 'err',
                'error': f'simulator import: {type(e).__name__}: {e}'}

    if rule not in RULE_PRESETS:
        return {**out, 'grade': 'err',
                'error': f'rule {rule!r} no longer exists'}

    try:
        preset = _resolve_composed_preset(rule)
    except Exception as e:  # noqa: BLE001
        return {**out, 'grade': 'err',
                'error': f'resolve {rule!r}: {type(e).__name__}: {e}'}

    expected = set((preset.get('params') or {}).keys())
    recorded = set(params.keys())
    missing = recorded - expected      # recorded param no longer in preset
    extra = expected - recorded        # preset added a new param since record

    schema_drift = None
    if missing or extra:
        schema_drift = {'missing_from_preset': sorted(missing),
                        'added_to_preset': sorted(extra)}
        out['schema_drift'] = schema_drift

    if size >= _OVERSIZE_THRESHOLD:
        return {**out, 'grade': 'med',
                'reason': f'size={size} >= {_OVERSIZE_THRESHOLD} (Xid risk on this GPU), skipped'}

    # Restrict params to the intersection so the runner doesn't
    # choke on unknown keys.
    safe_params = {k: v for k, v in params.items() if k in expected}

    try:
        h1 = _replay_once(ctx, rule, size, seed, safe_params, dt, cap)
    except Exception as e:  # noqa: BLE001
        return {**out, 'grade': 'err',
                'error': f'replay 1 {type(e).__name__}: {e}',
                'tb': traceback.format_exc().splitlines()[-3:]}

    if h1[0] == b'NONFINITE':
        return {**out, 'grade': 'crit',
                'reason': f'NaN/Inf appeared within {cap} steps'}

    try:
        h2 = _replay_once(ctx, rule, size, seed, safe_params, dt, cap)
    except Exception as e:  # noqa: BLE001
        return {**out, 'grade': 'err',
                'error': f'replay 2 {type(e).__name__}: {e}'}

    if h1 != h2:
        return {**out, 'grade': 'crit',
                'reason': (
                    f'non-deterministic: hash1={h1[1]:.6g}/{h1[2]:.6g} '
                    f'!= hash2={h2[1]:.6g}/{h2[2]:.6g}')}

    if schema_drift:
        out['grade'] = 'high'
        bits = []
        if missing: bits.append(f'recorded keys not in preset: {sorted(missing)}')
        if extra:   bits.append(f'preset added since record: {sorted(extra)}')
        out['reason'] = '; '.join(bits)
        return out

    out['grade'] = 'ok'
    return out


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='recordings',
                    help='Recording directory to scan recursively.')
    ap.add_argument('--cap', type=int, default=30,
                    help='Max steps per replay (default 30, two replays per recording).')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='med',
                    help='Min severity to print (default: med).')
    ap.add_argument('--json', help='Write per-recording report JSON.')
    ap.add_argument('--limit', type=int, help='Process at most N recordings.')
    args = ap.parse_args(argv)

    from test_harness import create_headless_context
    window, ctx = create_headless_context()

    paths = _find_recordings(args.root)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"no recordings under {args.root!r}")
        return 0

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, path in enumerate(paths, 1):
        sys.stdout.write(f"\r[{i:>3}/{len(paths)}] {os.path.basename(path):<60.60}")
        sys.stdout.flush()
        rows.append(_probe_recording(ctx, path, args.cap))
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    elapsed = time.perf_counter() - t0

    counts: dict = {k: 0 for k in _SEV_ORDER}
    for r in rows:
        counts[r['grade']] = counts.get(r['grade'], 0) + 1

    rows_sorted = sorted(rows, key=lambda r: _SEV_ORDER.get(r['grade'], 9))
    min_sev = _SEV_ORDER[args.severity]
    print(f"recording round-trip probe (cap={args.cap}) -- {elapsed:.1f}s "
          f"({len(paths)} recordings)")
    print(f"{'SEV':<6} {'RULE':<30} {'SIZE':>5}  {'PATH':<60} NOTES")
    print('-' * 140)
    for r in rows_sorted:
        if _SEV_ORDER.get(r['grade'], 9) > min_sev:
            continue
        note = r.get('reason') or r.get('error') or ''
        print(f"{r['grade']:<6} {(r.get('rule') or '?'):<30.30} "
              f"{r.get('size') or 0:>5}  {r['path']:<60.60} {note}")

    summary = '  '.join(f'{k}={v}' for k, v in counts.items() if v)
    print(f"\nSummary: {summary}  (n={len(paths)})")

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'rows': rows, 'counts': counts,
                       'cap': args.cap, 'elapsed_s': elapsed}, fh, indent=2)
        print(f"Wrote {args.json}")

    return 1 if (counts['err'] + counts['crit'] + counts['high']) else 0


if __name__ == '__main__':
    sys.exit(main())
