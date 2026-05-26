"""Golden-snapshot regression probe.

Probe #15 — every existing probe asks "is the engine internally
self-consistent today?" (no NaN, conservation holds, params do
something, channels carry signal, renderer shows pixels).  None ask:
"does the engine produce the SAME output today that I approved
yesterday?"

That second question is the one that matters when you can't visually
judge whether a Lenia organism, a Gray–Scott pattern, or a
causal-wave shape is *physically correct* — you can only judge
whether it matches a state you previously inspected and approved.

This probe converts that act of visual approval into a permanent
regression guard:

  1.  You open a rule in the GUI, watch it, decide "this looks right".
  2.  You run ``python -m ca_debug.golden_snapshots --bless <rule>``.
      The probe runs the rule headlessly at fixed (size, seed, dt,
      params), samples grid statistics + a bit-exact byte hash at
      checkpoint steps, and writes ``ca_debug/golden/<rule>.json``.
  3.  From then on, every nightly probe run calls
      ``python -m ca_debug.golden_snapshots`` (no flags) which re-runs
      every blessed rule and grades the divergence.

Grading is two-tier:

  hash-identical → ok  (bit-exact reproduction, gold standard)
  hash differs but stats within tol → high
                       (probable GPU FP nondeterminism — re-bless
                        suggested but not a regression)
  stats diverge above tol → crit
                       (engine output has materially changed; either
                        a real regression or an intentional change
                        that needs re-blessing after inspection)

Stats captured per checkpoint, per channel:
  mean, std, min, max, alive_count (>1e-6)

Relative-difference tolerance defaults to 1%% (--rtol 0.01).  For each
stat: |new - blessed| / max(1e-6, |blessed|) > rtol → diverged.

Storage: ``ca_debug/golden/<rule>.json``.  The directory is committed
to the repo so blessings persist across machines.  Each file records
``blessed_at`` timestamp and the user-provided ``--note`` if any.

Usage::

    # one-time, after visually validating a rule
    python -m ca_debug.golden_snapshots --bless lenia_3d --note "classic blob"

    # bulk bless every rule lacking a snapshot (for initial bootstrap)
    python -m ca_debug.golden_snapshots --bless-missing

    # routine regression check (no flags)
    python -m ca_debug.golden_snapshots
    python -m ca_debug.golden_snapshots --severity ok --rtol 0.005
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'ok': 4,
              'skip': 5, 'n/a': 6}

# Default headless run config.  Held fixed across blessing & checking;
# never change without re-blessing the whole catalogue.
_DEFAULT_SIZE = 48
_DEFAULT_SEED = 1001
_DEFAULT_CHECKPOINTS = (25, 100, 250)
# Active-cell threshold for alive_count stat.
_ALIVE_EPS = 1e-6

# Where snapshot files live.
_GOLDEN_DIR = Path(__file__).parent / 'golden'


def _checkpoint_steps(args_checkpoints: str | None) -> list[int]:
    if not args_checkpoints:
        return list(_DEFAULT_CHECKPOINTS)
    return sorted({int(s) for s in args_checkpoints.split(',') if s.strip()})


def _grid_stats(grid: np.ndarray) -> dict:
    """Per-channel summary stats (compact, robust across runs)."""
    n_ch = grid.shape[-1]
    stats = []
    for c in range(n_ch):
        ch = grid[..., c]
        finite = ch[np.isfinite(ch)]
        if finite.size == 0:
            stats.append({'mean': None, 'std': None, 'min': None,
                          'max': None, 'alive': 0, 'nonfinite': int(ch.size)})
            continue
        stats.append({
            'mean': float(finite.mean()),
            'std': float(finite.std()),
            'min': float(finite.min()),
            'max': float(finite.max()),
            'alive': int((np.abs(finite) > _ALIVE_EPS).sum()),
            'nonfinite': int(ch.size - finite.size),
        })
    return {'shape': list(grid.shape), 'channels': stats}


def _grid_hash(grid: np.ndarray) -> str:
    """Bit-exact SHA256 of the grid bytes (float32 little-endian)."""
    arr = np.ascontiguousarray(grid, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _capture_snapshots(ctx, rule: str, size: int, seed: int,
                       checkpoints: list[int]) -> dict:
    """Run rule and capture (hash, stats) at each checkpoint step.

    Uses preset defaults for params, dt, init.  Returns either a
    'snapshots' dict (success) or 'error' dict.
    """
    from simulator import _resolve_composed_preset
    from test_harness import HeadlessRunner
    try:
        preset = _resolve_composed_preset(rule)
    except Exception as e:  # noqa: BLE001
        return {'error': f'resolve: {type(e).__name__}: {e}'}

    if preset.get('kind') == 'viewport':
        return {'skip': 'viewport kind — pure render preset, no temporal state'}

    # SSBO / agent-arena rules have inherent ordering nondeterminism
    # (atomic counter increments, particle dispatch order) — their grid
    # output is not bit-reproducible across runs.  Same exclusion as
    # Probe #10 (param_coherence).
    if 'entity_arena' in preset:
        return {'skip': "entity_arena rule — agent-step ordering "
                        "nondeterminism prevents reliable hashing"}
    if int(preset.get('particle_count') or 0) > 0:
        return {'skip': 'particle SSBO rule — dispatch-order '
                        'nondeterminism prevents reliable hashing'}
    if int(preset.get('agent_count') or 0) > 0:
        return {'skip': 'agent-pass rule — agent dispatch-order '
                        'nondeterminism prevents reliable hashing'}

    # Honour preset's declared minimum world size.
    declared = int(preset.get('default_size') or 0)
    if declared > size:
        size = declared

    params = dict(preset.get('params') or {})
    dt = preset.get('dt')
    init = preset.get('init')

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = HeadlessRunner(ctx, rule, size=size, seed=seed)
    except Exception as e:  # noqa: BLE001
        return {'error': f'construct: {type(e).__name__}: {e}',
                'tb': traceback.format_exc().splitlines()[-3:]}

    snapshots = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            stepped = 0
            for target in sorted(checkpoints):
                if target < stepped:
                    continue
                for _ in range(target - stepped):
                    r.step()
                stepped = target
                grid = np.asarray(r.read_grid())
                snapshots.append({
                    'step': stepped,
                    'hash': _grid_hash(grid),
                    'stats': _grid_stats(grid),
                })
    except Exception as e:  # noqa: BLE001
        return {'error': f'step: {type(e).__name__}: {e}',
                'tb': traceback.format_exc().splitlines()[-3:],
                'partial_snapshots': snapshots}
    finally:
        try: r.release()
        except Exception: pass  # noqa: BLE001

    return {'snapshots': snapshots, 'size': size, 'seed': seed,
            'dt': dt, 'init': init, 'params': params,
            'checkpoints': checkpoints}


def _bless_rule(ctx, rule: str, size: int, seed: int,
                checkpoints: list[int], note: str | None) -> dict:
    cap = _capture_snapshots(ctx, rule, size, seed, checkpoints)
    if 'error' in cap:
        return {'rule': rule, 'grade': 'err', 'reason': cap['error']}
    if 'skip' in cap:
        return {'rule': rule, 'grade': 'skip', 'reason': cap['skip']}

    _GOLDEN_DIR.mkdir(exist_ok=True)
    path = _GOLDEN_DIR / f'{rule}.json'
    payload = {
        'rule': rule,
        'size': cap['size'],
        'seed': cap['seed'],
        'dt': cap['dt'],
        'init': cap['init'],
        'params': cap['params'],
        'checkpoints': cap['checkpoints'],
        'snapshots': cap['snapshots'],
        'blessed_at': datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds'),
        'note': note,
    }
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2, default=str)
    return {'rule': rule, 'grade': 'ok',
            'reason': f'blessed → {path.relative_to(_GOLDEN_DIR.parent.parent)}'}


def _stat_diverges(blessed: dict, current: dict, rtol: float) -> list[str]:
    """Return list of per-channel divergence descriptions exceeding rtol."""
    diffs: list[str] = []
    bch = blessed.get('channels') or []
    cch = current.get('channels') or []
    n = min(len(bch), len(cch))
    if len(bch) != len(cch):
        diffs.append(f'channel-count {len(cch)} vs blessed {len(bch)}')
    for i in range(n):
        b, c = bch[i], cch[i]
        for key in ('mean', 'std', 'min', 'max'):
            bv, cv = b.get(key), c.get(key)
            if bv is None or cv is None:
                if bv != cv:
                    diffs.append(f'ch{i}.{key}: blessed={bv} now={cv}')
                continue
            denom = max(1e-6, abs(bv))
            rel = abs(cv - bv) / denom
            if rel > rtol:
                diffs.append(
                    f'ch{i}.{key}: {bv:.4g} → {cv:.4g} (rel {rel:.2%})')
        # alive_count is integer; tolerate ±1% or ±5 cells whichever is larger.
        bv, cv = b.get('alive', 0), c.get('alive', 0)
        denom = max(5, int(0.01 * (bv or 1)))
        if abs(cv - bv) > denom:
            diffs.append(f'ch{i}.alive: {bv} → {cv} (Δ={cv - bv})')
    return diffs


def _check_rule(ctx, rule: str, rtol: float) -> dict:
    # Apply the same skip logic bless uses — these rules cannot be
    # reliably hashed (viewport: no temporal state; SSBO: ordering
    # nondeterminism).  Reports as 'skip' rather than 'n/a' for clarity.
    try:
        from simulator import _resolve_composed_preset
        preset = _resolve_composed_preset(rule)
    except Exception:  # noqa: BLE001
        preset = {}
    if preset.get('kind') == 'viewport':
        return {'rule': rule, 'grade': 'skip',
                'reason': 'viewport kind — no temporal state'}
    if 'entity_arena' in preset:
        return {'rule': rule, 'grade': 'skip',
                'reason': 'entity_arena rule — nondeterministic'}
    if int(preset.get('particle_count') or 0) > 0:
        return {'rule': rule, 'grade': 'skip',
                'reason': 'particle SSBO rule — nondeterministic'}
    if int(preset.get('agent_count') or 0) > 0:
        return {'rule': rule, 'grade': 'skip',
                'reason': 'agent-pass rule — nondeterministic'}

    path = _GOLDEN_DIR / f'{rule}.json'
    if not path.exists():
        return {'rule': rule, 'grade': 'n/a',
                'reason': 'no golden snapshot — bless with --bless'}
    try:
        with open(path) as fh:
            blessed = json.load(fh)
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'grade': 'err',
                'reason': f'load blessed: {type(e).__name__}: {e}'}

    cap = _capture_snapshots(ctx, rule,
                             int(blessed.get('size') or _DEFAULT_SIZE),
                             int(blessed.get('seed') or _DEFAULT_SEED),
                             list(blessed.get('checkpoints') or _DEFAULT_CHECKPOINTS))
    if 'error' in cap:
        return {'rule': rule, 'grade': 'err', 'reason': cap['error']}
    if 'skip' in cap:
        return {'rule': rule, 'grade': 'skip', 'reason': cap['skip']}

    blessed_snaps = {s['step']: s for s in blessed.get('snapshots') or []}
    current_snaps = {s['step']: s for s in cap['snapshots']}
    per_checkpoint = []
    worst = 'ok'
    worst_rank = _SEV_ORDER['ok']
    for step in sorted(blessed_snaps.keys() | current_snaps.keys()):
        b = blessed_snaps.get(step)
        c = current_snaps.get(step)
        if b is None or c is None:
            per_checkpoint.append({'step': step, 'grade': 'crit',
                                   'reason': 'checkpoint missing'})
            worst, worst_rank = 'crit', _SEV_ORDER['crit']
            continue
        if b['hash'] == c['hash']:
            per_checkpoint.append({'step': step, 'grade': 'ok',
                                   'reason': 'hash identical'})
            continue
        diffs = _stat_diverges(b['stats'], c['stats'], rtol)
        if not diffs:
            per_checkpoint.append({'step': step, 'grade': 'high',
                                   'reason': 'hash differs; stats within tol'})
            if _SEV_ORDER['high'] < worst_rank:
                worst, worst_rank = 'high', _SEV_ORDER['high']
        else:
            per_checkpoint.append({'step': step, 'grade': 'crit',
                                   'reason': 'stats diverged',
                                   'diffs': diffs})
            worst, worst_rank = 'crit', _SEV_ORDER['crit']

    return {'rule': rule, 'grade': worst,
            'blessed_at': blessed.get('blessed_at'),
            'note': blessed.get('note'),
            'checkpoints': per_checkpoint}


def _select_rules(args) -> list[str]:
    from simulator import RULE_PRESETS
    if args.rules:
        return [r.strip() for r in args.rules.split(',') if r.strip()]
    rules = sorted(RULE_PRESETS.keys())
    if args.skip_flagship:
        rules = [r for r in rules if not r.startswith('flagship_')]
    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(',') if s.strip()}
        rules = [r for r in rules if r not in skip_set]
    return rules


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names.')
    ap.add_argument('--bless', action='store_true',
                    help='Bless the listed rules (write new golden snapshots).')
    ap.add_argument('--bless-missing', action='store_true',
                    help='Bless every rule that lacks a golden snapshot.')
    ap.add_argument('--note', help='Bless note (recorded in snapshot file).')
    ap.add_argument('--size', type=int, default=_DEFAULT_SIZE,
                    help=f'Grid size for new blessings (default: {_DEFAULT_SIZE}). '
                         f'Ignored for --check (uses blessed size).')
    ap.add_argument('--seed', type=int, default=_DEFAULT_SEED)
    ap.add_argument('--checkpoints', help='Comma-separated step indices '
                    f'(default: {",".join(str(s) for s in _DEFAULT_CHECKPOINTS)}). '
                    f'Only used at bless time.')
    ap.add_argument('--rtol', type=float, default=0.01,
                    help='Relative tolerance for stat divergence (default: 0.01).')
    ap.add_argument('--skip-flagship', action='store_true')
    ap.add_argument('--skip', help='Comma-separated rules to omit.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='high',
                    help='Min severity to print (default: high).')
    ap.add_argument('--json', help='Write per-rule report JSON.')
    args = ap.parse_args(argv)

    from test_harness import create_headless_context
    _window, ctx = create_headless_context()

    checkpoints = _checkpoint_steps(args.checkpoints)
    rules = _select_rules(args)

    if args.bless_missing:
        if not _GOLDEN_DIR.exists():
            existing = set()
        else:
            existing = {p.stem for p in _GOLDEN_DIR.glob('*.json')}
        rules = [r for r in rules if r not in existing]
        if not rules:
            print('all rules already blessed.')
            return 0
        print(f'blessing {len(rules)} unblessed rule(s)...')

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, rule in enumerate(rules, 1):
        sys.stdout.write(f"\r[{i:>3}/{len(rules)}] {rule:<42}")
        sys.stdout.flush()
        if args.bless or args.bless_missing:
            rows.append(_bless_rule(ctx, rule, args.size, args.seed,
                                    checkpoints, args.note))
        else:
            rows.append(_check_rule(ctx, rule, args.rtol))
    sys.stdout.write('\r' + ' ' * 70 + '\r')
    elapsed = time.perf_counter() - t0

    counts: dict = {k: 0 for k in _SEV_ORDER}
    for row in rows:
        counts[row['grade']] = counts.get(row['grade'], 0) + 1

    mode = ('bless' if (args.bless or args.bless_missing) else 'check')
    print(f'\ngolden snapshots [{mode}] — {len(rules)} rules in {elapsed:.1f}s')
    for g in ('err', 'crit', 'high', 'med', 'ok', 'skip', 'n/a'):
        if counts.get(g):
            print(f'    {g:<5}  {counts[g]:>5}')

    sev_cap = _SEV_ORDER[args.severity]
    flagged = [r for r in rows
               if _SEV_ORDER.get(r['grade'], 9) <= sev_cap]
    if flagged:
        print(f'\nflagged ({args.severity}+):  {len(flagged)} rules')
        for row in flagged:
            print(f'  [{row["grade"]:<4}] {row["rule"]}')
            if row.get('reason'):
                print(f'          {row["reason"]}')
            for cp in row.get('checkpoints') or []:
                if _SEV_ORDER.get(cp['grade'], 9) > sev_cap:
                    continue
                print(f'          step {cp["step"]:>5}: [{cp["grade"]:<4}] '
                      f'{cp.get("reason", "")}')
                for d in (cp.get('diffs') or [])[:6]:
                    print(f'              {d}')
                if len(cp.get('diffs') or []) > 6:
                    print(f'              ... and '
                          f'{len(cp["diffs"]) - 6} more')

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'mode': mode, 'counts': counts, 'rows': rows,
                       'elapsed_s': elapsed, 'rtol': args.rtol},
                      fh, indent=2, default=str)
        print(f'\nwrote {args.json}')

    return 1 if (counts.get('err', 0) + counts.get('crit', 0)) else 0


if __name__ == '__main__':
    sys.exit(main())
