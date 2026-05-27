"""Probe #17 — shared-memory vs direct-fetch equivalence.

Many CA shaders in ``simulator.py`` contain a ``#if USE_SHARED_MEM`` /
``#else`` / ``#endif`` branch.  The two paths are intended to be
numerically equivalent (the docstrings claim *bit-identical* on cubic
grids).  In production the macro is undefined, so the ``#else`` /
"direct-fetch" path is the one that actually runs.  The shared-memory
loaders (cooperative tile fills with ``barrier()``) are non-trivial and
historically host-untested.

This probe compiles every affected shader twice — once with
``USE_SHARED_MEM=0`` and once with ``USE_SHARED_MEM=1`` — runs both
through the same ``HeadlessRunner`` for ``N`` steps from the same IC,
and diffs the final grid.

Grading (max_diff over all channels + pairs):

    ok     max_diff == 0           (bit-identical, as documented)
    low    max_diff <= 1e-6        (last-mantissa drift; reorder-only)
    med    max_diff <= 1e-4        (small but visible)
    high   max_diff <= 1e-2        (significant)
    crit   max_diff >  1e-2        (visibly wrong / divergent)
    skip   rule does not reference USE_SHARED_MEM
    err    compile / run crash for at least one variant

Usage::

    python -m ca_debug.shared_mem_equivalence
    python -m ca_debug.shared_mem_equivalence --rules lenia_3d,bz_3d
    python -m ca_debug.shared_mem_equivalence --size 32 --steps 30
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


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'low': 4, 'ok': 5, 'skip': 6}


def _grade(max_diff: float) -> str:
    if not np.isfinite(max_diff):
        return 'crit'
    if max_diff == 0.0:
        return 'ok'
    if max_diff <= 1e-6:
        return 'low'
    if max_diff <= 1e-4:
        return 'med'
    if max_diff <= 1e-2:
        return 'high'
    return 'crit'


# ---------------------------------------------------------------------------
# CA_RULES monkey-patch
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _force_shared_mem(value: int):
    """Prepend ``#define USE_SHARED_MEM <value>`` to every CA_RULES body.

    Restores the original strings on exit.  Doing this at module level
    forces the macro to a known value for any shader the rule dispatches,
    regardless of which keys its preset composes.
    """
    from simulator import CA_RULES
    backup = dict(CA_RULES)
    try:
        prefix = f'\n#define USE_SHARED_MEM {value}\n'
        for k, v in backup.items():
            CA_RULES[k] = prefix + v
        yield
    finally:
        CA_RULES.clear()
        CA_RULES.update(backup)


# ---------------------------------------------------------------------------
# Rule selection
# ---------------------------------------------------------------------------

def _candidate_rules():
    """Rules whose primary preset shader-key references USE_SHARED_MEM.

    Composed/multi-pass presets may reference several CA_RULES keys; we
    walk the resolved preset and admit the rule if ANY referenced key
    contains the conditional.
    """
    from simulator import CA_RULES, RULE_PRESETS, _resolve_composed_preset

    rules = []
    for name in sorted(RULE_PRESETS.keys()):
        try:
            preset = _resolve_composed_preset(name)
        except Exception:  # noqa: BLE001  preset lookup failure - skip the rule
            continue
        # Skip non-voxel kinds — they don't go through the lap19 path.
        if preset.get('kind') == 'viewport':
            continue
        if preset.get('agent_count') or 'entity_arena' in preset:
            continue
        # Collect every shader key the preset will dispatch.
        keys = set()
        if preset.get('shader'):
            keys.add(preset['shader'])
        for pspec in preset.get('passes') or []:
            if pspec.get('kind') in (None, 'voxel') and pspec.get('shader'):
                keys.add(pspec['shader'])
        # A key qualifies if it lives in CA_RULES and uses the macro.
        if any(k in CA_RULES and '#if USE_SHARED_MEM' in CA_RULES[k]
               for k in keys):
            rules.append(name)
    return rules


def _select_rules(args):
    if args.rules:
        names = [r.strip() for r in args.rules.split(',') if r.strip()]
        return names
    rules = _candidate_rules()
    if args.skip:
        skip = {s.strip() for s in args.skip.split(',') if s.strip()}
        rules = [r for r in rules if r not in skip]
    return rules


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _read_all(runner) -> list[np.ndarray]:
    """Return [pair1, pair2?] grids — mirrors ca_debug/determinism.py."""
    grids = [np.asarray(runner.read_grid()).copy()]
    tex_a2 = getattr(runner, 'tex_a2', None)
    if tex_a2 is not None:
        try:
            src = (tex_a2 if getattr(runner, 'ping2', 0) == 0
                   else runner.tex_b2)
            raw = np.frombuffer(src.read(),
                                dtype=runner._tex_np_dtype).reshape(
                runner.size, runner.size, runner.size, 4)
            grids.append(raw.astype(np.float32, copy=True))
        except Exception:  # noqa: BLE001  optional pair2 readback - never fatal
            pass
    return grids


def _run_once(ctx, rule, *, size, steps, seed, shared_mem):
    from test_harness import HeadlessRunner
    with _force_shared_mem(shared_mem):
        with contextlib.redirect_stdout(io.StringIO()):
            r = HeadlessRunner(ctx, rule, size=size, seed=seed)
        try:
            for _ in range(steps):
                r.step()
            return _read_all(r)
        finally:
            rel = getattr(r, 'release', None)
            if callable(rel):
                try: rel()
                except Exception: pass  # noqa: BLE001  GL release - never fatal


def _diff(a, b):
    if a.shape != b.shape:
        return {'shape_mismatch': (a.shape, b.shape),
                'max_diff': float('inf'),
                'n_diff_cells': int(np.prod(a.shape))}
    nan_a = ~np.isfinite(a)
    nan_b = ~np.isfinite(b)
    nan_only_a = int((nan_a & ~nan_b).sum())
    nan_only_b = int((~nan_a & nan_b).sum())
    both = ~nan_a & ~nan_b
    if both.any():
        d = np.abs(a[both] - b[both])
        max_diff = float(d.max())
        n_diff = int(np.any((a != b) & both, axis=-1).sum())
    else:
        max_diff = 0.0
        n_diff = 0
    return {'max_diff': max_diff,
            'n_diff_cells': n_diff,
            'nan_only_a': nan_only_a,
            'nan_only_b': nan_only_b}


def _run_rule(ctx, rule, args):
    try:
        a = _run_once(ctx, rule, size=args.size, steps=args.steps,
                      seed=args.seed, shared_mem=0)
        b = _run_once(ctx, rule, size=args.size, steps=args.steps,
                      seed=args.seed, shared_mem=1)
    except Exception as e:  # noqa: BLE001  per-rule crash - record and continue
        return {'rule': rule, 'grade': 'err',
                'error': f'{type(e).__name__}: {e}',
                'tb': traceback.format_exc(), 'max_diff': float('nan'),
                'pairs': []}
    if len(a) != len(b):
        return {'rule': rule, 'grade': 'err',
                'error': f'pair-count mismatch {len(a)} vs {len(b)}',
                'max_diff': float('nan'), 'pairs': []}
    pairs = []
    max_diff = 0.0
    nan_a_total = 0
    nan_b_total = 0
    for i, (x, y) in enumerate(zip(a, b)):
        s = _diff(x, y)
        s['pair_idx'] = i
        pairs.append(s)
        if s['max_diff'] > max_diff or not np.isfinite(s['max_diff']):
            max_diff = s['max_diff']
        nan_a_total += s.get('nan_only_a', 0)
        nan_b_total += s.get('nan_only_b', 0)
    grade = _grade(max_diff)
    if nan_a_total + nan_b_total > 0:
        grade = 'crit'
    return {'rule': rule, 'grade': grade, 'max_diff': max_diff,
            'pairs': pairs,
            'nan_only_direct': nan_a_total,
            'nan_only_shared': nan_b_total}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names (default: auto-detect).')
    ap.add_argument('--size', type=int, default=32,
                    help='Grid size — keep a multiple of 8 (default: 32).')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=1001)
    ap.add_argument('--skip', help='Comma-separated rules to omit.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='low',
                    help='Min severity to print (default: low — anything not bit-exact).')
    ap.add_argument('--json', help='Write per-rule report JSON to this path.')
    args = ap.parse_args(argv)

    from test_harness import create_headless_context
    _window, ctx = create_headless_context()

    rules = _select_rules(args)
    rows = []
    t0 = time.perf_counter()
    for i, rule in enumerate(rules, 1):
        sys.stdout.write(f"\r[{i:>3}/{len(rules)}] {rule:<42}")
        sys.stdout.flush()
        rows.append(_run_rule(ctx, rule, args))
    sys.stdout.write('\r' + ' ' * 70 + '\r')
    elapsed = time.perf_counter() - t0

    counts = {k: 0 for k in _SEV_ORDER}
    for r in rows:
        counts[r['grade']] = counts.get(r['grade'], 0) + 1
    sev_cap = _SEV_ORDER[args.severity]
    rows_sorted = sorted(
        rows,
        key=lambda r: (_SEV_ORDER.get(r['grade'], 9),
                       -(r.get('max_diff') if np.isfinite(r.get('max_diff', 0.0)) else 1e18)))

    print(f"shared-mem equivalence probe (size={args.size}, steps={args.steps}, "
          f"seed={args.seed}) — {elapsed:.1f}s, {len(rules)} rules")
    print(f"{'SEV':<5} {'RULE':<42} {'MAX_DIFF':>12}  {'N_DIFF':>10}  NOTES")
    print('-' * 96)
    for r in rows_sorted:
        if _SEV_ORDER.get(r['grade'], 9) > sev_cap:
            continue
        md = r.get('max_diff', float('nan'))
        md_s = '         nan' if (md != md) else f'{md:12.4g}'  # noqa: PLR0124
        n_diff = sum(p.get('n_diff_cells', 0) for p in r.get('pairs', []))
        notes = ''
        if r.get('error'):
            notes = r['error'][:80]
        else:
            nd = r.get('nan_only_direct', 0)
            ns = r.get('nan_only_shared', 0)
            if nd or ns:
                notes = f'asym-NaN direct={nd} shared={ns}'
        print(f"{r['grade']:<5} {r['rule']:<42} {md_s}  {n_diff:>10}  {notes}")
    summary = '  '.join(f'{k}={counts[k]}' for k in _SEV_ORDER
                        if counts.get(k))
    print(f"\nSummary: {summary or 'all-ok'}  (n={len(rows)})")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'args': vars(args), 'rows': rows, 'elapsed_s': elapsed},
                      f, indent=2, default=str)
        print(f"Wrote {args.json}")
    bad = counts.get('crit', 0) + counts.get('err', 0)
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
