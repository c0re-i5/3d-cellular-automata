"""Non-aligned grid-size probe.

Every compute shader dispatches ceil(size/8)³ workgroups of local-size
8×8×8.  When ``size`` is not a multiple of 8 the trailing workgroups
spawn threads with ``pos >= u_size`` that *must* be culled by the
``if (any(greaterThanEqual(pos, ivec3(u_size)))) return;`` guard.  If a
shader skips the guard, returns before a subsequent ``barrier()``, or
indexes shared memory without bounds checks, the trailing slab of the
grid develops uninitialised values, NaNs, or simply crashes the driver.

This probe runs each rule at a small aligned size (64) as a baseline
and at four non-aligned sizes (65, 100, 127, 129).  For each result we
check::

    - no crash
    - all cells finite (no NaN/Inf)
    - field is alive (not all-zero, not saturated to a single value)

Severity::

    crit   crash, OOB NaN/Inf
    dead   field is all-zero or single-valued at probe size
    ok     no anomalies
    err    exception during run

The trailing-slab-vs-adjacent-slab z-score is computed for every
result and recorded in the JSON output as a diagnostic, but it is
NOT used as a grade.  Any non-periodic boundary condition (clamp,
mirror, OOB-read-as-zero) creates a real discontinuity at every
boundary plane, present at aligned sizes (64, 72, 128) just as much
as at non-aligned ones (65, 100, 127); the slab-z metric cannot
distinguish physical boundary effects from a real workgroup-dispatch
bug.  An actual dispatch bug would manifest as a crash, NaN/Inf, or
a dead field, all of which ARE flagged.

Usage::

    python -m ca_debug.aligned_sizes
    python -m ca_debug.aligned_sizes --rules lenia_3d --sizes 65,100,127,129
    python -m ca_debug.aligned_sizes --steps 20 --json /tmp/aligned.json
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


_SEV_ORDER = {'err': 0, 'crit': 1, 'dead': 2, 'ok': 3, 'n/a': 4}


def _trailing_slab_stats(grid: np.ndarray, size: int) -> dict:
    """Compare the LAST ``size % 8`` z-planes vs the ADJACENT INNER slab
    of equal thickness (the ``size % 8`` planes immediately before the
    trailing workgroup boundary).

    Comparing to the leading slab (opposite edge) or to the bulk
    interior generates massive false positives for any rule with a
    directional init, asymmetric forcing (gravity, biased rain
    sources), or a centred blob -- the trailing slab's mean can
    legitimately differ from the rest of the field by orders of
    magnitude as part of normal physics.

    What CAN'T be normal physics is a *sudden discontinuity* at the
    workgroup boundary itself.  A real dispatch bug (uninitialised
    trailing slab, skipped writes, OOB reads) shows up as a sharp
    jump between plane (size-slab_thick-1) and plane (size-slab_thick);
    a smooth gradient continuing through the boundary is benign.

    Returns z-score of trailing-slab mean vs adjacent-inner-slab
    statistics (per channel, take the worst).  At aligned sizes
    (size % 8 == 0) there is no slab; return zeros.  Static channels
    (inner_std == 0) are skipped.
    """
    slab_thick = size % 8
    if slab_thick == 0:
        return {'slab_thick': 0, 'mean_z': 0.0,
                'trail_mean': 0.0, 'ref_mean': 0.0, 'ref_std': 0.0,
                'slab_nan': 0, 'slab_inf': 0}
    # grid layout from read_grid: (z, y, x, c) per harness comment.
    trail = grid[size - slab_thick:size]
    ref = grid[size - 2 * slab_thick:size - slab_thick]
    slab_nan = int((~np.isfinite(trail)).sum())
    slab_inf = int(np.isinf(trail).sum())
    worst_z = 0.0
    worst_trail_mean = 0.0
    worst_ref_mean = 0.0
    worst_ref_std = 0.0
    nch = grid.shape[-1]
    for c in range(nch):
        fin_trail = trail[..., c][np.isfinite(trail[..., c])]
        fin_ref = ref[..., c][np.isfinite(ref[..., c])]
        if fin_ref.size == 0 or fin_trail.size == 0:
            continue
        ref_std = float(fin_ref.std())
        if ref_std < 1e-9:
            continue
        trail_mean = float(fin_trail.mean())
        ref_mean = float(fin_ref.mean())
        # Floor the denominator so we don't divide by machine-noise std.
        # 1e-6 absolute   -> kills false positives when both slabs are
        #                    effectively zero (rule hasn't propagated)
        # 1% of |ref|     -> demands a meaningfully large delta when
        #                    the reference slab is super-uniform but at
        #                    a nontrivial value.
        eff_std = max(ref_std, 1e-6, 0.01 * abs(ref_mean))
        z = abs(trail_mean - ref_mean) / eff_std
        if z > worst_z:
            worst_z = z
            worst_trail_mean = trail_mean
            worst_ref_mean = ref_mean
            worst_ref_std = ref_std
    return {'slab_thick': slab_thick, 'mean_z': worst_z,
            'trail_mean': worst_trail_mean, 'ref_mean': worst_ref_mean,
            'ref_std': worst_ref_std,
            'slab_nan': slab_nan, 'slab_inf': slab_inf}


def _grade(crashed: bool, total_nan: int, total_inf: int, alive: bool,
           slab: dict) -> str:
    if crashed:
        return 'crit'
    if total_nan or total_inf:
        return 'crit'
    if not alive:
        return 'dead'
    # The slab-vs-reference z-score is a diagnostic, NOT a severity
    # grade.  Any non-periodic boundary condition (clamp, mirror, or
    # a rule that reads OOB as zero) creates a real discontinuity at
    # every boundary plane — present at aligned sizes (64, 72, 128)
    # just as much as at non-aligned (65, 100, 127).  Comparing the
    # trailing slab against ANY reference (interior, leading slab,
    # adjacent inner slab) can't tell physics from a dispatch bug,
    # because both produce the same pattern.  The probe still records
    # mean_z in JSON for inspection, but only crash / NaN / Inf /
    # dead-field anomalies are flagged here.
    return 'ok'


def _is_alive(grid: np.ndarray) -> bool:
    """A field is alive if it has finite variance > 1e-9 anywhere."""
    fin = grid[np.isfinite(grid)]
    if fin.size == 0:
        return False
    return float(fin.std()) > 1e-9


def _read_full_state(runner) -> list[np.ndarray]:
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
        except Exception:  # noqa: BLE001  optional pair2 readback
            pass
    return grids


def _run_size(ctx, rule: str, size: int, steps: int, seed: int) -> dict:
    from test_harness import HeadlessRunner
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = HeadlessRunner(ctx, rule, size=size, seed=seed)
            for _ in range(steps):
                r.step()
            grids = _read_full_state(r)
            try: r.release()
            except Exception: pass  # noqa: BLE001  best-effort release
    except Exception as e:  # noqa: BLE001  per-rule crash captured
        return {'crashed': True,
                'error': f'{type(e).__name__}: {e}',
                'tb': traceback.format_exc(),
                'grade': 'crit'}
    if not grids:
        return {'crashed': True, 'error': 'empty grids', 'grade': 'crit'}
    total_nan = sum(int((~np.isfinite(g)).sum()) for g in grids)
    total_inf = sum(int(np.isinf(g).sum()) for g in grids)
    alive = any(_is_alive(g) for g in grids)
    # Aggregate slab stats: worst (highest mean_z) across pairs.
    slabs = [_trailing_slab_stats(g, size) for g in grids]
    worst = max(slabs, key=lambda s: s.get('mean_z', 0.0))
    grade = _grade(False, total_nan, total_inf, alive, worst)
    return {'crashed': False, 'grade': grade,
            'n_pairs': len(grids),
            'total_nan': total_nan, 'total_inf': total_inf,
            'alive': alive,
            'slab': worst}


def _select_rules(args) -> list[str]:
    from simulator import RULE_PRESETS, _resolve_composed_preset
    if args.rules:
        return [r.strip() for r in args.rules.split(',') if r.strip()]
    rules = []
    for r in sorted(RULE_PRESETS.keys()):
        try:
            preset = _resolve_composed_preset(r)
        except Exception:  # noqa: BLE001  preset lookup
            continue
        if preset.get('kind') == 'viewport':
            continue
        if preset.get('agent_count') or 'entity_arena' in preset:
            continue
        if (preset.get('passes') or [{}])[0].get('kind') == 'particle':
            continue
        rules.append(r)
    if args.skip_flagship:
        rules = [r for r in rules if not r.startswith('flagship_')]
    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(',') if s.strip()}
        rules = [r for r in rules if r not in skip_set]
    return rules


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names (default: all).')
    ap.add_argument('--sizes', default='64,65,100,127,129',
                    help='Comma-separated grid sizes (default: 64,65,100,127,129).')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=1001)
    ap.add_argument('--skip-flagship', action='store_true')
    ap.add_argument('--skip', help='Comma-separated rules to omit.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='dead',
                    help='Min severity to print (default: dead).')
    ap.add_argument('--json', help='Write per-rule report JSON to this path.')
    args = ap.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(',') if s.strip()]

    from test_harness import create_headless_context
    window, ctx = create_headless_context()

    rules = _select_rules(args)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, rule in enumerate(rules, 1):
        per_size = {}
        for sz in sizes:
            sys.stdout.write(f"\r[{i:>3}/{len(rules)}] {rule:<40} sz={sz:<4}")
            sys.stdout.flush()
            per_size[sz] = _run_size(ctx, rule, sz, args.steps, args.seed)
        # Worst severity across sizes determines the row's grade.
        worst_sev = min(_SEV_ORDER.get(per_size[sz].get('grade', 'ok'), 9)
                        for sz in sizes)
        worst_grade = next((g for g, v in _SEV_ORDER.items() if v == worst_sev),
                           'ok')
        rows.append({'rule': rule, 'grade': worst_grade,
                     'per_size': per_size})
    sys.stdout.write('\r' + ' ' * 70 + '\r')
    elapsed = time.perf_counter() - t0

    counts = {k: 0 for k in _SEV_ORDER}
    for r in rows:
        counts[r['grade']] = counts.get(r['grade'], 0) + 1

    rows_sorted = sorted(rows,
                         key=lambda r: _SEV_ORDER.get(r['grade'], 9))
    min_sev = _SEV_ORDER[args.severity]

    print(f"Non-aligned size probe (sizes={sizes}, steps={args.steps}, "
          f"seed={args.seed}) -- {elapsed:.1f}s")
    print(f"{'SEV':<6} {'RULE':<38}  PER-SIZE")
    print('-' * 96)
    for r in rows_sorted:
        if _SEV_ORDER.get(r['grade'], 9) > min_sev:
            continue
        cells = []
        for sz in sizes:
            v = r['per_size'][sz]
            g = v.get('grade', '?')
            note = ''
            if v.get('crashed'):
                note = f"!{v.get('error', '')[:25]}"
            elif v.get('total_nan') or v.get('total_inf'):
                note = f"nan={v.get('total_nan')} inf={v.get('total_inf')}"
            elif g == 'dead':
                note = 'dead'
            else:
                # Diagnostic: always show slab_z if interesting (>1)
                mz = v.get('slab', {}).get('mean_z', 0.0)
                if mz > 1.0:
                    note = f"slab_z={mz:.2g}"
            cells.append(f"{sz}:{g}{(' ' + note) if note else ''}")
        print(f"{r['grade']:<6} {r['rule']:<38}  " + ' | '.join(cells))
    summary = '  '.join(f'{k}={counts[k]}' for k in
                        ('crit', 'dead', 'ok', 'err')
                        if counts.get(k))
    print(f"\nSummary: {summary or 'all-ok'}  (n={len(rows)})")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'args': vars(args), 'sizes': sizes, 'rows': rows,
                       'elapsed_s': elapsed}, f, indent=2, default=str)
        print(f"Wrote {args.json}")
    return 0 if counts.get('crit', 0) == 0 and counts.get('err', 0) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
