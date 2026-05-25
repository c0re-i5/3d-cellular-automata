"""boundary-mode honour probe.

Each preset declares a ``boundary`` field (``toroidal``, ``clamped``,
``mirror`` / ``neumann`` / ``reflect`` / ``zero_flux``).  The harness
maps these to integer codes 0/1/2 and writes them into the ``u_boundary``
shader uniform on every pass that declares one.  Most shaders consume
``u_boundary`` indirectly via a ``fetch()`` / ``sample()`` helper that
chooses between periodic / clamped / mirrored indexing.

A rule's output should therefore *change* when the boundary mode
changes — at least near the domain edges, and ideally only there.  If
it doesn't change, either:

  * the uniform is declared but never read (silent ignore), or
  * the helper is hard-coded to one mode regardless of the uniform, or
  * the field is genuinely zero / constant near the edges in this
    initial condition window (false positive).

This probe runs each rule twice with two different boundary modes and
grades the result:

    n/a    no pass has a ``u_boundary`` uniform (rule is hard-coded
           periodic or doesn't sample neighbours).
    err    crash during construction or stepping.
    crit   alt boundary produces NaN/Inf when default doesn't.
    high   ``u_boundary`` uniform exists on >=1 pass, but the two
           outputs are bit-identical (uniform plumbed but ignored).
    med    outputs differ, but edge slab diff is *smaller* than the
           interior diff (boundary appears to affect interior more
           than edges — suspicious).
    ok     outputs differ, with the change concentrated near the
           boundary as expected.

Usage::

    python -m ca_debug.boundary_honour
    python -m ca_debug.boundary_honour --rules lenia_3d,smoothlife_3d
    python -m ca_debug.boundary_honour --size 64 --steps 30 --severity med
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


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'ok': 4, 'n/a': 5}

# Boundaries the harness recognises (anything else maps to 0 / toroidal).
_PERIODIC = {'toroidal', 'periodic', 'wrap'}
_CLAMPED = {'clamped'}
_MIRROR = {'mirror', 'neumann', 'reflect', 'zero_flux'}


def _read_pair(runner) -> list[np.ndarray]:
    grids = [np.asarray(runner.read_grid()).copy()]
    tex_a2 = getattr(runner, 'tex_a2', None)
    if tex_a2 is not None:
        try:
            src = tex_a2 if getattr(runner, 'ping2', 0) == 0 else runner.tex_b2
            raw = np.frombuffer(src.read(),
                                dtype=runner._tex_np_dtype).reshape(
                runner.size, runner.size, runner.size, 4)
            grids.append(raw.astype(np.float32, copy=True))
        except Exception:  # noqa: BLE001  optional pair2 readback
            pass
    return grids


def _stack_grids(grids: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([g.reshape(-1) for g in grids])


def _uses_boundary(runner) -> bool:
    """True iff at least one compute pass has a u_boundary uniform handle."""
    per_pass = getattr(runner, '_u_per_pass', None)
    if not per_pass:
        return False
    return any(u.get('boundary') is not None for u in per_pass)


_ALL_MODES = ('toroidal', 'clamped', 'mirror')


def _normalise_declared(declared: str) -> str:
    d = (declared or 'toroidal').lower()
    if d in _PERIODIC:
        return 'toroidal'
    if d in _CLAMPED:
        return 'clamped'
    if d in _MIRROR:
        return 'mirror'
    return 'toroidal'


def _run_trial(ctx, rule: str, size: int, seed: int, steps: int,
               boundary: str) -> dict:
    from test_harness import HeadlessRunner
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = HeadlessRunner(ctx, rule, size=size, seed=seed)
            # Mutate the runner's resolved preset before stepping; the
            # boundary value is re-read on every step.
            r.preset['boundary'] = boundary
            for _ in range(steps):
                r.step()
            grids = _read_pair(r)
            state = _stack_grids(grids)
            shape = grids[0].shape  # (z, y, x, c)
            try: r.release()
            except Exception: pass  # noqa: BLE001
    except Exception as e:  # noqa: BLE001  per-trial crash captured
        return {'crashed': True,
                'error': f'{type(e).__name__}: {e}',
                'tb': traceback.format_exc()}
    n_nan = int((~np.isfinite(state)).sum())
    n_inf = int(np.isinf(state).sum())
    return {'crashed': False, 'state': state, 'main_shape': shape,
            'main_grid': grids[0],
            'n_nan': n_nan, 'n_inf': n_inf,
            'mean': float(np.nanmean(state)),
            'std': float(np.nanstd(state))}


def _edge_mask(shape: tuple[int, ...], k: int) -> np.ndarray | None:
    """Boolean (z,y,x) mask selecting the outer k-voxel shell.  None if grid is too small."""
    z, y, x = shape[:3]
    if min(z, y, x) <= 2 * k:
        return None
    m = np.zeros((z, y, x), dtype=bool)
    m[:k, :, :] = True
    m[-k:, :, :] = True
    m[:, :k, :] = True
    m[:, -k:, :] = True
    m[:, :, :k] = True
    m[:, :, -k:] = True
    return m


def _edge_interior_diff(g1: np.ndarray, g2: np.ndarray, k: int
                        ) -> tuple[float, float]:
    """Mean |g1-g2| over (edge slab, interior).  Both are (z,y,x,c)."""
    diff = np.abs(g1 - g2)
    mask_edge = _edge_mask(diff.shape, k)
    if mask_edge is None:
        return float(np.nanmean(diff)), 0.0
    mask_int = ~mask_edge
    de = diff[mask_edge]
    di = diff[mask_int]
    de = de[np.isfinite(de)]
    di = di[np.isfinite(di)]
    edge = float(de.mean()) if de.size else 0.0
    interior = float(di.mean()) if di.size else 0.0
    return edge, interior


def _edge_signal(grid: np.ndarray, k: int) -> float:
    """Max |value| on the outer k-voxel shell.

    A near-zero result means the field has not reached the boundary in any
    meaningful way: every boundary fetch is reading either zero or a
    constant so wrap / clamp / mirror cannot possibly differ.  We use
    max-abs rather than std because a field that is mostly zero with a
    tiny diffusive tail (e.g. Lenia growth fronts before they reach the
    edge) has nonzero std but functionally-zero values that any
    threshold-based rule will treat as identically dead.
    """
    mask = _edge_mask(grid.shape, k)
    if mask is None:
        edge = grid
    else:
        edge = grid[mask]
    edge = edge[np.isfinite(edge)]
    if edge.size == 0:
        return 0.0
    return float(np.max(np.abs(edge)))


def _grade(trials: dict[str, dict], k_edge: int) -> tuple[str, dict]:
    """Grade based on max pairwise diff across all three boundary modes."""
    metrics: dict = {}
    # Crash check.
    for name in trials:
        if trials[name].get('crashed'):
            metrics['crashed_in'] = name
            metrics['error'] = trials[name].get('error')
            return 'err', metrics
    # NaN/Inf appearing only in non-default trials.
    bad = {n: trials[n]['n_nan'] + trials[n]['n_inf'] for n in trials}
    metrics['bad_counts'] = bad
    if bad.get('default', 0) == 0 and any(v > 0 for k, v in bad.items() if k != 'default'):
        return 'crit', metrics

    # Field scale across all trials.
    scale = max(max(trials[n]['std'] for n in trials), 1e-6)
    metrics['scale'] = scale

    # Find the maximum pairwise difference across all mode pairs.
    names = list(trials.keys())
    best_pair: tuple[str, str] | None = None
    best_max_abs = 0.0
    best_g1: np.ndarray | None = None
    best_g2: np.ndarray | None = None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s1 = trials[names[i]]['state']
            s2 = trials[names[j]]['state']
            if s1.shape != s2.shape:
                continue
            finite = np.isfinite(s1) & np.isfinite(s2)
            if not finite.any():
                continue
            ma = float(np.max(np.abs(s1[finite] - s2[finite])))
            if ma > best_max_abs:
                best_max_abs = ma
                best_pair = (names[i], names[j])
                best_g1 = trials[names[i]]['main_grid']
                best_g2 = trials[names[j]]['main_grid']
    metrics['max_abs_diff'] = best_max_abs
    metrics['best_pair'] = best_pair

    # Outermost-face content (across all trials) determines whether the
    # test is informative at all.
    edge_sig = max(
        _edge_signal(trials[n]['main_grid'], k=1) for n in trials)
    metrics['edge_signal'] = edge_sig
    if edge_sig < 0.05 * scale:
        # Below ~5% of field scale, threshold-based rules (Lenia,
        # SmoothLife, NCA) treat phantom-neighbour values as
        # functionally-zero regardless of boundary mode, and the test
        # cannot distinguish honour from silent-ignore.
        metrics['reason'] = (
            f'no boundary contact (max|edge|={edge_sig:.2g} '
            f'< 5% of scale {scale:.2g})')
        return 'n/a', metrics

    # Bit-identical (or FP-shimmer-identical) across ALL pairs and
    # edge has signal -> uniform plumbed but ignored by the shader.
    if best_max_abs < 1e-6 * scale:
        return 'high', metrics

    edge, interior = _edge_interior_diff(best_g1, best_g2, k_edge)
    metrics['edge_diff'] = edge
    metrics['interior_diff'] = interior
    if interior > 2.0 * edge and interior > 1e-6 * scale:
        return 'med', metrics
    return 'ok', metrics


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
        # Skip particle-SSBO rules (their voxel shader is a no-op).
        if preset.get('particle_count'):
            continue
        # audit_skip rules are intentionally trivial (e.g. ``sandbox``
        # ships an empty world for brush-mode building); skip.
        if preset.get('audit_skip'):
            continue
        rules.append(r)
    if args.skip_flagship:
        rules = [r for r in rules if not r.startswith('flagship_')]
    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(',') if s.strip()}
        rules = [r for r in rules if r not in skip_set]
    return rules


def _probe_rule(ctx, rule: str, size: int, steps: int, seed: int) -> dict:
    from simulator import _resolve_composed_preset
    from test_harness import HeadlessRunner
    try:
        preset = _resolve_composed_preset(rule)
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'grade': 'err',
                'error': f'preset resolve: {type(e).__name__}: {e}'}
    declared = _normalise_declared(preset.get('boundary'))

    # Quick uses-boundary check on a one-shot construction.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r0 = HeadlessRunner(ctx, rule, size=size, seed=seed)
            has_uniform = _uses_boundary(r0)
            try: r0.release()
            except Exception: pass  # noqa: BLE001
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'grade': 'err',
                'error': f'construct: {type(e).__name__}: {e}'}
    if not has_uniform:
        return {'rule': rule, 'grade': 'n/a',
                'reason': 'no u_boundary uniform', 'declared': declared}

    # k = edge slab thickness in voxels for edge/interior diff stats.
    k_edge = max(2, size // 8)

    # Run all three boundary modes; the 'default' label tracks the
    # rule's declared mode for reporting purposes.
    trials: dict[str, dict] = {}
    for mode in _ALL_MODES:
        label = 'default' if mode == declared else mode
        trials[label] = _run_trial(ctx, rule, size, seed, steps, mode)
    grade, metrics = _grade(trials, k_edge)
    out: dict = {'rule': rule, 'grade': grade, 'declared': declared,
                 'size': size, 'steps': steps, **metrics}
    return out


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names (default: all).')
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--steps', type=int, default=70,
                    help='Steps per trial (default: 70).  Some PDE rules '
                    'with slow growth from centred ICs need 60+ steps for '
                    'the field to reach the boundary face.')
    ap.add_argument('--seed', type=int, default=1001)
    ap.add_argument('--skip-flagship', action='store_true')
    ap.add_argument('--skip', help='Comma-separated rules to omit.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='med',
                    help='Min severity to print (default: med).')
    ap.add_argument('--json', help='Write per-rule report JSON to this path.')
    args = ap.parse_args(argv)

    from test_harness import create_headless_context
    window, ctx = create_headless_context()

    rules = _select_rules(args)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, rule in enumerate(rules, 1):
        sys.stdout.write(f"\r[{i:>3}/{len(rules)}] {rule:<42}")
        sys.stdout.flush()
        rows.append(_probe_rule(ctx, rule, args.size, args.steps, args.seed))
    sys.stdout.write('\r' + ' ' * 70 + '\r')
    elapsed = time.perf_counter() - t0

    counts = {k: 0 for k in _SEV_ORDER}
    for r in rows:
        counts[r['grade']] = counts.get(r['grade'], 0) + 1

    rows_sorted = sorted(rows, key=lambda r: _SEV_ORDER.get(r['grade'], 9))
    min_sev = _SEV_ORDER[args.severity]

    print(f"boundary-honour probe (size={args.size}, steps={args.steps}, "
          f"seed={args.seed}) -- {elapsed:.1f}s")
    print(f"{'SEV':<6} {'RULE':<42}  {'DECL':<10}  NOTES")
    print('-' * 110)
    for r in rows_sorted:
        if _SEV_ORDER.get(r['grade'], 9) > min_sev:
            continue
        note_parts: list[str] = []
        if 'reason' in r:
            note_parts.append(r['reason'])
        if 'error' in r:
            note_parts.append(str(r['error'])[:60])
        if 'edge_diff' in r:
            note_parts.append(
                f"edge={r['edge_diff']:.3g} int={r['interior_diff']:.3g}")
        if r.get('max_abs_diff') is not None and 'edge_diff' not in r:
            note_parts.append(f"max|Δ|={r.get('max_abs_diff'):.3g}")
        if r.get('best_pair'):
            note_parts.append(f"pair={r['best_pair'][0]}/{r['best_pair'][1]}")
        if r.get('bad_counts'):
            bad = r['bad_counts']
            if any(bad.values()):
                note_parts.append(f"bad={bad}")
        note = '  '.join(note_parts) or '-'
        print(f"{r['grade']:<6} {r['rule']:<42}  "
              f"{r.get('declared','?'):<10}  {note}")
    summary = '  '.join(f'{k}={counts[k]}' for k in _SEV_ORDER
                        if counts.get(k))
    print(f"\nSummary: {summary}  (n={len(rows)})")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'args': vars(args), 'rows': rows,
                       'elapsed_s': elapsed}, f, indent=2, default=str)
        print(f"Wrote {args.json}")
    return 0 if counts.get('crit', 0) == 0 and counts.get('err', 0) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
