"""dt-convergence probe for PDE-style CA rules.

Many rules integrate a continuous field with an explicit Euler step
``u_field += u_dt * rhs(u_field)``.  The harness exposes the integrator
timestep as the ``u_dt`` uniform and the per-preset ``dt`` parameter.

For a correctly-implemented explicit integrator, refining the timestep
should *converge*: halving ``dt`` (and doubling the step count to keep
total simulated time fixed) should produce a result that is closer to
the dt/4 result than the original.

This probe runs each rule at three timesteps that span 4× refinement
over the same total simulated time and grades:

    n/a    rule has no ``u_dt`` uniform on any pass, or the uniform is
           plumbed but functionally ignored (preflight identity test:
           bit-identical output at dt = dt0 vs 4*dt0 after one step).
    crit   NaN/Inf appears at the smaller dt that wasn't present at
           the larger dt (a *smaller* timestep should be *more* stable;
           inverted stability is a bug indicator).
    high   err_BC > 2 * err_AB  (refinement actively diverges).
    med    err_BC > 1.1 * err_AB  (refinement does not converge cleanly).
    ok     err_BC <= 1.1 * err_AB  (refinement converges).

Errors are mean absolute difference between the final grids of the
trials, taken over all finite cells.  Trial pair A/B uses (dt0, N) vs
(dt0/2, 2N); trial pair B/C uses (dt0/2, 2N) vs (dt0/4, 4N).

Usage::

    python -m ca_debug.dt_convergence
    python -m ca_debug.dt_convergence --rules lenia_3d,smoothlife_3d
    python -m ca_debug.dt_convergence --steps 20 --json /tmp/dt.json
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


def _read_pair(runner) -> list[np.ndarray]:
    """Return list of grids: main field, optional field2."""
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
    """Concatenate per-pair grids along channel axis for diffing."""
    return np.concatenate([g.reshape(-1) for g in grids])


def _uses_dt(runner) -> bool:
    """True iff at least one compute pass has a u_dt uniform handle."""
    per_pass = getattr(runner, '_u_per_pass', None)
    if not per_pass:
        return False
    return any(u.get('dt') is not None for u in per_pass)


def _dt_has_effect(ctx, rule: str, size: int, seed: int, dt0: float) -> bool:
    """Preflight: does the rule's output depend on dt?

    Some rules plumb the ``u_dt`` uniform into shaders that don't
    actually use it (or multiply by 0).  We detect this by running a
    single step at dt0 vs 4*dt0 and comparing the final state.  If
    bit-identical, dt is effectively ignored.
    """
    from test_harness import HeadlessRunner
    with contextlib.redirect_stdout(io.StringIO()):
        r1 = HeadlessRunner(ctx, rule, size=size, seed=seed, dt=dt0)
        r1.step()
        s1 = _stack_grids(_read_pair(r1))
        try: r1.release()
        except Exception: pass  # noqa: BLE001
        r2 = HeadlessRunner(ctx, rule, size=size, seed=seed, dt=dt0 * 4.0)
        r2.step()
        s2 = _stack_grids(_read_pair(r2))
        try: r2.release()
        except Exception: pass  # noqa: BLE001
    if s1.shape != s2.shape:
        return True
    return not np.array_equal(s1, s2)


def _run_trial(ctx, rule: str, size: int, seed: int, dt: float, steps: int) -> dict:
    from test_harness import HeadlessRunner
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = HeadlessRunner(ctx, rule, size=size, seed=seed, dt=dt)
            for _ in range(steps):
                r.step()
            state = _stack_grids(_read_pair(r))
            try: r.release()
            except Exception: pass  # noqa: BLE001
    except Exception as e:  # noqa: BLE001  per-trial crash captured
        return {'crashed': True,
                'error': f'{type(e).__name__}: {e}',
                'tb': traceback.format_exc()}
    n_nan = int((~np.isfinite(state)).sum())
    n_inf = int(np.isinf(state).sum())
    return {'crashed': False, 'state': state,
            'n_nan': n_nan, 'n_inf': n_inf,
            'mean': float(np.nanmean(state)),
            'std': float(np.nanstd(state))}


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean |a - b| over cells where BOTH are finite.  NaN if no overlap."""
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs(a[mask] - b[mask])))


def _grade(trials: dict[str, dict]) -> tuple[str, dict]:
    """Returns (grade, metrics_dict)."""
    metrics: dict = {}
    # Crash → err
    for name in ('A', 'B', 'C'):
        if trials[name].get('crashed'):
            metrics['crashed_in'] = name
            metrics['error'] = trials[name].get('error')
            return 'err', metrics
    # NaN/Inf appearing in B or C that wasn't in A → crit (inverted stability)
    a_bad = trials['A']['n_nan'] + trials['A']['n_inf']
    b_bad = trials['B']['n_nan'] + trials['B']['n_inf']
    c_bad = trials['C']['n_nan'] + trials['C']['n_inf']
    metrics.update({'A_bad': a_bad, 'B_bad': b_bad, 'C_bad': c_bad})
    if a_bad == 0 and (b_bad > 0 or c_bad > 0):
        return 'crit', metrics
    # Compute pairwise errors
    err_ab = _mean_abs_diff(trials['A']['state'], trials['B']['state'])
    err_bc = _mean_abs_diff(trials['B']['state'], trials['C']['state'])
    metrics['err_AB'] = err_ab
    metrics['err_BC'] = err_bc
    # If neither side moved, mark ok (nothing to test).
    if not np.isfinite(err_ab) or not np.isfinite(err_bc):
        return 'ok', metrics
    # Both essentially zero (no evolution at this step count, or trivially
    # stable) -> ok.  Threshold relative to field scale.
    scale = max(trials['C']['std'], 1e-6)
    if err_ab < 1e-7 * scale and err_bc < 1e-7 * scale:
        metrics['no_evolution'] = True
        return 'ok', metrics
    if err_ab < 1e-12:
        # err_AB is effectively zero -> refinement didn't change anything
        # at the coarsest step, can't compute a meaningful ratio.
        return 'ok', metrics
    ratio = err_bc / err_ab
    metrics['ratio_BC_over_AB'] = ratio
    if ratio > 2.0:
        return 'high', metrics
    if ratio > 1.1:
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
        # Particle-SSBO rules piggyback on a no-op voxel shader (e.g.
        # ``particle_lenia`` uses wave_3d at zero).  The voxel u_dt is
        # functionally inert there; the real timestep is ``particle_dt``.
        # Skip them — this probe targets voxel integrators.
        if preset.get('particle_count'):
            continue
        rules.append(r)
    if args.skip_flagship:
        rules = [r for r in rules if not r.startswith('flagship_')]
    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(',') if s.strip()}
        rules = [r for r in rules if r not in skip_set]
    return rules


def _probe_rule(ctx, rule: str, size: int, base_steps: int, seed: int) -> dict:
    from simulator import _resolve_composed_preset
    try:
        preset = _resolve_composed_preset(rule)
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'grade': 'err',
                'error': f'preset resolve: {type(e).__name__}: {e}'}
    dt0 = preset.get('dt')
    if dt0 is None or dt0 <= 0:
        return {'rule': rule, 'grade': 'n/a', 'reason': 'no preset dt'}
    # Cheap uses-dt check using a one-shot runner construction.
    from test_harness import HeadlessRunner
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r0 = HeadlessRunner(ctx, rule, size=size, seed=seed)
            uses_dt = _uses_dt(r0)
            try: r0.release()
            except Exception: pass  # noqa: BLE001
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'grade': 'err',
                'error': f'construct: {type(e).__name__}: {e}'}
    if not uses_dt:
        return {'rule': rule, 'grade': 'n/a', 'reason': 'no u_dt uniform'}
    # Preflight: does dt actually change the output?
    try:
        effective = _dt_has_effect(ctx, rule, size, seed, dt0)
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'grade': 'err',
                'error': f'preflight: {type(e).__name__}: {e}'}
    if not effective:
        return {'rule': rule, 'grade': 'n/a',
                'reason': 'dt has u_dt uniform but no effect on output'}
    # Run the three trials.
    trials = {}
    for name, (dt_mul, step_mul) in (
        ('A', (1.0, 1)), ('B', (0.5, 2)), ('C', (0.25, 4))
    ):
        trials[name] = _run_trial(ctx, rule, size, seed,
                                  dt=float(dt0) * dt_mul,
                                  steps=base_steps * step_mul)
    grade, metrics = _grade(trials)
    out: dict = {'rule': rule, 'grade': grade, 'dt0': float(dt0),
                 'base_steps': base_steps, 'size': size, **metrics}
    # Drop large state arrays from the report.
    return out


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names (default: all).')
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--steps', type=int, default=20,
                    help='Base step count N for trial A; B uses 2N, C uses 4N.')
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

    print(f"dt-convergence probe (size={args.size}, base_steps={args.steps}, "
          f"seed={args.seed}) -- {elapsed:.1f}s")
    print(f"{'SEV':<6} {'RULE':<42}  NOTES")
    print('-' * 96)
    for r in rows_sorted:
        if _SEV_ORDER.get(r['grade'], 9) > min_sev:
            continue
        note_parts: list[str] = []
        if 'reason' in r:
            note_parts.append(r['reason'])
        if 'error' in r:
            note_parts.append(str(r['error'])[:60])
        if 'ratio_BC_over_AB' in r:
            note_parts.append(
                f"err_AB={r['err_AB']:.3g} err_BC={r['err_BC']:.3g} "
                f"ratio={r['ratio_BC_over_AB']:.2f}")
        if r.get('B_bad') or r.get('C_bad'):
            note_parts.append(
                f"A_bad={r.get('A_bad')} B_bad={r.get('B_bad')} "
                f"C_bad={r.get('C_bad')}")
        note = '  '.join(note_parts) or '-'
        print(f"{r['grade']:<6} {r['rule']:<42}  {note}")
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
