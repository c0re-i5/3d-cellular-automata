"""init-variant smoke probe.

Many presets list multiple ``init_variants`` — alternative initial-
condition functions selectable from the GUI.  Some are used heavily
(default lenia FBM, gray-scott blob); others are rarely picked and
may have rotted (missing helper, wrong density, all-zero output).

This probe constructs a fresh ``HeadlessRunner`` for *each* declared
init variant of *each* rule, reads the initial state, steps a small
handful of times, and grades the result.

Per variant, we check:

    err    crash during runner construction or stepping.
    crit   NaN or Inf in initial state or after stepping.
    high   initial state is degenerate (``>99%`` of voxels share a
           single value) AND the field does not evolve in
           ``--steps`` steps — i.e. the variant produces an init
           that the rule cannot move off.  Sparse seed inits (a
           handful of bright voxels in a dark cube) are common and
           legitimate, so degeneracy alone is not flagged.
    med    not degenerate, but field did not change in ``--steps``
           steps (frozen).
    ok     finite, evolves.
    n/a    rule has no usable init variants to test (kind=viewport,
           agent-only, audit_skip, etc.).

A typical bug caught: a variant references a helper that was renamed
or moved, so the runner falls back to all-zero — caught by ``high``.
Another: a variant uses a density value that no longer makes sense
for the rule, so the rule freezes — caught by ``med``.

Usage::

    python -m ca_debug.init_variants
    python -m ca_debug.init_variants --rules lenia_3d,smoothlife_3d
    python -m ca_debug.init_variants --size 64 --steps 5 --severity med
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


def _read_main(runner) -> np.ndarray:
    return np.asarray(runner.read_grid()).copy()


def _construct(ctx, rule: str, size: int, seed: int,
               init_override: str | None):
    from test_harness import HeadlessRunner
    with contextlib.redirect_stdout(io.StringIO()):
        return HeadlessRunner(ctx, rule, size=size, seed=seed,
                              init_override=init_override)


def _saturation_fraction(grid: np.ndarray) -> tuple[float, float]:
    """Return (mode_fraction, n_unique_finite_values_capped).

    ``mode_fraction`` is the fraction of voxels equal (within 1e-6
    of field scale) to the most common value.  If the entire field
    is one constant, this is 1.0.  A healthy CA init has structure,
    so the fraction is well below 1.
    """
    flat = grid.reshape(-1).astype(np.float64, copy=False)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 1.0, 0
    scale = max(float(np.abs(flat).max()), 1e-6)
    # Discretise to a coarse bin width of 1% of scale and take the mode.
    bins = np.round(flat / (0.01 * scale)).astype(np.int64)
    _, counts = np.unique(bins, return_counts=True)
    mode_frac = float(counts.max()) / float(flat.size)
    # Cap unique-count reporting so we don't try to log millions.
    n_unique = int(min(counts.size, 9999))
    return mode_frac, n_unique


def _run_one_variant(ctx, rule: str, variant: str | None,
                     size: int, seed: int, steps: int) -> dict:
    """Run a single init variant; return a metrics dict.  Never raises."""
    try:
        r = _construct(ctx, rule, size, seed, variant)
    except Exception as e:  # noqa: BLE001  construction crash captured
        return {'variant': variant, 'grade': 'err',
                'phase': 'construct',
                'error': f'{type(e).__name__}: {e}',
                'tb': traceback.format_exc()}
    try:
        grid0 = _read_main(r)
        n_nan0 = int((~np.isfinite(grid0)).sum())
        n_inf0 = int(np.isinf(grid0).sum())
        mode_frac0, n_unique0 = _saturation_fraction(grid0)
        mean0 = float(np.nanmean(grid0))
        std0 = float(np.nanstd(grid0))

        # Step and capture post-step state.
        post_err: str | None = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(steps):
                    r.step()
            grid_n = _read_main(r)
        except Exception as e:  # noqa: BLE001  stepping crash captured
            post_err = f'{type(e).__name__}: {e}'
            grid_n = grid0  # nothing else to report against
        n_nan_n = int((~np.isfinite(grid_n)).sum())
        n_inf_n = int(np.isinf(grid_n)).sum() if grid_n is grid0 else int(
            np.isinf(grid_n).sum())

        # Did the state change?
        scale = max(float(np.nanstd(grid0)),
                    float(np.nanstd(grid_n)), 1e-6)
        finite = np.isfinite(grid0) & np.isfinite(grid_n)
        if finite.any():
            max_abs_delta = float(np.max(np.abs(grid0[finite] - grid_n[finite])))
        else:
            max_abs_delta = 0.0
        evolved = max_abs_delta > 1e-6 * scale
    finally:
        try: r.release()
        except Exception: pass  # noqa: BLE001

    out: dict = {
        'variant': variant,
        'n_nan_init': n_nan0, 'n_inf_init': n_inf0,
        'n_nan_step': n_nan_n, 'n_inf_step': n_inf_n,
        'mode_frac_init': mode_frac0,
        'n_unique_init': n_unique0,
        'mean_init': mean0, 'std_init': std0,
        'max_abs_delta': max_abs_delta,
    }
    if post_err is not None:
        out['grade'] = 'err'
        out['phase'] = 'step'
        out['error'] = post_err
        return out

    # Grade in severity order.
    if n_nan0 + n_inf0 > 0:
        out['grade'] = 'crit'
        out['reason'] = f'init has NaN/Inf ({n_nan0}/{n_inf0})'
        return out
    if n_nan_n + n_inf_n > 0:
        out['grade'] = 'crit'
        out['reason'] = f'step has NaN/Inf ({n_nan_n}/{n_inf_n})'
        return out
    out['evolved'] = evolved
    degenerate = mode_frac0 >= 0.99
    if not evolved and degenerate:
        out['grade'] = 'high'
        out['reason'] = (
            f'degenerate init ({mode_frac0:.1%} mode) AND no change '
            f'after {steps} steps (max|Δ|={max_abs_delta:.2g}, '
            f'scale={scale:.2g})')
        return out
    if not evolved:
        out['grade'] = 'med'
        out['reason'] = (
            f'no change after {steps} steps '
            f'(max|Δ|={max_abs_delta:.2g}, scale={scale:.2g})')
        return out
    out['grade'] = 'ok'
    return out


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
    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(',') if s.strip()}
        rules = [r for r in rules if r not in skip_set]
    return rules


def _variants_for_rule(preset: dict) -> list[str]:
    """Return the de-duplicated list of init variants to test.

    Always includes the declared ``init``.  Variants listed in
    ``init_variants`` are appended (the GUI exposes them as
    alternatives).  An empty list means there is nothing to probe.
    """
    seen: list[str] = []
    primary = preset.get('init')
    if primary:
        seen.append(primary)
    for v in (preset.get('init_variants') or []):
        if v and v not in seen:
            seen.append(v)
    return seen


def _probe_rule(ctx, rule: str, size: int, steps: int, seed: int) -> dict:
    from simulator import _resolve_composed_preset
    try:
        preset = _resolve_composed_preset(rule)
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'worst_grade': 'err',
                'error': f'preset resolve: {type(e).__name__}: {e}',
                'variants': []}
    variants = _variants_for_rule(preset)
    if not variants:
        return {'rule': rule, 'worst_grade': 'n/a',
                'reason': 'no init declared', 'variants': []}
    per_variant: list[dict] = []
    for v in variants:
        per_variant.append(_run_one_variant(ctx, rule, v, size, seed, steps))
    worst = min((p['grade'] for p in per_variant),
                key=lambda g: _SEV_ORDER.get(g, 9))
    return {'rule': rule, 'worst_grade': worst,
            'declared_init': preset.get('init'),
            'n_variants': len(variants),
            'variants': per_variant}


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names (default: all).')
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--steps', type=int, default=5,
                    help='Steps per variant before re-reading state (default: 5).')
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
    n_variants_total = 0
    for i, rule in enumerate(rules, 1):
        sys.stdout.write(f"\r[{i:>3}/{len(rules)}] {rule:<42}")
        sys.stdout.flush()
        row = _probe_rule(ctx, rule, args.size, args.steps, args.seed)
        rows.append(row)
        n_variants_total += len(row.get('variants') or [])
    sys.stdout.write('\r' + ' ' * 70 + '\r')
    elapsed = time.perf_counter() - t0

    rule_counts: dict[str, int] = {k: 0 for k in _SEV_ORDER}
    variant_counts: dict[str, int] = {k: 0 for k in _SEV_ORDER}
    for r in rows:
        rule_counts[r['worst_grade']] = rule_counts.get(r['worst_grade'], 0) + 1
        for v in r.get('variants') or []:
            variant_counts[v['grade']] = variant_counts.get(v['grade'], 0) + 1

    rows_sorted = sorted(rows,
                         key=lambda r: _SEV_ORDER.get(r['worst_grade'], 9))
    min_sev = _SEV_ORDER[args.severity]

    print(f"init-variants probe (size={args.size}, steps={args.steps}, "
          f"seed={args.seed}) -- {elapsed:.1f}s "
          f"({n_variants_total} variants across {len(rows)} rules)")
    print(f"{'SEV':<6} {'RULE':<42}  {'VARIANT':<28}  NOTES")
    print('-' * 130)
    for r in rows_sorted:
        if _SEV_ORDER.get(r['worst_grade'], 9) > min_sev:
            continue
        if r.get('error') and not r.get('variants'):
            print(f"{r['worst_grade']:<6} {r['rule']:<42}  "
                  f"{'(resolve)':<28}  {str(r['error'])[:60]}")
            continue
        for v in r['variants']:
            if _SEV_ORDER.get(v['grade'], 9) > min_sev:
                continue
            note_parts: list[str] = []
            if v.get('reason'):
                note_parts.append(v['reason'])
            if v.get('error'):
                note_parts.append(str(v['error'])[:60])
            note = '  '.join(note_parts) or '-'
            print(f"{v['grade']:<6} {r['rule']:<42}  "
                  f"{str(v.get('variant') or '?')[:28]:<28}  {note}")
    rule_summary = '  '.join(f'{k}={rule_counts[k]}' for k in _SEV_ORDER
                             if rule_counts.get(k))
    var_summary = '  '.join(f'{k}={variant_counts[k]}' for k in _SEV_ORDER
                            if variant_counts.get(k))
    print(f"\nRules:    {rule_summary}  (n={len(rows)})")
    print(f"Variants: {var_summary}  (n={n_variants_total})")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'args': vars(args), 'rows': rows,
                       'elapsed_s': elapsed}, f, indent=2, default=str)
        print(f"Wrote {args.json}")
    return (0 if rule_counts.get('crit', 0) == 0
            and rule_counts.get('err', 0) == 0 else 1)


if __name__ == '__main__':
    sys.exit(main())
