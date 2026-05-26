"""Render-side visibility audit probe.

Probe #14 — every existing probe inspects the *simulation* state
(numbers in the grid).  The user, however, sees the *renderer's*
output — a thresholded mask on top of the same grid.  A preset can
evolve a perfectly reasonable simulation that the renderer culls
entirely, presenting an empty viewport (Bug J / `causal_ca`).

Per rule we replay headlessly to its audit horizon, sample the
``vis_default`` channel at checkpoints, and compute the same
visibility mask the GPU renderer would compute given the preset's
declared ``render_mode`` + ``voxel_threshold`` / ``iso_threshold`` /
``vis_range`` + ``vis_abs`` transform.  We then grade by the fraction
of the grid that would render at each checkpoint.

Modes handled (per simulator's ``renderer_mode`` string):
  * 'voxel'      → mask = val > voxel_threshold          (hard cut-off)
  * 'iso'        → mask = val > iso_threshold            (raymarch)
  * 'volumetric' → mask = normalised(val, vis_range) > 1e-3 floor
                   (volumetric raymarch accumulates alpha even for
                   faint values; we flag only true near-zero fields)

Grades:
  err   construction or stepping crashed.
  crit  zero visible voxels at the FINAL checkpoint
        (Bug J class — user opens rule, sees nothing, ever).
  high  visible voxels only after >75% of audit horizon
        (rule "wakes up" too late to look alive on first view).
  med   <0.05% of grid visible at the final checkpoint
        (dim/sparse render even if technically non-empty).
  ok    substantive image by the final checkpoint
        (>=0.05% visible, or visible by midpoint).
  skip  ``kind == 'viewport'`` (fractal raymarchers don't use the
        voxel-grid render path), or unknown render_mode.

Bonus diagnostic (does NOT affect grade): at the final checkpoint we
record (p1, p99) of the vis_default channel.  If the signal range
falls completely outside ``vis_range`` we note 'dim' or 'saturated' so
the colour-map utilisation can be triaged manually.

This DOES NOT check shader rendering correctness — it can't see the
fragment shader's output.  It checks whether the same mask the GPU
applies (per-voxel value > threshold) admits any voxels at all.

Usage::

    python -m ca_debug.visibility_audit
    python -m ca_debug.visibility_audit --cap 200 --severity med
    python -m ca_debug.visibility_audit --json /tmp/vis.json
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


_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'med': 3, 'ok': 4,
              'skip': 5, 'n/a': 6}

# Volumetric mode renders ANY voxel with normalised value above this
# floor (raymarch accumulates alpha aggressively).  Below this the
# image is effectively black.
_VOLUMETRIC_FLOOR = 1e-3

# Med threshold — "dim/sparse" image fraction.  0.05% of a 48^3 grid
# is ~55 voxels — visible-but-very-sparse.
_MED_FRACTION = 5e-4

# Default checkpoint fractions of audit horizon.
_CHECKPOINTS = (0.25, 0.5, 0.75, 1.0)

# vis_modes where channel[vis_default] + voxel/iso_threshold accurately
# models the renderer's visibility mask.  Composite modes (rgb_channels,
# hsv_phase, rgba_blend) combine multiple channels through their own
# fragment-shader logic; flagging on vis_default channel alone would
# produce false positives — defer those to manual review.
_DIRECT_VIS_MODES = {'density', 'bipolar', 'signed'}


def _read_main(runner) -> np.ndarray:
    """Read main grid as (D, H, W, C) ndarray."""
    g = runner.read_grid()
    return np.asarray(g)


def _vis_value(channel: np.ndarray, vis_abs) -> np.ndarray:
    """Apply the renderer's vis_abs transform.

    Mirrors simulator.py's fragment shader logic:
      0 / False → val as-is
      1 / True  → abs(val)
      2         → abs(val - 0.5) * 2.0   (signed bipolar mapping)
    """
    mode = int(vis_abs) if isinstance(vis_abs, (int, np.integer, bool)) else 0
    if mode == 1:
        return np.abs(channel)
    if mode == 2:
        return np.abs(channel - 0.5) * 2.0
    return channel


def _visible_mask(val: np.ndarray, renderer_mode: str,
                  voxel_threshold: float, iso_threshold: float,
                  vis_range: tuple[float, float]) -> np.ndarray:
    """Return boolean mask of voxels the GPU renderer would draw."""
    if renderer_mode == 'voxel':
        return val > voxel_threshold
    if renderer_mode == 'iso':
        return val > iso_threshold
    # volumetric (default): normalised value above accumulation floor.
    lo, hi = float(vis_range[0]), float(vis_range[1])
    span = hi - lo if hi > lo else 1.0
    normalised = (val - lo) / span
    return normalised > _VOLUMETRIC_FLOOR


def _probe_rule(ctx, rule: str, size: int, seed: int,
                default_cap: int) -> dict:
    from simulator import _resolve_composed_preset
    from test_harness import HeadlessRunner

    try:
        preset = _resolve_composed_preset(rule)
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'grade': 'err',
                'reason': f'resolve: {type(e).__name__}: {e}'}

    if preset.get('kind') == 'viewport':
        return {'rule': rule, 'grade': 'skip',
                'reason': 'viewport kind — no voxel-grid render path'}

    renderer_mode = (preset.get('render_mode') or 'volumetric').lower()
    if renderer_mode not in ('voxel', 'iso', 'volumetric'):
        return {'rule': rule, 'grade': 'skip',
                'reason': f'unknown render_mode {renderer_mode!r}'}

    vis_mode = (preset.get('vis_mode') or 'density').lower().strip()
    if vis_mode not in _DIRECT_VIS_MODES:
        return {'rule': rule, 'grade': 'skip',
                'reason': f'composite vis_mode {vis_mode!r}; mask logic '
                          f'mixes channels — manual review'}

    # Honour preset's declared minimum world size (a few rules — e.g.
    # lenia_multi at default_size=80 — explicitly die below it).
    default_size = int(preset.get('default_size') or 0)
    if default_size > size:
        size = default_size

    vis_default = int(preset.get('vis_default', 0))
    vis_abs = preset.get('vis_abs', False)
    voxel_threshold = float(preset.get('voxel_threshold', 0.5))
    iso_threshold = float(preset.get('iso_threshold', 0.5))
    vis_range = preset.get('vis_range', (0.0, 1.0))

    audit_steps = int(preset.get('audit_steps') or default_cap)
    checkpoint_steps = sorted({max(1, int(audit_steps * f))
                               for f in _CHECKPOINTS})

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = HeadlessRunner(ctx, rule, size=size, seed=seed)
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'grade': 'err',
                'reason': f'construct: {type(e).__name__}: {e}',
                'tb': traceback.format_exc().splitlines()[-3:]}

    samples: list[dict] = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            stepped = 0
            for target in checkpoint_steps:
                for _ in range(target - stepped):
                    r.step()
                stepped = target
                grid = _read_main(r)
                if grid.ndim != 4:
                    samples.append({'step': stepped, 'error':
                                    f'unexpected grid shape {grid.shape}'})
                    continue
                n_ch = grid.shape[-1]
                if not (0 <= vis_default < n_ch):
                    samples.append({'step': stepped, 'error':
                                    f'vis_default={vis_default} '
                                    f'out of range [0,{n_ch})'})
                    continue
                ch = grid[..., vis_default]
                if not np.all(np.isfinite(ch)):
                    samples.append({'step': stepped, 'error': 'NaN/Inf'})
                    continue
                val = _vis_value(ch, vis_abs)
                mask = _visible_mask(val, renderer_mode, voxel_threshold,
                                     iso_threshold, vis_range)
                visible = int(mask.sum())
                total = int(mask.size)
                samples.append({
                    'step': stepped,
                    'visible': visible,
                    'total': total,
                    'fraction': visible / max(1, total),
                })
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'grade': 'err',
                'reason': f'step: {type(e).__name__}: {e}',
                'tb': traceback.format_exc().splitlines()[-3:],
                'samples': samples}
    finally:
        try: r.release()
        except Exception: pass  # noqa: BLE001

    # Final-checkpoint signal range, for colormap-utilisation diagnostic.
    diag = None
    try:
        ch_final = grid[..., vis_default]
        finite = ch_final[np.isfinite(ch_final)]
        if finite.size:
            p1 = float(np.percentile(finite, 1))
            p99 = float(np.percentile(finite, 99))
            lo, hi = float(vis_range[0]), float(vis_range[1])
            note = None
            if p99 < lo:
                note = f'dim: p99={p99:.3g} below vis_range[0]={lo}'
            elif p1 > hi:
                note = f'saturated: p1={p1:.3g} above vis_range[1]={hi}'
            elif (p99 - p1) < 0.05 * max(1e-9, hi - lo):
                note = (f'narrow: signal span {p99 - p1:.3g} '
                        f'<5% of vis_range span {hi - lo:.3g}')
            diag = {'p1': p1, 'p99': p99, 'vis_range': [lo, hi],
                    'note': note}
    except Exception:  # noqa: BLE001
        pass

    # Grade.
    valid = [s for s in samples if 'fraction' in s]
    if not valid:
        return {'rule': rule, 'grade': 'err',
                'reason': 'no valid checkpoints sampled',
                'samples': samples,
                'config': {'render_mode': renderer_mode,
                           'vis_default': vis_default,
                           'voxel_threshold': voxel_threshold,
                           'iso_threshold': iso_threshold,
                           'vis_range': list(vis_range),
                           'vis_abs': vis_abs}}

    final = valid[-1]
    mid = valid[len(valid) // 2]
    grade = 'ok'
    reason = ''
    if final['visible'] == 0:
        grade = 'crit'
        reason = (f'zero visible voxels through {final["step"]} steps '
                  f'(render_mode={renderer_mode!r}, '
                  f'vis_default=[{vis_default}], '
                  f'thresh={voxel_threshold if renderer_mode == "voxel" else iso_threshold})')
    elif all(s['visible'] == 0 for s in valid[:-1]) and final['visible'] > 0:
        # Only last checkpoint has anything visible.
        grade = 'high'
        reason = (f'visible only after step {final["step"]}; '
                  f'first {len(valid) - 1}/{len(valid)} checkpoints empty')
    elif final['fraction'] < _MED_FRACTION and mid['visible'] == 0:
        grade = 'med'
        reason = (f'only {final["visible"]} voxels visible at step '
                  f'{final["step"]} ({final["fraction"] * 100:.3g}% of grid)')

    return {'rule': rule, 'grade': grade, 'reason': reason,
            'samples': samples, 'diag': diag,
            'config': {'render_mode': renderer_mode,
                       'vis_default': vis_default,
                       'voxel_threshold': voxel_threshold,
                       'iso_threshold': iso_threshold,
                       'vis_range': list(vis_range),
                       'vis_abs': vis_abs}}


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
    ap.add_argument('--size', type=int, default=48,
                    help='Grid size (default: 48 — matches discovery catalog).')
    ap.add_argument('--seed', type=int, default=1001)
    ap.add_argument('--cap', type=int, default=200,
                    help='Default audit horizon when preset has no audit_steps.')
    ap.add_argument('--skip-flagship', action='store_true')
    ap.add_argument('--skip', help='Comma-separated rules to omit.')
    ap.add_argument('--severity', choices=list(_SEV_ORDER.keys()), default='med',
                    help='Min severity to print (default: med).')
    ap.add_argument('--json', help='Write per-rule report JSON.')
    args = ap.parse_args(argv)

    from test_harness import create_headless_context
    _window, ctx = create_headless_context()

    rules = _select_rules(args)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, rule in enumerate(rules, 1):
        sys.stdout.write(f"\r[{i:>3}/{len(rules)}] {rule:<42}")
        sys.stdout.flush()
        rows.append(_probe_rule(ctx, rule, args.size, args.seed, args.cap))
    sys.stdout.write('\r' + ' ' * 70 + '\r')
    elapsed = time.perf_counter() - t0

    counts: dict = {k: 0 for k in _SEV_ORDER}
    for row in rows:
        counts[row['grade']] = counts.get(row['grade'], 0) + 1

    print(f'\nvisibility audit — {len(rules)} rules in {elapsed:.1f}s')
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
            cfg = row.get('config') or {}
            if cfg:
                rm = cfg["render_mode"]
                thresh_key = 'voxel_threshold' if rm == 'voxel' else 'iso_threshold'
                print(f'          mode={rm}  vis[{cfg["vis_default"]}]  '
                      f'{thresh_key}={cfg[thresh_key]}  '
                      f'vis_range={cfg["vis_range"]}  '
                      f'vis_abs={cfg["vis_abs"]}')
            for s in row.get('samples') or []:
                if 'fraction' in s:
                    print(f'          step {s["step"]:>5}:  '
                          f'{s["visible"]:>7} / {s["total"]} '
                          f'({s["fraction"] * 100:.3g}%)')
                else:
                    print(f'          step {s["step"]:>5}:  {s.get("error", "?")}')
            diag = row.get('diag')
            if diag and diag.get('note'):
                print(f'          diag: {diag["note"]}')

    # Surface colormap-utilisation notes even for 'ok' rules.
    util_notes = [r for r in rows
                  if r.get('diag') and r['diag'].get('note')
                  and r['grade'] == 'ok']
    if util_notes:
        print(f'\ncolormap-utilisation notes ({len(util_notes)} ok rules):')
        for row in util_notes[:30]:
            print(f'  {row["rule"]:<40}  {row["diag"]["note"]}')
        if len(util_notes) > 30:
            print(f'  ... and {len(util_notes) - 30} more (use --json for all)')

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'counts': counts, 'rows': rows,
                       'elapsed_s': elapsed,
                       'size': args.size, 'cap': args.cap},
                      fh, indent=2, default=str)
        print(f'\nwrote {args.json}')

    return 1 if (counts.get('err', 0) + counts.get('crit', 0)) else 0


if __name__ == '__main__':
    sys.exit(main())
