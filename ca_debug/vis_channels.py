"""vis-channel ↔ shader-output consistency probe.

Each preset declares ``vis_channels`` (a list of human-readable
labels) and ``vis_default`` (an index into that list selected on
first load).  In ``vis_mode='density'`` or ``vis_mode='bipolar'``
the integer ``u_vis_channel`` uniform directly selects one of the
four RGBA components of the main texture for rendering, so
``vis_channels[i]`` corresponds to texture channel ``i``.

When a preset declares N labels but the shader only writes some
of those channels, picking one of the silent labels yields a
black screen — a confusing UX bug that goes unnoticed until a
user happens to click that entry.

This probe checks:

  err   crash during construction or stepping.
  crit  ``vis_default`` is out of range for ``vis_channels``.
        Selecting it would IndexError in the GUI.
  high  in density/bipolar mode, a named ``vis_channels[i]``
        corresponds to a channel that is identically zero
        (std AND max-abs both < eps) across the entire grid
        after stepping -- selecting it renders an empty screen.
        Names equal to ``'_'`` (placeholder convention) are
        skipped: an underscore signals the slot is intentionally
        unused.
  med   the ``vis_default`` channel itself has very low signal
        (std < 1e-4) -- the GUI's first-load view is nearly
        blank, even though other channels carry the real state.
        Composite-mode rules where the entire grid is
        identically zero (max-abs == 0) also fall here -- the
        renderer will show a black screen.
  ok    vis_default in range, default channel has signal,
        all named channels in density/bipolar mode are populated.
  n/a   rule has no ``vis_channels`` declaration, or a vis_mode
        for which the index-to-channel mapping is not direct
        (rgb_channels / hsv_phase / rgba_blend).

Usage::

    python -m ca_debug.vis_channels
    python -m ca_debug.vis_channels --severity high
    python -m ca_debug.vis_channels --rules lenia_3d
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

# Modes where vis_channels[i] ↔ texture channel i one-to-one.
_DIRECT_MODES = {'density', 'bipolar', 'signed'}


def _read_main(runner) -> np.ndarray:
    return np.asarray(runner.read_grid()).copy()


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


def _probe_rule(ctx, rule: str, size: int, default_steps: int, seed: int) -> dict:
    from simulator import _resolve_composed_preset
    from test_harness import HeadlessRunner
    try:
        preset = _resolve_composed_preset(rule)
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'grade': 'err',
                'error': f'preset resolve: {type(e).__name__}: {e}'}

    vis_channels = preset.get('vis_channels')
    vis_default = preset.get('vis_default', 0)
    vis_mode = (preset.get('vis_mode') or 'density').lower().strip()
    # Honour per-preset audit_steps when set: a few rules have slow
    # physics (crystal growth, mycelium maturation) where auxiliary
    # channels only populate after thousands of steps.
    steps = int(preset.get('audit_steps') or default_steps)

    if not vis_channels:
        return {'rule': rule, 'grade': 'n/a',
                'reason': 'no vis_channels declared'}

    # Static check: vis_default in range.
    if not (0 <= int(vis_default) < len(vis_channels)):
        return {'rule': rule, 'grade': 'crit',
                'reason': (
                    f'vis_default={vis_default} out of range for '
                    f'{len(vis_channels)} channels'),
                'vis_channels': vis_channels}

    # Runtime check: step and inspect channels.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = HeadlessRunner(ctx, rule, size=size, seed=seed)
            for _ in range(steps):
                r.step()
            grid = _read_main(r)
            try: r.release()
            except Exception: pass  # noqa: BLE001
    except Exception as e:  # noqa: BLE001
        return {'rule': rule, 'grade': 'err',
                'error': f'{type(e).__name__}: {e}',
                'tb': traceback.format_exc()}

    if grid.ndim != 4 or grid.shape[-1] < 1:
        return {'rule': rule, 'grade': 'err',
                'reason': f'unexpected grid shape {grid.shape}'}
    n_ch = grid.shape[-1]

    # Per-channel std and max-abs across the whole grid.
    ch_std = [float(np.nanstd(grid[..., c])) for c in range(n_ch)]
    ch_max = [float(np.nanmax(np.abs(grid[..., c]))) for c in range(n_ch)]

    out: dict = {'rule': rule, 'vis_channels': vis_channels,
                 'vis_default': int(vis_default), 'vis_mode': vis_mode,
                 'ch_std': ch_std, 'ch_max': ch_max,
                 'size': size, 'steps': steps}

    # Only direct-mapping vis_modes can be checked per-named-channel.
    if vis_mode not in _DIRECT_MODES:
        # Still check the grid has SOMETHING to render -- in
        # composite modes vis_default might be a preset selector, but
        # we can at least check the grid isn't identically zero.
        # An identically-zero grid (max-abs == 0 across every channel)
        # is rendered as a black screen; a flat-uniform-nonzero grid
        # (saturated growth model) is not a bug.
        if max(ch_max) < 1e-6:
            out['grade'] = 'med'
            out['reason'] = (
                f'composite mode {vis_mode!r}: entire grid is '
                f'identically zero after {steps} steps '
                f'(max |ch|={max(ch_max):.2g})')
            return out
        out['grade'] = 'ok'
        out['reason'] = f'composite vis_mode {vis_mode!r}; channels not directly index-mapped'
        return out

    # Direct mode: vis_channels[i] -> grid[..., i].
    n_named = len(vis_channels)
    if n_named > n_ch:
        out['grade'] = 'high'
        out['reason'] = (
            f'{n_named} channel names declared but grid only has '
            f'{n_ch} channels')
        return out
    # Silent = both std AND max-abs near zero (a flat-uniform-nonzero
    # channel is boring but not buggy). Names equal to '_' are explicit
    # placeholders and are skipped. Requires at least one OTHER channel
    # to be lively (max ch_std > 1e-3) so we don't keep flagging
    # already-dead grids -- those belong to Probe #6.
    others_lively = max(ch_std) > 1e-3
    silent = [
        (i, vis_channels[i], ch_std[i])
        for i in range(n_named)
        if (vis_channels[i] or '').strip() not in ('', '_')
        and ch_std[i] < 1e-6
        and ch_max[i] < 1e-6
        and others_lively
    ]
    out['silent_named'] = silent
    # Default-channel signal check (separate from silent-named).
    default_std = ch_std[int(vis_default)]
    out['default_std'] = default_std
    if silent:
        out['grade'] = 'high'
        out['reason'] = (
            'silent named channel(s): '
            + ', '.join(f'[{i}] {nm!r} std={s:.2g}' for i, nm, s in silent))
        return out
    if default_std < 1e-4:
        out['grade'] = 'med'
        out['reason'] = (
            f'default channel [{vis_default}] {vis_channels[vis_default]!r} '
            f'has very low signal (std={default_std:.2g})')
        return out
    out['grade'] = 'ok'
    return out


def main(argv=None):
    os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

    ap = argparse.ArgumentParser()
    ap.add_argument('--rules', help='Comma-separated rule names (default: all).')
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--steps', type=int, default=30,
                    help='Default step count when preset has no audit_steps.')
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

    print(f"vis-channels probe (size={args.size}, steps={args.steps}, "
          f"seed={args.seed}) -- {elapsed:.1f}s")
    print(f"{'SEV':<6} {'RULE':<42}  {'MODE':<14}  NOTES")
    print('-' * 130)
    for r in rows_sorted:
        if _SEV_ORDER.get(r['grade'], 9) > min_sev:
            continue
        parts: list[str] = []
        if r.get('reason'):
            parts.append(r['reason'])
        if r.get('error'):
            parts.append(str(r['error'])[:80])
        note = '  '.join(parts) or '-'
        print(f"{r['grade']:<6} {r['rule']:<42}  "
              f"{str(r.get('vis_mode','?'))[:14]:<14}  {note}")
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
