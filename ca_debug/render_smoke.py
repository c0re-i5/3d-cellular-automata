"""Probe #27 — render-side framebuffer regression guard.

Every other probe stops at the *simulation state*: ``golden_snapshots`` hashes
raw voxel bytes, ``visibility_audit`` inspects a sim-state occupancy mask, the
oracles compare physical fields.  Nothing reads the actual GL framebuffer, so
the whole ``sim-state -> pixels`` path (view-texture bake, camera, raymarch /
voxel-raster / SDF dispatch, colormap, brightness) was unguarded.  A black
screen, a mis-bound view texture, or a colormap that maps everything to
background is invisible to the rest of the suite even while the sim is healthy.
(The SDF spectral/diverging-as-grayscale bug fixed this session would have been
caught here.)

Why a *baseline*, not absolute thresholds
-----------------------------------------
"Is this frame visible enough?" is hopelessly rule-dependent: many rules are
near-background at a quick smoke (slow developers like ``bz_spiral_waves``,
transients like ``fire``, spatially-uniform early fields like ``brusselator_3d``)
yet render beautifully once evolved.  Absolute cutoffs flag all of those as
false positives.  So this probe mirrors ``golden_snapshots``: it renders every
rule headless at a fixed (size, steps, seed, resolution), computes a compact
render fingerprint, and on a known-good build *blesses* it to
``ca_debug/render_baseline.json``.  Later runs re-render and flag only rules
whose fingerprint *regresses* — a rule that was visibly rendering and is now
blank, a large structural change in the frame, or a render that now errors.

Fingerprint (per rule)
----------------------
  px      fraction of pixels differing from the modal background colour
  lum     mean frame luminance
  grid    8x8 block-averaged luminance (0-255) — coarse structural signature
  mode    renderer_mode at render time

Comparison flags a regression when a previously-visible rule goes blank
(px collapses), when the 8x8 grid drifts beyond tolerance, or on render error.

Usage::

    python -m ca_debug.render_smoke --bless            # capture baseline (good build)
    python -m ca_debug.render_smoke                     # check against baseline
    python -m ca_debug.render_smoke --rules lenia_3d,fire --png /tmp/shots
    python -m ca_debug.render_smoke --bless --rules mandelbulb_3d --note "..."
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import traceback
import zlib

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'render_baseline.json')

# Severity buckets (lower index = more severe), matching sibling probes.
_SEV_ORDER = {'err': 0, 'crit': 1, 'high': 2, 'new': 3, 'ok': 4}

# Fixed render config baked into the baseline.  Changing any of these
# invalidates the stored fingerprints (re-bless required).
_GRID = 8  # NxN luminance signature


def _write_png(path: str, img: np.ndarray) -> None:
    """Write an HxWx3 uint8 array as a PNG using only the stdlib."""
    h, w, _ = img.shape
    rows = b''.join(b'\x00' + img[y].tobytes() for y in range(h))

    def _chunk(typ: bytes, data: bytes) -> bytes:
        body = typ + data
        return (struct.pack('>I', len(data)) + body
                + struct.pack('>I', zlib.crc32(body) & 0xffffffff))

    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)))
        f.write(_chunk(b'IDAT', zlib.compress(rows, 6)))
        f.write(_chunk(b'IEND', b''))


def _fingerprint(img: np.ndarray) -> dict:
    """Compact render signature from an HxWx3 uint8 frame."""
    lum = img.mean(axis=2)
    bg = int(np.bincount(lum.astype(np.int64).ravel(), minlength=1).argmax())
    px = float((np.abs(lum - bg) > 12).mean())
    # Block-averaged luminance grid (robust to small pixel jitter).
    h, w = lum.shape
    gy, gx = h // _GRID, w // _GRID
    grid = lum[:gy * _GRID, :gx * _GRID].reshape(
        _GRID, gy, _GRID, gx).mean(axis=(1, 3))
    return {
        'px': px,
        'lum': float(lum.mean()),
        'grid': [int(round(v)) for v in grid.ravel()],
    }


def _render_one(simulator, rule: str, args) -> dict:
    """Build, step, and render one rule headless.  Returns a fingerprint row."""
    import glfw

    sim = None
    try:
        sim = simulator.Simulator(size=args.size, rule=rule, headless=True)
        sim.seed = args.seed
        sim._reset()
        for _ in range(args.steps):
            sim._step_sim()
        sim._rec_width, sim._rec_height = args.res, args.res
        sim._init_rec_fbo()
        sim._render_to_rec_fbo()
        raw = sim._rec_fbo.read(components=3)
        img = np.frombuffer(raw, dtype=np.uint8).reshape(args.res, args.res, 3)
        mode = sim.renderer_mode

        fp = _fingerprint(img)
        fp['mode'] = mode
        fp['rule'] = rule
        fp['error'] = None

        if args.png:
            os.makedirs(args.png, exist_ok=True)
            _write_png(os.path.join(args.png, f'{rule}.png'), img[::-1].copy())
        return fp
    except Exception as exc:  # noqa: BLE001 — any failure is a finding
        return {
            'rule': rule, 'mode': '?', 'px': 0.0, 'lum': 0.0, 'grid': [],
            'error': f'{type(exc).__name__}: {exc}',
            'trace': traceback.format_exc(),
        }
    finally:
        if sim is not None:
            try:
                sim.ctx.release()
            except Exception:  # noqa: BLE001
                pass
            win = getattr(sim, 'window', None)
            if win is not None:
                try:
                    glfw.destroy_window(win)
                except Exception:  # noqa: BLE001
                    pass


def _grid_dist(a: list[int], b: list[int]) -> float:
    """Mean per-cell luminance distance in [0,1]; 1.0 if shapes differ."""
    if not a or not b or len(a) != len(b):
        return 1.0
    aa = np.asarray(a, dtype=np.float32) / 255.0
    bb = np.asarray(b, dtype=np.float32) / 255.0
    return float(np.abs(aa - bb).mean())


def _compare(cur: dict, base: dict, args) -> tuple[str, str]:
    """Grade a current fingerprint against its baseline.  Returns (sev, note)."""
    if cur['error'] is not None:
        return 'err', cur['error']
    if base is None:
        return 'new', 'no baseline (run --bless)'

    gd = _grid_dist(cur.get('grid', []), base.get('grid', []))
    base_px, cur_px = base['px'], cur['px']

    # Regression: a rule that was clearly rendering is now (near) blank.
    if base_px >= args.visible and cur_px < base_px * args.collapse:
        return 'crit', f'px {base_px:.3f}->{cur_px:.3f} (rendered, now blank)'
    if cur['mode'] != base.get('mode'):
        return 'high', f"mode {base.get('mode')}->{cur['mode']}"
    if gd >= args.grid_crit:
        return 'crit', f'grid drift {gd:.3f}'
    if gd >= args.grid_high:
        return 'high', f'grid drift {gd:.3f}'
    return 'ok', f'px={cur_px:.3f} gd={gd:.3f}'


def _select_rules(simulator, args, baseline: dict) -> list[str]:
    presets = simulator.RULE_PRESETS
    if args.rules:
        rules = [r.strip() for r in args.rules.split(',') if r.strip()]
    elif args.bless:
        rules = list(presets.keys())
    elif baseline:
        rules = list(baseline.get('rules', {}).keys())
    else:
        rules = list(presets.keys())
    if args.mode:
        rules = [r for r in rules
                 if r in presets
                 and presets[r].get('render_mode', 'volumetric') == args.mode]
    if args.skip:
        skip = {s.strip() for s in args.skip.split(',')}
        rules = [r for r in rules if r not in skip]
    unknown = [r for r in rules if r not in presets]
    if unknown:
        print(f"WARN unknown rules ignored: {', '.join(unknown)}")
        rules = [r for r in rules if r in presets]
    return rules


def _load_baseline() -> dict:
    if not os.path.exists(_BASELINE_PATH):
        return {}
    with open(_BASELINE_PATH) as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bless', action='store_true',
                    help='Capture/refresh the render baseline instead of checking.')
    ap.add_argument('--rules', help='Comma-separated rule list (default: all baselined rules).')
    ap.add_argument('--mode', choices=('voxel', 'volumetric', 'sdf_viewport'),
                    help='Only test presets with this render_mode.')
    ap.add_argument('--size', type=int, default=48)
    ap.add_argument('--steps', type=int, default=16)
    ap.add_argument('--res', type=int, default=192)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--skip', help='Comma-separated rules to omit.')
    ap.add_argument('--collapse', type=float, default=0.4,
                    help='crit if cur_px drops below this fraction of baseline px (default: 0.4).')
    ap.add_argument('--visible', type=float, default=2e-2,
                    help='Baseline px above which a rule counts as "was rendering" (default: 2e-2).')
    ap.add_argument('--grid-high', type=float, default=0.06, dest='grid_high',
                    help='8x8 luminance drift (0-1) flagged high (default: 0.06).')
    ap.add_argument('--grid-crit', type=float, default=0.12, dest='grid_crit',
                    help='8x8 luminance drift (0-1) flagged crit (default: 0.12).')
    ap.add_argument('--note', help='Note stored with --bless.')
    ap.add_argument('--png', help='Directory to dump one rendered PNG per rule.')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--json', help='Write the full per-rule report to this path.')
    args = ap.parse_args(argv)

    os.environ.setdefault('CA_DISABLE_PRESET_OVERRIDES', '1')
    import simulator

    baseline = _load_baseline()
    cfg = baseline.get('config') if baseline else None
    if not args.bless and baseline and cfg:
        mismatch = [k for k in ('size', 'steps', 'res', 'seed')
                    if cfg.get(k) != getattr(args, k)]
        if mismatch:
            print(f"WARN baseline was blessed with different {', '.join(mismatch)} "
                  f"-- comparison may be noisy. Using baseline config.")
            for k in ('size', 'steps', 'res', 'seed'):
                setattr(args, k, cfg.get(k, getattr(args, k)))

    base_rules = baseline.get('rules', {}) if baseline else {}
    rules = _select_rules(simulator, args, baseline)
    if not rules:
        print("render_smoke: no rules selected")
        return 1

    rows: list[dict] = []
    worst = 'ok'
    for i, rule in enumerate(rules, 1):
        sys.stdout.write(f"\r[{i:>3}/{len(rules)}] {rule:<36}")
        sys.stdout.flush()
        fp = _render_one(simulator, rule, args)
        if not args.bless:
            sev, note = _compare(fp, base_rules.get(rule), args)
            fp['sev'], fp['note'] = sev, note
            if _SEV_ORDER[sev] < _SEV_ORDER[worst]:
                worst = sev
        rows.append(fp)
    sys.stdout.write('\r' + ' ' * 52 + '\r')

    # -- Bless ---------------------------------------------------------------
    if args.bless:
        store = {r['rule']: {'px': r['px'], 'lum': r['lum'], 'grid': r['grid'],
                             'mode': r['mode']}
                 for r in rows if r['error'] is None}
        errs = [r for r in rows if r['error'] is not None]
        # Preserve rules not re-blessed this run (partial --rules bless).
        if args.rules and base_rules:
            merged = dict(base_rules)
            merged.update(store)
            store = merged
        out = {
            'config': {'size': args.size, 'steps': args.steps, 'res': args.res,
                       'seed': args.seed},
            'note': args.note or (baseline.get('note', '') if args.rules else ''),
            'rules': dict(sorted(store.items())),
        }
        with open(_BASELINE_PATH, 'w') as f:
            json.dump(out, f, indent=1)
        print(f"blessed {len(store)} rules -> {os.path.relpath(_BASELINE_PATH, _REPO_ROOT)}")
        if errs:
            print(f"  {len(errs)} rules errored and were NOT blessed:")
            for r in errs:
                print(f"    {r['rule']:<26} {r['error']}")
        return 1 if errs else 0

    # -- Check report --------------------------------------------------------
    print(f"render regression -- {len(rows)} rules vs baseline "
          f"(size={args.size} steps={args.steps} res={args.res})")
    if baseline.get('note'):
        print(f"  baseline note: {baseline['note']}")
    header = f"  {'rule':<26} {'mode':<12} {'verdict':<6}  detail"
    printed = False
    for row in sorted(rows, key=lambda r: (_SEV_ORDER[r['sev']], r['rule'])):
        if not args.verbose and row['sev'] == 'ok':
            continue
        if not printed:
            print(header)
            printed = True
        print(f"  {row['rule']:<26} {row['mode']:<12} {row['sev'].upper():<6}  {row['note']}")

    counts = {k: 0 for k in _SEV_ORDER}
    for row in rows:
        counts[row['sev']] += 1
    summary = ' '.join(f"{k}={counts[k]}" for k in ('ok', 'new', 'high', 'crit', 'err')
                       if counts[k])
    print(f"\nworst={worst}  {summary}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'worst': worst, 'counts': counts,
                       'rules': [{k: v for k, v in r.items() if k != 'trace'}
                                 for r in rows]}, f, indent=2)
        print(f"Wrote {args.json}")

    return 1 if (counts['crit'] + counts['high'] + counts['err']) else 0


if __name__ == '__main__':
    sys.exit(main())
