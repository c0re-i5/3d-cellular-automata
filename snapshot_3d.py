"""3D voxel snapshot + analysis tool.

Captures the FULL 4-channel voxel state of the simulator at a chosen step
and writes it to disk as a compressed .npz. Then provides analytical
inspection that is NOT filtered through the renderer's 1-channel colormap
(which is what made distinct rules look identical in 2D strips).

Subcommands:
  capture  <rule> --step N [--out f.npz] [--size N] [--seed S]
  record   <rule> --steps 0,10,50,...   --out-dir DIR/  [--size N] [--seed S]
  inspect  <file.npz>
  diff     <a.npz> <b.npz>
  compare  <a.npz> <b.npz> [<c.npz> ...]
  audit-channels [--size N] [--steps N] [--filter PAT] [--csv FILE]

File format (.npz):
  voxels : (W, H, D, 4) float16   — full RGBA channels, never collapsed
  series : (T, 4*K)    float32    — per-step scalar timeline (record only)
  meta   : 0-d object array with dict: rule, step, seed, dims, init,
                                       preset_keys, params, dtype, version
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

VERSION = 1


# ───────────────────────────── capture ─────────────────────────────

def _build_sim(rule: str, size: int, seed: int):
    os.environ.setdefault('CA_DISABLE_PRESET_OVERRIDES', '1')
    import simulator as S
    sim = S.Simulator(size=size, rule=rule, headless=True)
    sim.seed = seed
    sim._reset()
    return sim


def _read_voxels(sim) -> np.ndarray:
    """Return (W, H, D, 4) array of the current sim state (float16)."""
    src = sim.tex_a if sim.ping == 0 else sim.tex_b
    raw = src.read()
    arr = np.frombuffer(raw, dtype=sim._tex_np_dtype).reshape(
        sim.D, sim.H, sim.W, 4)  # texture3D layout: depth-major
    # Move to (W, H, D, 4) for friendlier indexing
    arr = arr.transpose(2, 1, 0, 3)
    return arr.astype(np.float16, copy=True)


def _meta_dict(sim, rule: str, step: int, seed: int) -> dict:
    p = sim.preset
    return {
        'version': VERSION,
        'rule': rule,
        'step': step,
        'seed': seed,
        'dims': [int(sim.W), int(sim.H), int(sim.D)],
        'dtype_native': str(sim._tex_np_dtype),
        'init': sim._current_init if hasattr(sim, '_current_init') else None,
        'preset_keys': sorted(list(p.keys())) if isinstance(p, dict) else [],
        'params': {k: v for k, v in (p.get('params', {}) or {}).items()
                   if isinstance(v, (int, float, str, bool, list, tuple))},
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }


def _save_npz(path: Path, voxels: np.ndarray, meta: dict, series=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'voxels': voxels,
        'meta': np.array(json.dumps(meta), dtype=object),
    }
    if series is not None:
        payload['series'] = series['data']
        payload['series_cols'] = np.array(series['cols'], dtype=object)
        payload['series_steps'] = np.asarray(series['steps'], dtype=np.int64)
    np.savez_compressed(path, **payload)


def _load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    voxels = z['voxels']
    meta = json.loads(str(z['meta'].item()))
    series = None
    if 'series' in z.files:
        series = {
            'data': z['series'],
            'cols': list(z['series_cols']),
            'steps': z['series_steps'],
        }
    return voxels, meta, series


def _per_step_scalars(arr: np.ndarray) -> list[float]:
    """Compact per-channel scalar fingerprint for timeline series."""
    out = []
    a32 = arr.astype(np.float32)
    for c in range(4):
        x = a32[..., c]
        out.extend([
            float(x.mean()),
            float(x.std()),
            float((x > 0.05).sum() / x.size),  # alive fraction
            float(x.max() - x.min()),
        ])
    return out


SERIES_COLS = [f'ch{c}_{stat}' for c in range(4)
               for stat in ('mean', 'std', 'alive', 'range')]


def cmd_capture(args):
    sim = _build_sim(args.rule, args.size, args.seed)
    for _ in range(args.step):
        sim._step_sim()
    sim.step_count = args.step
    voxels = _read_voxels(sim)
    meta = _meta_dict(sim, args.rule, args.step, args.seed)
    out = Path(args.out) if args.out else Path('snapshots') / \
        f'{args.rule}_s{args.size}_t{args.step}_seed{args.seed}.npz'
    _save_npz(out, voxels, meta)
    sim.ctx.release()
    print(f'wrote {out}  shape={tuple(voxels.shape)}  '
          f'dtype={voxels.dtype}  size={out.stat().st_size/1e6:.2f} MB')


def cmd_record(args):
    steps = sorted({int(s) for s in args.steps.split(',') if s.strip()})
    if not steps:
        sys.exit('--steps requires at least one integer')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = _build_sim(args.rule, args.size, args.seed)
    series_data: list[list[float]] = []
    series_steps: list[int] = []

    max_step = steps[-1]
    snap_set = set(steps)

    # capture step 0 if requested
    cur = 0
    if cur in snap_set:
        v = _read_voxels(sim)
        meta = _meta_dict(sim, args.rule, cur, args.seed)
        _save_npz(out_dir / f'{args.rule}_t{cur:06d}.npz', v, meta)
    series_data.append(_per_step_scalars(_read_voxels(sim).astype(np.float32)))
    series_steps.append(cur)

    while cur < max_step:
        sim._step_sim()
        cur += 1
        sim.step_count = cur
        v = _read_voxels(sim)
        # per-step series is cheap on small grids; on big grids skip non-snapshot steps
        if args.size <= 64 or cur in snap_set:
            series_data.append(_per_step_scalars(v.astype(np.float32)))
            series_steps.append(cur)
        if cur in snap_set:
            meta = _meta_dict(sim, args.rule, cur, args.seed)
            _save_npz(out_dir / f'{args.rule}_t{cur:06d}.npz', v, meta)
            print(f'  captured t={cur:6d}  alive_ch0={(v[...,0]>0.05).mean():.4f}  '
                  f'mean_ch0={v[...,0].astype(np.float32).mean():.4f}')

    # write timeline series alongside snapshots
    series = {
        'data': np.asarray(series_data, dtype=np.float32),
        'cols': SERIES_COLS,
        'steps': series_steps,
    }
    np.savez_compressed(out_dir / f'{args.rule}_series.npz',
                        data=series['data'],
                        cols=np.array(series['cols'], dtype=object),
                        steps=np.asarray(series['steps'], dtype=np.int64),
                        meta=np.array(json.dumps({
                            'version': VERSION, 'rule': args.rule,
                            'seed': args.seed, 'size': args.size,
                            'dims': [int(sim.W), int(sim.H), int(sim.D)],
                            'snapshot_steps': steps,
                        }), dtype=object))
    sim.ctx.release()
    print(f'\nwrote {len(steps)} snapshots + series to {out_dir}/')


# ───────────────────────────── inspect ─────────────────────────────

def _hist_ascii(x: np.ndarray, bins: int = 16, width: int = 40) -> str:
    """One-line ASCII histogram bar block."""
    if x.size == 0 or x.std() == 0:
        return '(flat)'
    h, edges = np.histogram(x, bins=bins, range=(float(x.min()), float(x.max())))
    h = h / h.max()
    blocks = ' ▁▂▃▄▅▆▇█'
    return ''.join(blocks[min(8, int(v * 8))] for v in h)


def _channel_stats(c: np.ndarray) -> dict:
    c32 = c.astype(np.float32)
    finite = np.isfinite(c32)
    if not finite.all():
        c32 = np.where(finite, c32, 0)
    pcts = np.percentile(c32, [1, 50, 99])
    return {
        'min': float(c32.min()),
        'max': float(c32.max()),
        'mean': float(c32.mean()),
        'std': float(c32.std()),
        'p1': float(pcts[0]),
        'p50': float(pcts[1]),
        'p99': float(pcts[2]),
        'alive_frac': float((c32 > 0.05).mean()),
        'sat_frac': float((c32 > 0.95).mean()),
        'zero_frac': float((np.abs(c32) < 1e-4).mean()),
        'unique_quantized': int(np.unique(np.round(c32 * 100).astype(np.int32)).size),
    }


def _gradient_stats(ch: np.ndarray) -> dict:
    a = ch.astype(np.float32)
    gx = np.diff(a, axis=0)
    gy = np.diff(a, axis=1)
    gz = np.diff(a, axis=2)
    # pad to common shape via abs-mean
    return {
        'grad_mean': float((np.abs(gx).mean() + np.abs(gy).mean() + np.abs(gz).mean()) / 3),
        'grad_max': float(max(np.abs(gx).max(), np.abs(gy).max(), np.abs(gz).max())),
    }


def _connected_components_count(mask: np.ndarray) -> int:
    """Cheap 6-connectivity component counter via scipy if available, else flood-fill skip."""
    try:
        from scipy.ndimage import label
        _, n = label(mask)
        return int(n)
    except ImportError:
        return -1  # signal not available


def _radial_profile(ch: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Mean of |ch| as a function of distance from grid center."""
    a = np.abs(ch.astype(np.float32))
    W, H, D = a.shape
    cz, cy, cx = (D - 1) / 2, (H - 1) / 2, (W - 1) / 2
    z, y, x = np.indices(a.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
    rmax = r.max()
    bins = np.linspace(0, rmax, n_bins + 1)
    out = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        m = (r >= bins[i]) & (r < bins[i + 1])
        if m.any():
            out[i] = a[m].mean()
    return out


def _fft_top_wavelengths(ch: np.ndarray, k: int = 3) -> list[tuple[float, float]]:
    """Return k strongest spatial frequencies as (wavelength_in_voxels, magnitude)."""
    a = ch.astype(np.float32)
    a = a - a.mean()
    if a.std() < 1e-8:
        return []
    F = np.fft.fftn(a)
    P = np.abs(F)
    P_flat = P.flatten()
    # zero out DC (already removed by mean-sub) and very-low freqs
    P_flat[0] = 0
    idx = np.argpartition(P_flat, -k)[-k:]
    idx = idx[np.argsort(-P_flat[idx])]
    W, H, D = a.shape
    out = []
    for i in idx:
        kz, ky, kx = np.unravel_index(i, a.shape)
        # fold to [-N/2, N/2]
        kx = kx if kx <= W // 2 else kx - W
        ky = ky if ky <= H // 2 else ky - H
        kz = kz if kz <= D // 2 else kz - D
        kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
        wavelength = float(min(W, H, D) / kmag) if kmag > 0 else float('inf')
        out.append((wavelength, float(P_flat[i])))
    return out


def _verdict(stats4: list[dict], xch_corr: np.ndarray) -> str:
    """One-line verdict from per-channel stats and channel cross-correlation."""
    alive_ch0 = stats4[0]['alive_frac']
    sat_ch0 = stats4[0]['sat_frac']
    used = [i for i, s in enumerate(stats4) if s['std'] > 1e-4]
    if alive_ch0 < 0.001:
        return 'DEAD (ch0 alive < 0.1%)'
    if sat_ch0 > 0.95:
        return 'SATURATED (ch0 sat > 95%)'
    if len(used) <= 1:
        return f'SINGLE_CHANNEL (only ch{used[0] if used else 0} has variance)'
    # check redundancy
    high_corr = []
    for i in used:
        for j in used:
            if j > i and abs(xch_corr[i, j]) > 0.95:
                high_corr.append((i, j))
    if high_corr and len(used) - len(high_corr) <= 1:
        return f'REDUNDANT_CHANNELS (corr>0.95 between {high_corr})'
    if all(s['std'] < 0.02 for s in stats4):
        return 'WEAK (all channels std < 0.02)'
    return f'STRUCTURED (ch_used={used})'


def _channel_corr(arr: np.ndarray) -> np.ndarray:
    """4x4 Pearson correlation between channels."""
    flat = arr.astype(np.float32).reshape(-1, 4).T  # (4, N)
    c = np.zeros((4, 4), dtype=np.float32)
    stds = flat.std(axis=1)
    for i in range(4):
        for j in range(4):
            if stds[i] < 1e-8 or stds[j] < 1e-8:
                c[i, j] = float('nan')
            else:
                c[i, j] = float(np.corrcoef(flat[i], flat[j])[0, 1])
    return c


def _print_inspection(voxels: np.ndarray, meta: dict, series=None):
    print(f"\n=== {meta.get('rule')} @ step {meta.get('step')} (seed {meta.get('seed')}) ===")
    print(f"dims={meta.get('dims')}  init={meta.get('init')!r}  "
          f"native_dtype={meta.get('dtype_native')}  ts={meta.get('timestamp')}")
    print(f"voxels: shape={tuple(voxels.shape)}  dtype={voxels.dtype}  "
          f"mem={voxels.nbytes/1e6:.1f} MB")

    stats = []
    print(f"\n  {'ch':<3} {'min':>8} {'max':>8} {'mean':>8} {'std':>8} "
          f"{'p1':>7} {'p50':>7} {'p99':>7} {'alive%':>7} {'sat%':>6} "
          f"{'zero%':>6} {'uniq':>6}  hist")
    for c in range(4):
        s = _channel_stats(voxels[..., c])
        stats.append(s)
        h = _hist_ascii(voxels[..., c].astype(np.float32))
        print(f"  {c:<3d} {s['min']:>8.3f} {s['max']:>8.3f} {s['mean']:>8.3f} "
              f"{s['std']:>8.3f} {s['p1']:>7.3f} {s['p50']:>7.3f} {s['p99']:>7.3f} "
              f"{s['alive_frac']*100:>6.2f}% {s['sat_frac']*100:>5.2f}% "
              f"{s['zero_frac']*100:>5.2f}% {s['unique_quantized']:>6d}  {h}")

    print("\nchannel cross-correlation (Pearson):")
    xc = _channel_corr(voxels)
    print("       " + "  ".join(f'{i:>7}' for i in range(4)))
    for i in range(4):
        print(f"  ch{i}  " + "  ".join(
            ('   nan ' if np.isnan(xc[i, j]) else f'{xc[i,j]:>7.3f}')
            for j in range(4)))

    print("\nspatial gradient (per channel):")
    for c in range(4):
        g = _gradient_stats(voxels[..., c])
        print(f"  ch{c}: grad_mean={g['grad_mean']:.4f}  grad_max={g['grad_max']:.4f}")

    print("\nconnected components (ch0 > 0.05, 6-connectivity):")
    cc = _connected_components_count(voxels[..., 0].astype(np.float32) > 0.05)
    if cc < 0:
        print("  (scipy not installed — skip)")
    else:
        print(f"  {cc} components")

    print("\nradial profile from grid center (ch0, mean |v|):")
    rp = _radial_profile(voxels[..., 0])
    rp_norm = rp / max(rp.max(), 1e-8)
    print("  " + _hist_ascii(rp, bins=len(rp)) + f"   max={rp.max():.4f}")
    print("  " + " ".join(f'{v:.3f}' for v in rp_norm))

    print("\ntop-3 spatial wavelengths (FFT, voxels):")
    for c in range(4):
        if stats[c]['std'] < 1e-4:
            print(f"  ch{c}: (flat)")
            continue
        tops = _fft_top_wavelengths(voxels[..., c], k=3)
        s = "  ".join(f'λ={w:.1f}vx (P={p:.0f})' for w, p in tops)
        print(f"  ch{c}: {s}")

    print(f"\nVERDICT: {_verdict(stats, xc)}")

    if series is not None:
        print(f"\ntimeline series: {series['data'].shape[0]} samples, "
              f"{len(series['cols'])} cols")
        # show ch0 alive trace
        if 'ch0_alive' in series['cols']:
            idx = series['cols'].index('ch0_alive')
            tr = series['data'][:, idx]
            print(f"  ch0_alive: min={tr.min():.4f}  max={tr.max():.4f}  "
                  f"final={tr[-1]:.4f}  trace: {_hist_ascii(tr, bins=24)}")


def cmd_inspect(args):
    voxels, meta, series = _load_npz(Path(args.file))
    _print_inspection(voxels, meta, series)


# ───────────────────────────── diff / compare ─────────────────────────────

def _pair_report(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str):
    if a.shape != b.shape:
        print(f'  shape mismatch {a.shape} vs {b.shape} — skip')
        return
    a32 = a.astype(np.float32); b32 = b.astype(np.float32)
    print(f"\n  {name_a}  vs  {name_b}")
    print(f"  {'ch':<3} {'L2':>10} {'Linf':>10} {'corr':>8} {'mean_a':>8} {'mean_b':>8}")
    for c in range(4):
        d = a32[..., c] - b32[..., c]
        l2 = float(np.sqrt((d * d).mean()))
        linf = float(np.abs(d).max())
        sa, sb = a32[..., c].std(), b32[..., c].std()
        corr = float('nan') if sa < 1e-8 or sb < 1e-8 else \
            float(np.corrcoef(a32[..., c].flatten(), b32[..., c].flatten())[0, 1])
        print(f"  {c:<3d} {l2:>10.5f} {linf:>10.5f} {corr:>8.3f} "
              f"{a32[...,c].mean():>8.3f} {b32[...,c].mean():>8.3f}")


def cmd_diff(args):
    va, ma, _ = _load_npz(Path(args.a))
    vb, mb, _ = _load_npz(Path(args.b))
    print(f"a: {ma.get('rule')} t={ma.get('step')} dims={ma.get('dims')}")
    print(f"b: {mb.get('rule')} t={mb.get('step')} dims={mb.get('dims')}")
    _pair_report(va, vb, ma.get('rule', 'a'), mb.get('rule', 'b'))


# ───────────────────────────── audit-channels ─────────────────────────────

def _audit_one(rule: str, size: int, steps: int, seed: int):
    """Returns (used_count, std_per_ch, xch_corr, verdict, error_or_none)."""
    try:
        sim = _build_sim(rule, size, seed)
    except Exception as e:
        return None, None, None, None, f'build: {e}'
    try:
        for _ in range(steps):
            sim._step_sim()
        sim.step_count = steps
        try:
            vox = _read_voxels(sim)
        except Exception as e:
            return None, None, None, None, f'read: {e}'
    finally:
        try:
            sim.ctx.release()
        except Exception:
            pass
    a32 = vox.astype(np.float32)
    stds = [float(a32[..., c].std()) for c in range(4)]
    xc = _channel_corr(vox)
    stats = [_channel_stats(vox[..., c]) for c in range(4)]
    used = sum(1 for s in stds if s > 1e-4)
    verdict = _verdict(stats, xc)
    return used, stds, xc, verdict, None


def cmd_audit_channels(args):
    """Audit channel usage across all leaf rules."""
    os.environ.setdefault('CA_DISABLE_PRESET_OVERRIDES', '1')
    import simulator as S
    import re
    pat = re.compile(args.filter) if args.filter else None
    rules = []
    for name in sorted(S.RULE_PRESETS.keys()):
        if name.startswith('flagship_'):
            continue
        preset = S.RULE_PRESETS[name]
        if not isinstance(preset, dict):
            continue
        init = preset.get('init', '') or ''
        if isinstance(init, str) and init.startswith('compose:'):
            continue
        if pat and not pat.search(name):
            continue
        rules.append(name)

    print(f'Auditing {len(rules)} rules at size={args.size}, steps={args.steps}, '
          f'seed={args.seed}...', flush=True)
    t0 = time.time()
    rows = []
    for i, rule in enumerate(rules):
        used, stds, xc, verdict, err = _audit_one(rule, args.size, args.steps, args.seed)
        if err:
            print(f'  [{i+1:3d}/{len(rules)}] {rule:32s} ERR  {err}', flush=True, file=sys.stderr)
            rows.append({'rule': rule, 'used': None, 'error': err})
            continue
        notes = []
        if used > 1:
            for j in range(1, 4):
                if not np.isnan(xc[0, j]) and abs(xc[0, j]) > 0.95 and stds[j] > 1e-4:
                    notes.append(f'ch0~ch{j}({xc[0,j]:+.2f})')
        notes_str = '  '.join(notes)
        verdict_short = verdict.split(' ')[0]
        print(f'  [{i+1:3d}/{len(rules)}] {rule:32s} -> {used}ch  {verdict_short}  {notes_str}',
              flush=True, file=sys.stderr)
        rows.append({
            'rule': rule, 'used': used, 'verdict': verdict,
            'std0': stds[0], 'std1': stds[1], 'std2': stds[2], 'std3': stds[3],
            'corr01': float(xc[0, 1]) if not np.isnan(xc[0, 1]) else None,
            'corr02': float(xc[0, 2]) if not np.isnan(xc[0, 2]) else None,
            'corr03': float(xc[0, 3]) if not np.isnan(xc[0, 3]) else None,
            'corr12': float(xc[1, 2]) if not np.isnan(xc[1, 2]) else None,
            'corr13': float(xc[1, 3]) if not np.isnan(xc[1, 3]) else None,
            'corr23': float(xc[2, 3]) if not np.isnan(xc[2, 3]) else None,
            'notes': notes_str,
        })
    dt = time.time() - t0
    print(f'\nDone in {dt:.1f}s\n')

    # Sorted summary table to stdout
    ok = [r for r in rows if r.get('used') is not None]
    ok.sort(key=lambda r: (-r['used'], r['rule']))
    print(f'{"rule":32s} {"used":>4s} {"std0":>8s} {"std1":>8s} {"std2":>8s} {"std3":>8s}  notes')
    print('-' * 100)
    for r in ok:
        print(f'  {r["rule"]:30s} {r["used"]:>4d}  '
              f'{r["std0"]:>7.4f} {r["std1"]:>7.4f} {r["std2"]:>7.4f} {r["std3"]:>7.4f}  '
              f'{r["notes"]}')
    err_rows = [r for r in rows if r.get('error')]
    for r in err_rows:
        print(f'  {r["rule"]:30s}  ERR  {r["error"]}')

    # Summary counts
    by_used = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for r in ok:
        by_used[r['used']] = by_used.get(r['used'], 0) + 1
    redundant = sum(1 for r in ok if r['notes'])
    print(f'\n=== Summary ===')
    for k in (1, 2, 3, 4):
        print(f'  {k}-channel rules: {by_used.get(k, 0):3d}')
    print(f'  0-channel (dead): {by_used.get(0, 0):3d}')
    print(f'  redundant ch0~chN: {redundant:3d}')
    print(f'  errored:          {len(err_rows):3d}')

    if args.csv:
        import csv as _csv
        cols = ['rule', 'used', 'verdict', 'std0', 'std1', 'std2', 'std3',
                'corr01', 'corr02', 'corr03', 'corr12', 'corr13', 'corr23',
                'notes', 'error']
        with open(args.csv, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'\nWrote CSV: {args.csv}')


def cmd_compare(args):
    files = [Path(f) for f in args.files]
    loaded = [(_load_npz(f), f.name) for f in files]
    print(f"\n=== compare {len(loaded)} snapshots ===")
    print(f"  {'file':<50} {'rule':<28} {'step':>6} {'ch0_mean':>9} "
          f"{'ch0_alive':>10} {'ch_used':>8} {'verdict'}")
    cached = []
    for (vox, meta, _), name in loaded:
        ch_stats = [_channel_stats(vox[..., c]) for c in range(4)]
        used = sum(1 for s in ch_stats if s['std'] > 1e-4)
        xc = _channel_corr(vox)
        v = _verdict(ch_stats, xc)
        print(f"  {name:<50} {meta.get('rule', '?'):<28} "
              f"{meta.get('step', 0):>6d} {ch_stats[0]['mean']:>9.4f} "
              f"{ch_stats[0]['alive_frac']:>10.4f} {used:>8d}  {v}")
        cached.append((vox, meta, name))

    if len(cached) >= 2:
        print(f"\n=== pairwise (channel-0 correlation matrix) ===")
        n = len(cached)
        names = [c[2][:28] for c in cached]
        print("       " + "  ".join(f'{i:>6}' for i in range(n)))
        for i in range(n):
            row = []
            ai = cached[i][0][..., 0].astype(np.float32).flatten()
            for j in range(n):
                if cached[j][0].shape != cached[i][0].shape:
                    row.append('   N/A')
                    continue
                bj = cached[j][0][..., 0].astype(np.float32).flatten()
                if ai.std() < 1e-8 or bj.std() < 1e-8:
                    row.append('   nan')
                else:
                    row.append(f'{np.corrcoef(ai, bj)[0,1]:>6.3f}')
            print(f"  [{i}]  " + "  ".join(row) + f"   {names[i]}")


# ───────────────────────────── main ─────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    pc = sub.add_parser('capture', help='single snapshot')
    pc.add_argument('rule')
    pc.add_argument('--step', type=int, default=100)
    pc.add_argument('--size', type=int, default=64)
    pc.add_argument('--seed', type=int, default=42)
    pc.add_argument('--out', type=str, default=None)
    pc.set_defaults(func=cmd_capture)

    pr = sub.add_parser('record', help='multi-checkpoint sequence')
    pr.add_argument('rule')
    pr.add_argument('--steps', type=str, required=True,
                    help='comma list: e.g. 0,10,50,100,200,500')
    pr.add_argument('--size', type=int, default=64)
    pr.add_argument('--seed', type=int, default=42)
    pr.add_argument('--out-dir', type=str, default=None)
    pr.set_defaults(func=lambda a: cmd_record(_with_default_dir(a)))

    pi = sub.add_parser('inspect', help='analyse one snapshot')
    pi.add_argument('file')
    pi.set_defaults(func=cmd_inspect)

    pd = sub.add_parser('diff', help='pairwise diff of two snapshots')
    pd.add_argument('a'); pd.add_argument('b')
    pd.set_defaults(func=cmd_diff)

    pcomp = sub.add_parser('compare', help='compare N snapshots side-by-side')
    pcomp.add_argument('files', nargs='+')
    pcomp.set_defaults(func=cmd_compare)

    pa = sub.add_parser('audit-channels',
                        help='audit channel usage across all leaf rules')
    pa.add_argument('--size', type=int, default=24)
    pa.add_argument('--steps', type=int, default=30)
    pa.add_argument('--seed', type=int, default=42)
    pa.add_argument('--filter', type=str, default=None,
                    help='regex; only audit rules matching this pattern')
    pa.add_argument('--csv', type=str, default=None,
                    help='write per-rule rows to CSV file')
    pa.set_defaults(func=cmd_audit_channels)

    args = p.parse_args(argv)
    args.func(args)


def _with_default_dir(args):
    if not args.out_dir:
        args.out_dir = f'snapshots/{args.rule}_s{args.size}_seed{args.seed}'
    return args


if __name__ == '__main__':
    main()
