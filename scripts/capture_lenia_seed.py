#!/usr/bin/env python3
"""Capture Lenia organism seeds into the seed library (seeds/lenia/*.npz).

Runs a rule headless, lets it settle, then extracts the N largest connected
components from the final state and saves each as a padded cubic crop.
These crops are stamped back into fresh volumes by the `lenia_seeded` init
variant (see simulator.init_lenia_seeded) — turning the FBM nucleation
lottery into guaranteed organisms.

Usage:
    # Capture from explicit params (validated geminium glider regime):
    python scripts/capture_lenia_seed.py lenia_geminium \\
        --params '{"Growth center":0.3066,"Growth width":0.0647,"Kernel radius":4.5267,"Ring position":0.5798}' \\
        --dt 0.0132 --init lenia_blobs --size 64 --steps 300 --seed 111 \\
        --name gem_glider

    # Capture from the top organism-ranked discovery for a rule:
    python scripts/capture_lenia_seed.py flow_lenia --from-discoveries
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'seeds', 'lenia')


def extract_components(field3d, n_max=3, threshold=0.10, pad=3, min_cells=30):
    """Return up to n_max cubic crops of the largest connected components."""
    from scipy import ndimage
    mask = field3d > threshold
    structure = np.zeros((3, 3, 3), dtype=np.int32)
    structure[1, 1, :] = 1
    structure[1, :, 1] = 1
    structure[:, 1, 1] = 1
    labels, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return []
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    crops = []
    for k in order[:n_max]:
        if sizes[k] < min_cells:
            continue
        lbl = k + 1
        where = np.argwhere(labels == lbl)
        lo = where.min(axis=0) - pad
        hi = where.max(axis=0) + pad + 1
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, field3d.shape)
        # Cubic crop (max extent on all axes) so orientation permutations
        # in init_lenia_seeded are shape-safe.
        ext = int((hi - lo).max())
        ctr = (lo + hi) // 2
        lo = np.maximum(ctr - ext // 2, 0)
        hi = np.minimum(lo + ext, field3d.shape)
        lo = hi - ext  # re-anchor if clipped at the far edge
        lo = np.maximum(lo, 0)
        crop = np.zeros((ext, ext, ext), dtype=np.float32)
        src = field3d[lo[0]:lo[0]+ext, lo[1]:lo[1]+ext, lo[2]:lo[2]+ext]
        crop[:src.shape[0], :src.shape[1], :src.shape[2]] = src
        # Zero out everything not in this component (avoid capturing
        # neighbouring structure fog).
        lblcrop = labels[lo[0]:lo[0]+ext, lo[1]:lo[1]+ext, lo[2]:lo[2]+ext]
        crop[:src.shape[0], :src.shape[1], :src.shape[2]] *= (lblcrop == lbl)
        crops.append(crop)
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rule')
    ap.add_argument('--params', type=str, default=None,
                    help='JSON dict of rule params')
    ap.add_argument('--dt', type=float, default=None)
    ap.add_argument('--init', type=str, default=None)
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--name', type=str, default=None,
                    help='Seed base name (default: rule name)')
    ap.add_argument('--threshold', type=float, default=0.10)
    ap.add_argument('--max-seeds', type=int, default=3)
    ap.add_argument('--from-discoveries', action='store_true',
                    help='Use params/dt/init of the top lenia_organism-ranked '
                         'discovery for this rule instead of --params')
    args = ap.parse_args()

    from test_harness import (create_headless_context, destroy_context,
                              HeadlessRunner, _get_metric)

    params = json.loads(args.params) if args.params else None
    dt, init, seed = args.dt, args.init, args.seed

    if args.from_discoveries:
        with open(os.path.join(os.path.dirname(SEED_DIR), '..',
                               'discoveries.json')) as f:
            d = json.load(f)
        rows = [e for e in (d if isinstance(d, list) else d.get('discoveries', d))
                if isinstance(e, dict) and e.get('rule') == args.rule]
        if not rows:
            sys.exit(f'no discoveries for {args.rule}')
        best = max(rows, key=lambda e: _get_metric(e, 'lenia_organism'))
        params, dt = best['params'], best.get('dt')
        init = best.get('init_variant')
        seed = best.get('seed', seed)
        print(f'from discoveries: org={_get_metric(best, "lenia_organism"):.3f} '
              f'params={params} dt={dt} init={init} seed={seed}')

    window, ctx = create_headless_context()
    runner = HeadlessRunner(ctx, args.rule, size=args.size, seed=seed,
                            params=params, dt=dt, init_override=init)
    for _ in range(args.steps):
        runner.step()
    grid = runner.read_grid()
    runner.release()
    destroy_context(window)

    field = grid[:, :, :, 0]
    crops = extract_components(field, n_max=args.max_seeds,
                               threshold=args.threshold)
    if not crops:
        sys.exit('no components above threshold — nothing captured')

    os.makedirs(SEED_DIR, exist_ok=True)
    base = args.name or args.rule
    for i, crop in enumerate(crops):
        path = os.path.join(SEED_DIR, f'{base}_{i}.npz')
        np.savez_compressed(path, field=crop)
        print(f'saved {path}  shape={crop.shape} mass={crop.sum():.1f} '
              f'max={crop.max():.3f}')


if __name__ == '__main__':
    main()
