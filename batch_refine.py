#!/usr/bin/env python3
"""Search the local parameter neighbourhood of a *refined* discovery.

Reads a refined entry (by --idx or --hash) from discoveries.json,
takes its params as the centre, and runs `--trials` mutated trials.
If the parent's refinement report has `param_elasticity`, the
mutation scale of each param is widened/narrowed by its |Pearson r|
so high-impact params get explored more carefully.

Survivors above --min-quality are appended to discoveries.json with a
`derived_from` block pointing back to the parent hash, so the lineage
is browsable in the simulator.

Usage:
    python batch_refine.py --idx 24759 --trials 200
    python batch_refine.py --hash 5723a8d517 --trials 500 --span 0.20
    python batch_refine.py --idx 0 --trials 100 --no-elasticity --min-quality 0.30

Output: appended to discoveries.json (same schema). Each new entry has
    "derived_from": {"parent_hash": "...", "parent_index": N,
                     "centre_params": {...}, "span": X, "metric": "..."}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import fcntl

import numpy as np

from schema import get_field

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DISC_PATH = os.path.join(THIS_DIR, 'discoveries.json')
LOCK_PATH = DISC_PATH + '.lock'


def _short_hash(entry):
    """Mirrors refine.short_hash and simulator._refine_short_hash."""
    import hashlib
    rule = get_field(entry, 'rule', '')
    params = sorted((str(k), float(v))
                    for k, v in (get_field(entry, 'params', {}) or {}).items())
    seed = int(get_field(entry, 'seed', 0))
    key = json.dumps([rule, params, seed], sort_keys=True)
    return hashlib.sha1(key.encode('utf-8')).hexdigest()[:10]


def _load_discoveries():
    with open(DISC_PATH) as f:
        return json.load(f)


def _resolve_parent(all_disc, args):
    """Return (parent_index, parent_entry) from --idx or --hash."""
    if args.idx is not None:
        if not (0 <= args.idx < len(all_disc)):
            sys.exit(f"--idx {args.idx} out of range (0..{len(all_disc) - 1})")
        return args.idx, all_disc[args.idx]
    if args.hash:
        for i, e in enumerate(all_disc):
            if _short_hash(e) == args.hash:
                return i, e
        sys.exit(f"--hash {args.hash} not found in {DISC_PATH}")
    sys.exit("specify --idx N or --hash H")


def _elasticity_scales(report, param_names, base_span):
    """Return {param: scale} where high-|r| params get more exploration.

    Mapping: scale = base_span * (0.5 + |r| * 1.0)  clamped to [0.25, 2.0] * base.
    Params with no elasticity entry default to base_span.
    """
    el = (report or {}).get('param_elasticity') or {}
    if not el:
        return {p: base_span for p in param_names}
    scales = {}
    for p in param_names:
        r = el.get(p)
        if r is None or not np.isfinite(r):
            scales[p] = base_span
        else:
            s = base_span * (0.5 + abs(float(r)))
            scales[p] = float(np.clip(s, 0.25 * base_span, 2.0 * base_span))
    return scales


def _mutate(centre, ranges, scales, rng):
    """Per-param mutation with per-param scale."""
    out = {}
    for name, val in centre.items():
        if name not in ranges:
            out[name] = val
            continue
        lo, hi = ranges[name]
        if lo > hi:
            lo, hi = hi, lo
        scale = scales.get(name, 0.10)
        if isinstance(lo, int) and isinstance(hi, int):
            delta = max(1, int((hi - lo) * scale))
            new = int(val + rng.randint(-delta, delta + 1))
            out[name] = max(lo, min(hi, new))
        else:
            span = hi - lo
            delta = rng.normal(0, span * scale)
            out[name] = float(max(lo, min(hi, val + delta)))
    return out


def _append_survivors(survivors):
    """Atomically append survivors to discoveries.json under flock."""
    if not survivors:
        return 0
    os.makedirs(os.path.dirname(LOCK_PATH) or '.', exist_ok=True)
    with open(LOCK_PATH, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with open(DISC_PATH) as f:
                data = json.load(f)
            data.extend(survivors)
            tmp = DISC_PATH + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, DISC_PATH)
            return len(survivors)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--idx', type=int, help='Parent discovery index')
    g.add_argument('--hash', type=str, help='Parent discovery short hash')

    p.add_argument('--trials', type=int, default=200,
                   help='Number of mutated trials (default: 200)')
    p.add_argument('--span', type=float, default=0.10,
                   help='Base mutation scale as fraction of param range '
                        '(default: 0.10)')
    p.add_argument('--no-elasticity', action='store_true',
                   help='Ignore param_elasticity from the parent report')
    p.add_argument('--size', type=int, default=48,
                   help='Grid size for trials (default: 48 — fast)')
    p.add_argument('--steps', type=int, default=200,
                   help='Steps per trial (default: 200)')
    p.add_argument('--min-quality', type=float, default=0.25,
                   help='Drop trials with score below this (default: 0.25)')
    p.add_argument('--metric', type=str, default='score',
                   help='Sort metric (default: score)')
    p.add_argument('--seed', type=int, default=None,
                   help='RNG seed (default: time-based)')
    p.add_argument('--dynamics', action='store_true',
                   help='Enable dynamics capture (slower, finds gliders)')
    p.add_argument('--vary-init', action='store_true',
                   help='Pick a random init_variant per trial from the '
                        "preset's init_variants list (default: lock to parent).")
    p.add_argument('--vary-density', action='store_true',
                   help="Jitter init_density within the preset's density "
                        "range (default: lock to parent).")
    p.add_argument('--dry-run', action='store_true',
                   help='Show plan; do not run trials')
    args = p.parse_args()

    all_disc = _load_discoveries()
    parent_idx, parent = _resolve_parent(all_disc, args)
    parent_hash = _short_hash(parent)
    rule = parent['rule']
    centre = dict(get_field(parent, 'params', {}) or {})
    if not centre:
        sys.exit(f"parent #{parent_idx} ({rule}) has no params — nothing to mutate")

    # Try to load the refinement report (only if entry was refined).
    report = None
    refblock = parent.get('refinement') or {}
    if refblock.get('dir'):
        rp = os.path.join(THIS_DIR, refblock['dir'], 'report.json')
        if os.path.exists(rp):
            try:
                with open(rp) as f:
                    report = json.load(f)
            except Exception as e:  # noqa: BLE001  malformed JSON, treat as missing
                print(f"[batch_refine] warn: cannot read {rp}: {e}",
                      file=sys.stderr)

    # Import sim machinery only after arg-parse to keep --help fast.
    from simulator import RULE_PRESETS, _resolve_composed_preset
    from test_harness import (
        create_headless_context, destroy_context,
        run_trial, _make_discovery, _DENSITY_RANGES,
    )
    preset = _resolve_composed_preset(rule)
    ranges = preset.get('param_ranges') or {}
    scales = (
        {p: args.span for p in centre} if args.no_elasticity
        else _elasticity_scales(report, list(centre), args.span)
    )

    print(f"[batch_refine] parent #{parent_idx} {rule} hash={parent_hash}")
    print(f"[batch_refine] centre params: {centre}")
    print(f"[batch_refine] per-param scales: "
          f"{ {k: round(v,3) for k,v in scales.items()} }")
    if report and not args.no_elasticity:
        el = report.get('param_elasticity') or {}
        if el:
            print(f"[batch_refine] elasticity (Pearson r vs active_frac): "
                  f"{ {k: round(float(v),3) for k,v in el.items()} }")
    print(f"[batch_refine] {args.trials} trials, size={args.size}, "
          f"steps={args.steps}, min_quality={args.min_quality}, "
          f"metric={args.metric}, dynamics={args.dynamics}")

    # Preserve the parent's init topology / density / dt so survivors
    # are honest neighbours in (params, init, dt) space rather than
    # accidental rediscoveries with the preset's default init.
    parent_init_variant = parent.get('init_variant') or None
    parent_init_density = parent.get('init_density')
    parent_dt = parent.get('dt')

    # Optional broader exploration: cycle through init_variants and/or
    # jitter density. When disabled (default), every trial uses the
    # parent's exact init setup.
    init_variants_pool = list(preset.get('init_variants') or [])
    if args.vary_init and len(init_variants_pool) <= 1:
        print(f"[batch_refine] --vary-init requested but preset has "
              f"{len(init_variants_pool)} variants; falling back to locked init",
              file=sys.stderr)
    init_type = preset.get('init', '')
    density_range = _DENSITY_RANGES.get(init_type)
    if args.vary_density and not density_range:
        print(f"[batch_refine] --vary-density requested but init '{init_type}' "
              f"has no density range; falling back to locked density",
              file=sys.stderr)

    if parent_init_variant:
        tag = 'cycled per trial' if (args.vary_init and len(init_variants_pool) > 1) \
              else 'locked to parent'
        print(f"[batch_refine] init_variant: {parent_init_variant} ({tag})")
        if args.vary_init and len(init_variants_pool) > 1:
            print(f"[batch_refine]   pool: {init_variants_pool}")
    if parent_init_density is not None:
        tag = 'jittered per trial' if (args.vary_density and density_range) \
              else 'locked to parent'
        print(f"[batch_refine] init_density: {parent_init_density:.3f} ({tag})")
        if args.vary_density and density_range:
            print(f"[batch_refine]   range: [{density_range[0]:.2f}, {density_range[1]:.2f}]")
    if parent_dt is not None:
        print(f"[batch_refine] dt: {parent_dt:.4g} (locked to parent)")
    if args.dry_run:
        return

    seed = args.seed if args.seed is not None else int(time.time()) & 0x7fffffff
    rng = np.random.RandomState(seed)
    window, ctx = create_headless_context()

    derived_meta = {
        'parent_hash': parent_hash,
        'parent_index': parent_idx,
        'centre_params': centre,
        'span': args.span,
        'metric': args.metric,
        'elasticity_weighted': bool(report) and not args.no_elasticity,
        'init_variant': parent_init_variant,
        'init_density': parent_init_density,
        'dt': parent_dt,
        'vary_init': bool(args.vary_init and len(init_variants_pool) > 1),
        'vary_density': bool(args.vary_density and density_range),
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    survivors = []
    t0 = time.time()
    for t in range(args.trials):
        params = _mutate(centre, ranges, scales, rng)
        # Per-trial init selection.
        if args.vary_init and len(init_variants_pool) > 1:
            trial_init = init_variants_pool[rng.randint(len(init_variants_pool))]
        else:
            trial_init = parent_init_variant
        if args.vary_density and density_range:
            d_lo, d_hi = density_range
            trial_density = float(rng.uniform(d_lo, d_hi))
        else:
            trial_density = parent_init_density
        trial_seed = int(rng.randint(0, 10_000_000))
        try:
            result = run_trial(
                ctx, rule, size=args.size, steps=args.steps,
                seed=trial_seed, params=params,
                dt=parent_dt,
                init_density=trial_density,
                init_override=trial_init,
                verbose=False,
                capture_dynamics=args.dynamics,
            )
        except Exception as e:  # noqa: BLE001  trial may crash on bad params, score=0
            print(f"  [{t+1:4d}/{args.trials}] FAIL: {e}", file=sys.stderr)
            continue
        # run_trial doesn't echo init_variant back; stamp it ourselves so
        # _make_discovery propagates it into the saved entry.
        if trial_init:
            result['init_variant'] = trial_init
        if trial_density is not None:
            result['init_density'] = trial_density
        score = float(result.get(args.metric, result.get('score', 0.0)))
        # NOTE: `result` here is a run_trial output dict, NOT a discovery
        # entry, so it's intentionally read with raw .get().
        kept = score >= args.min_quality
        if kept:
            d = _make_discovery(result)
            d['derived_from'] = dict(derived_meta)
            survivors.append(d)
        if (t + 1) % 10 == 0 or kept:
            elapsed = time.time() - t0
            rate = (t + 1) / max(elapsed, 1e-6)
            eta = (args.trials - t - 1) / max(rate, 1e-6)
            tag = 'KEEP' if kept else '    '
            print(f"  [{t+1:4d}/{args.trials}] {tag} {args.metric}={score:.3f}  "
                  f"survivors={len(survivors)}  rate={rate:.1f}/s  eta={eta:.0f}s")

    destroy_context(window)
    n_saved = _append_survivors(survivors)
    elapsed = time.time() - t0
    print(f"\n[batch_refine] done in {elapsed:.1f}s: "
          f"{n_saved}/{args.trials} survivors appended to discoveries.json")
    print(f"[batch_refine] new entries link back via derived_from.parent_hash={parent_hash}")


if __name__ == '__main__':
    main()
