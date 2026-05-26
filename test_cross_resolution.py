#!/usr/bin/env python3
"""Cross-resolution correctness & perf harness for simulator.py.

For every preset rule, runs the simulator at a range of grid sizes using
the *same machinery the search/UI uses* (test_harness.HeadlessRunner)
and reports:
  - NaN / Inf appearance
  - Per-channel min / max / mean / var at end of run
  - Per-channel stats for the second texture pair (pair 2) on multi-pair
    rules, since rules like compressible_euler_3d store half their
    conserved state there.
  - Scaling behaviour: does mean/var stay consistent across sizes?
  - Wall-clock per step

Why HeadlessRunner instead of a custom run-loop:

The previous version of this script wrote uniform-random data into a
single rgba32f texture and dispatched the compute shader manually. That
broke three classes of rule:

  - Composite presets (`compose: [...]`) like flagship_dual_lenia have
    no top-level `shader` key, so `preset['shader']` raised KeyError.
  - Multi-pair rules (`extra_fields: N`) like compressible_euler_3d
    expect a second texture pair bound at images 2/3, plus a specific
    pair-2 init. Without it, the shader read garbage and immediately
    blew up to NaN — a false-positive "engine bug".
  - Rules with init-specific assumptions (hydrodynamics, BZ, Lenia,
    quantum) need their preset's `init` function. Uniform-random gave
    them unphysical starting conditions and the resulting variance
    collapse looked like a numerics bug.

HeadlessRunner already resolves composites, allocates extra_fields
texture pairs, runs the preset's `init` function, applies preset params
+ dt + boundary, and steps. Using it removes all three blind spots.

Usage:
    python3 test_cross_resolution.py                 # default 32/64/128
    python3 test_cross_resolution.py --sizes 64,128,256
    python3 test_cross_resolution.py --only reaction_diffusion_3d,bz_3d
    python3 test_cross_resolution.py --big           # include 256,384,512
    python3 test_cross_resolution.py --scale-steps   # steps ∝ size/min_size
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Force the harness to honour the size we ask for. Without this,
# `pref_size = preset['search_size'] or preset['default_size']` bumps
# small sizes up — which defeats the whole point of cross-resolution
# testing (we want to *see* what happens at size=32 for a rule whose
# default is 192).
os.environ.setdefault('CA_HARNESS_ALLOW_UNDERSIZE', '1')

import moderngl  # noqa: E402  (env-var must be set first)

from simulator import RULE_PRESETS  # noqa: E402
from test_harness import HeadlessRunner  # noqa: E402


# ---------------------------------------------------------------------------
# Stats over a (size, size, size, 4) grid.
# ---------------------------------------------------------------------------
def _channel_stats(arr):
    """Return per-channel {min,max,mean,var} dicts (None where empty)."""
    out = []
    for c in range(arr.shape[-1]):
        col = arr[..., c]
        col = col[np.isfinite(col)]
        if col.size == 0:
            out.append(None)
        else:
            out.append({
                "min":  float(col.min()),
                "max":  float(col.max()),
                "mean": float(col.mean()),
                "var":  float(col.var()),
            })
    return out


def _read_pair2(runner):
    """If the rule uses a second texture pair, read it back as (s,s,s,4).
    Returns None if no pair 2 is allocated."""
    tex_a2 = getattr(runner, 'tex_a2', None)
    if tex_a2 is None:
        return None
    src2 = runner.tex_a2 if getattr(runner, 'ping2', 0) == 0 else runner.tex_b2
    np_dtype = runner._tex_np_dtype
    raw = np.frombuffer(src2.read(), dtype=np_dtype).reshape(
        runner.size, runner.size, runner.size, 4)
    if np_dtype != np.float32:
        raw = raw.astype(np.float32)
    return raw


# ---------------------------------------------------------------------------
# Run one preset at one size, return a dict of diagnostics.
# ---------------------------------------------------------------------------
def run_once(ctx, preset_name, size, steps=30, seed=7):
    try:
        runner = HeadlessRunner(ctx, preset_name, size=size, seed=seed)
    except Exception as e:  # noqa: BLE001  preset alloc may fail on bad combos
        return {"error": f"alloc: {type(e).__name__}: {e}"}

    actual_size = runner.size

    ctx.finish()
    t0 = time.perf_counter()
    try:
        for _ in range(steps):
            runner.step()
        ctx.finish()
        elapsed = time.perf_counter() - t0
        grid = runner.read_grid()
        pair2 = _read_pair2(runner)
    except Exception as e:  # noqa: BLE001  shader run may fail
        try:
            runner.release()
        except Exception:  # noqa: BLE001  release is best-effort
            pass
        return {"error": f"run: {type(e).__name__}: {e}"}

    nan = int(np.isnan(grid).sum())
    inf = int(np.isinf(grid).sum())
    if pair2 is not None:
        nan += int(np.isnan(pair2).sum())
        inf += int(np.isinf(pair2).sum())

    stats = {
        "time_ms_per_step": 1000.0 * elapsed / steps,
        "nan": nan,
        "inf": inf,
        "actual_size": actual_size,
        "channels": _channel_stats(grid),
    }
    if pair2 is not None:
        stats["channels_pair2"] = _channel_stats(pair2)

    try:
        runner.release()
    except Exception:  # noqa: BLE001  release is best-effort
        pass
    return stats


# ---------------------------------------------------------------------------
# Summary across sizes: flag genuine instabilities, not noise.
# ---------------------------------------------------------------------------
EXPLOSION_THRESHOLD = 1e4   # any channel value > this is probably a blow-up
BLOWUP_RATIO        = 1e3   # variance ratio between adjacent sizes


def _compare_one(per_size, key):
    """Compare `channels` or `channels_pair2` across sizes."""
    warns = []
    sizes = sorted(s for s, r in per_size.items()
                   if isinstance(r, dict) and key in r)
    if len(sizes) < 2:
        return warns
    n_ch = max(len(per_size[sz][key]) for sz in sizes)
    pair_tag = "" if key == "channels" else " (pair2)"
    for c in range(n_ch):
        for sz in sizes:
            ch = per_size[sz][key][c] if c < len(per_size[sz][key]) else None
            if ch is None:
                continue
            if max(abs(ch["min"]), abs(ch["max"])) > EXPLOSION_THRESHOLD:
                warns.append(
                    f"ch{c}{pair_tag} @ size={sz}: out of plausible range "
                    f"[{ch['min']:.3g}, {ch['max']:.3g}]"
                )
        for i in range(len(sizes) - 1):
            a_list = per_size[sizes[i]][key]
            b_list = per_size[sizes[i + 1]][key]
            a = a_list[c] if c < len(a_list) else None
            b = b_list[c] if c < len(b_list) else None
            if a is None or b is None:
                continue
            va, vb = a["var"], b["var"]
            if va > 1.0 and vb > 1e-6 and va / vb > BLOWUP_RATIO:
                warns.append(
                    f"ch{c}{pair_tag}: variance collapses {sizes[i]}→{sizes[i+1]} "
                    f"({va:.3g} → {vb:.3g}) — CFL-unstable at small grid?"
                )
    return warns


def compare_scaling(per_size):
    warns = _compare_one(per_size, "channels")
    warns += _compare_one(per_size, "channels_pair2")
    return warns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _is_composite(name):
    """Returns True for `compose: [...]` presets. Reported, not skipped —
    HeadlessRunner resolves them transparently."""
    return 'compose' in RULE_PRESETS.get(name, {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="32,64,128")
    ap.add_argument("--big", action="store_true",
                    help="Also test 256 & 384 & 512 (needs lots of VRAM)")
    ap.add_argument("--steps", type=int, default=30,
                    help="Step count at the smallest size. With "
                         "--scale-steps, larger sizes get proportionally "
                         "more steps so simulated time is matched.")
    ap.add_argument("--scale-steps", action="store_true",
                    help="Scale steps ∝ size/min_size so that "
                         "front-propagation rules and diffusion-paced "
                         "rules reach equivalent simulated time at every "
                         "size. Without this flag, all sizes run the same "
                         "step count and slow rules look like they 'die' "
                         "at high resolution.")
    ap.add_argument("--only", default="",
                    help="Comma-separated preset names to restrict to")
    ap.add_argument("--json", default="",
                    help="Write full results to this JSON path")
    args = ap.parse_args()

    sizes = sorted(int(s) for s in args.sizes.split(","))
    if args.big:
        sizes = sorted(set(sizes + [256, 384, 512]))
    only = set(s.strip() for s in args.only.split(",") if s.strip())

    ctx = moderngl.create_standalone_context(require=430)
    try:
        _run_all(ctx, args, sizes, only)
    finally:
        ctx.release()


def _run_all(ctx, args, sizes, only):
    print(f"GL renderer : {ctx.info.get('GL_RENDERER','?')}")
    print(f"GL vendor   : {ctx.info.get('GL_VENDOR','?')}")
    print(f"GL version  : {ctx.info.get('GL_VERSION','?')}")
    print(f"Sizes       : {sizes}")
    base_steps = args.steps
    step_mode = "scaled" if args.scale_steps else "fixed"
    print(f"Steps       : {base_steps} @ smallest size [{step_mode}]")
    print(f"Presets     : {len(RULE_PRESETS)}" +
          (f" (filtered to {len(only)})" if only else ""))
    print()

    all_results = {}
    n_presets = sum(1 for n in RULE_PRESETS if not only or n in only)
    grand_t0 = time.perf_counter()
    idx = 0

    min_size = min(sizes)
    for preset_name in RULE_PRESETS:
        if only and preset_name not in only:
            continue
        idx += 1
        per_size = {}
        composite_tag = " [composite]" if _is_composite(preset_name) else ""
        line = f"[{idx:>2}/{n_presets}] {preset_name:32s}{composite_tag}"
        sys.stdout.write(line); sys.stdout.flush()
        for sz in sizes:
            steps_here = (base_steps * sz // min_size
                          if args.scale_steps else base_steps)
            sys.stdout.write(f"  [{sz}…]"); sys.stdout.flush()
            try:
                r = run_once(ctx, preset_name, sz, steps=steps_here)
            except Exception as e:  # noqa: BLE001  trial may crash on bad params
                r = {"error": f"harness exc: {type(e).__name__}: {e}"}
            per_size[sz] = r
            sys.stdout.write("\b" * (len(f"  [{sz}…]")))
            if "error" in r:
                cell = f"  [{sz:>3}] ERR"
            elif r.get("nan", 0) or r.get("inf", 0):
                cell = f"  [{sz:>3}] NAN({r['nan']})/INF({r['inf']})"
            else:
                cell = f"  [{sz:>3}] {r['time_ms_per_step']:5.1f}ms"
            sys.stdout.write(cell); sys.stdout.flush()
        warns = compare_scaling(per_size)
        wall = time.perf_counter() - grand_t0
        sys.stdout.write(f"   ({wall:5.0f}s elapsed)\n")
        if warns:
            sys.stdout.write("    " + "\n    ".join(warns) + "\n")
        sys.stdout.flush()
        all_results[preset_name] = per_size

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nWrote {args.json}")

    bad = any(
        "error" in r or r.get("nan", 0) or r.get("inf", 0)
        for preset in all_results.values() for r in preset.values()
    )
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
