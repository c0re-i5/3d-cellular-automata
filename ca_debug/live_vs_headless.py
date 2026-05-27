"""Probe #20 — live-vs-headless equivalence.

For each "comparable" preset, run the interactive `Simulator` (in
headless GLFW mode, so no window is shown but the *production* step
function `_step_sim()` executes) and the `HeadlessRunner.step()` path
with the same seed and size, then diff the resulting grids.

Why this exists
---------------
`HeadlessRunner.step()` in `test_harness.py` and `Simulator._step_sim()`
in `simulator.py` are *parallel* implementations of the same logical
pipeline. They drift over time: pass-binding order changes, new
uniforms get added, ping-pong rules diverge, etc. Drift between the two
means search/audit results don't reflect what a user actually sees in
the live simulator — and that is the worst category of bug because
silent.

What's comparable
-----------------
We only diff presets whose *feature surface* both engines fully
support. We skip any preset that uses:
  - particles (`particle_count > 0`)              live-only
  - extra fields (`extra_fields >= 1`)            live-only (p3, p4, ...)
  - sparse dispatch                               live-only
  - viewport kind                                 fractal-render
  - element CA (different init paths)
  - entity arena (covered by Probe #19)
  - non-standard precision (preset['precision'])

Even within the "comparable" set we tolerate tiny absolute differences
(epsilon below) — both engines submit the *same* shader source and
uniforms, so any divergence indicates a real implementation drift, but
fp32 ordering / driver scheduler / vendor quirks can produce ULP-level
noise we don't want to flag as "bug".

Usage
-----
    python -m ca_debug.live_vs_headless                # default sweep
    python -m ca_debug.live_vs_headless --rules a,b,c  # explicit list
    python -m ca_debug.live_vs_headless --size 64 --steps 20
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

# Suppress simulator's stdout chatter during construction
_DEVNULL = open(os.devnull, "w")


def _comparable(name, preset):
    """Return True if HeadlessRunner can faithfully reproduce Simulator
    for this preset."""
    if preset.get("kind") == "viewport":
        return False
    if preset.get("is_element_ca", False):
        return False
    if int(preset.get("particle_count", 0)) > 0:
        return False
    if int(preset.get("extra_fields", 0)) >= 1:
        return False
    if preset.get("sparse_dispatch", False):
        return False
    if "entity_arena" in preset:
        return False  # Probe #19 covers this end
    if "agent_count" in preset:
        return False  # rare; skip to keep scope tight
    if preset.get("precision") not in (None, "fp32"):
        return False
    return True


def _candidate_presets():
    from simulator import RULE_PRESETS, _resolve_composed_preset
    names = []
    for n in RULE_PRESETS:
        try:
            p = _resolve_composed_preset(n)
        except Exception:  # noqa: BLE001 — skip presets that fail to resolve
            continue
        if _comparable(n, p):
            names.append(n)
    return sorted(names)


def _run_headless(name, *, size, seed, steps):
    """Run N steps in HeadlessRunner. Returns final grid (float32)."""
    import moderngl
    from test_harness import HeadlessRunner

    ctx = moderngl.create_standalone_context(require=430)
    try:
        runner = HeadlessRunner(ctx, name, size=size, seed=seed)
        actual_size = runner.size  # may have been auto-bumped by preset
        try:
            for _ in range(steps):
                runner.step()
            grid = runner.read_grid()
        finally:
            runner.release()
        return grid, actual_size
    finally:
        ctx.release()


def _run_live(name, *, size, seed, steps):
    """Run N steps in Simulator(headless=True). Returns final grid (float32)."""
    # Late import; pulls in glfw + heavy modules.
    with contextlib.redirect_stdout(_DEVNULL), contextlib.redirect_stderr(_DEVNULL):
        from simulator import Simulator
        import glfw

        sim = Simulator(size=size, rule=name, headless=True)
        try:
            # Honor requested seed (Simulator defaults to 42 → reset).
            if sim.seed != seed:
                sim.seed = seed
                sim._reset()
            actual_size = sim.size
            for _ in range(steps):
                sim._step_sim()
            # Read current src texture.
            src = sim.tex_a if sim.ping == 0 else sim.tex_b
            raw = np.frombuffer(src.read(), dtype=np.float32)
            grid = raw.reshape(actual_size, actual_size, actual_size, 4).copy()
        finally:
            # Tear down GL + window so subsequent presets can rebuild.
            try:
                glfw.destroy_window(sim.window)
            except Exception:  # noqa: BLE001 — best-effort
                pass
            try:
                glfw.terminate()
            except Exception:  # noqa: BLE001 — best-effort
                pass
        return grid, actual_size


def _diff(grid_h, grid_l):
    """Return diff stats dict."""
    if grid_h.shape != grid_l.shape:
        return {
            "shape_mismatch": True,
            "headless_shape": list(grid_h.shape),
            "live_shape": list(grid_l.shape),
        }
    nan_h = int(np.isnan(grid_h).sum())
    nan_l = int(np.isnan(grid_l).sum())
    # Treat NaN==NaN as equal for comparison.
    finite_mask = np.isfinite(grid_h) & np.isfinite(grid_l)
    diff = np.zeros_like(grid_h)
    diff[finite_mask] = np.abs(grid_h[finite_mask] - grid_l[finite_mask])
    nonfinite_disagree = int((np.isfinite(grid_h) != np.isfinite(grid_l)).sum())
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    # Worst voxel for diagnostics.
    if max_abs > 0:
        idx = np.unravel_index(int(diff.argmax()), diff.shape)
        worst = {
            "idx": [int(i) for i in idx],
            "headless": float(grid_h[idx]),
            "live": float(grid_l[idx]),
        }
    else:
        worst = None
    return {
        "shape": list(grid_h.shape),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "nan_headless": nan_h,
        "nan_live": nan_l,
        "nonfinite_disagree": nonfinite_disagree,
        "worst": worst,
    }


def _verdict(stats, *, eps_abs):
    if stats.get("shape_mismatch"):
        return "FAIL"
    if stats["nonfinite_disagree"] > 0:
        return "FAIL"
    if stats["max_abs"] > eps_abs:
        return "FAIL"
    if stats["max_abs"] > 0:
        return "warn"  # within eps but not bit-identical
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", help="comma-separated subset of presets")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=1e-5,
                    help="max abs diff tolerated before FAIL")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--max-presets", type=int, default=0,
                    help="cap how many comparable presets to test (0=all)")
    args = ap.parse_args()

    if args.rules:
        names = [n.strip() for n in args.rules.split(",") if n.strip()]
    else:
        names = _candidate_presets()
        if args.max_presets > 0:
            names = names[: args.max_presets]

    print(f"live-vs-headless — size={args.size} steps={args.steps} "
          f"seed={args.seed} eps={args.eps}  (presets: {len(names)})")
    t0 = time.time()
    results = []
    n_ok = n_warn = n_bad = 0
    for name in names:
        try:
            grid_h, size_h = _run_headless(
                name, size=args.size, seed=args.seed, steps=args.steps)
        except Exception as e:  # noqa: BLE001 — record + continue
            results.append({"name": name, "headless_error": repr(e)})
            n_bad += 1
            print(f"  ERR  {name:<36} headless: {e!r}")
            continue
        try:
            grid_l, size_l = _run_live(
                name, size=args.size, seed=args.seed, steps=args.steps)
        except Exception as e:  # noqa: BLE001
            results.append({"name": name, "live_error": repr(e)})
            n_bad += 1
            print(f"  ERR  {name:<36} live: {e!r}")
            continue
        if size_h != size_l:
            print(f"  SKIP {name:<36} size mismatch h={size_h} l={size_l}")
            results.append({"name": name, "skip": "size_mismatch",
                            "size_headless": size_h, "size_live": size_l})
            continue
        stats = _diff(grid_h, grid_l)
        verdict = _verdict(stats, eps_abs=args.eps)
        stats["name"] = name
        stats["verdict"] = verdict
        results.append(stats)
        if verdict == "ok":
            n_ok += 1
            print(f"  ok   {name:<36} bit-identical")
        elif verdict == "warn":
            n_warn += 1
            print(f"  warn {name:<36} max_abs={stats['max_abs']:.2e} "
                  f"mean_abs={stats['mean_abs']:.2e}")
        else:
            n_bad += 1
            w = stats.get("worst")
            extra = ""
            if w is not None:
                extra = (f"  worst@{tuple(w['idx'])} "
                         f"h={w['headless']:.4g} l={w['live']:.4g}")
            print(f"  BAD  {name:<36} max_abs={stats.get('max_abs', float('nan')):.2e}"
                  f"{extra}")

    dt = time.time() - t0
    print(f"\nSummary: ok={n_ok}  warn={n_warn}  bad={n_bad}  "
          f"(n={len(names)}, {dt:.1f}s)")
    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
