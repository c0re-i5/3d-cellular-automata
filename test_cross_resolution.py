#!/usr/bin/env python3
"""Cross-resolution correctness & perf harness for simulator.py.

For every preset rule, runs the compute shader at a range of grid sizes
and reports:
  - NaN / Inf appearance
  - Per-channel min / max / mean / var at end of run
  - Scaling behaviour: does mean/var stay consistent across sizes?
    (If the rule's h_sq / h_inv scaling is correct, bulk statistics after
     an equal *simulated* time should converge — modulo finite-grid effects.)
  - Wall-clock per step (to surface the huge grids that still run fine
    versus the ones that fall off a cliff).

Usage:
    python3 test_cross_resolution.py                 # default 32/64/128
    python3 test_cross_resolution.py --sizes 64,128,256
    python3 test_cross_resolution.py --only reaction_diffusion_3d,bz_3d
    python3 test_cross_resolution.py --big           # include 256,384,512
"""

import argparse, json, time, sys, os, traceback
import numpy as np
import moderngl

# ---------------------------------------------------------------------------
# Load simulator module without executing its GUI main()
# ---------------------------------------------------------------------------
_SRC = open(os.path.join(os.path.dirname(__file__), "simulator.py")).read()
_END = _SRC.find("class Simulator")
_NS: dict = {"__name__": "_harness"}
exec(compile(_SRC[:_END], "simulator.py", "exec"), _NS)

CA_RULES        = _NS["CA_RULES"]
COMPUTE_HEADER  = _NS["COMPUTE_HEADER"]
RULE_PRESETS    = _NS["RULE_PRESETS"]
_BOUNDARY       = _NS["_BOUNDARY_NAME_TO_MODE"]

# ---------------------------------------------------------------------------
# Per-rule default parameter vector for headless tests (4 floats).
# Pulled from the preset where possible; ordering matches u_param0..3.
# ---------------------------------------------------------------------------
def preset_params(preset):
    """Return (param0..3) for this preset in insertion order."""
    p = list(preset.get("params", {}).values())
    while len(p) < 4:
        p.append(0.0)
    return tuple(float(x) for x in p[:4])


# ---------------------------------------------------------------------------
# Run one preset at one size, return a dict of diagnostics.
# ---------------------------------------------------------------------------
def run_once(ctx, preset_name, preset, size, steps=20, seed=7):
    shader_name = preset["shader"]
    if shader_name not in CA_RULES:
        return {"skip": f"no shader {shader_name}"}

    src = (COMPUTE_HEADER + CA_RULES[shader_name]).replace(
        "#version 430",
        "#version 430\n#define USE_SHARED_MEM 1",
    )
    try:
        prog = ctx.compute_shader(src)
    except Exception as e:
        return {"error": f"compile: {e}"}

    rng = np.random.default_rng(seed)
    data = rng.random((size, size, size, 4), dtype=np.float32)
    try:
        tex_a = ctx.texture3d((size, size, size), 4, data.tobytes(), dtype="f4")
        tex_b = ctx.texture3d((size, size, size), 4, dtype="f4")
    except Exception as e:
        prog.release()
        return {"error": f"alloc: {e}"}
    for t in (tex_a, tex_b):
        t.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def setu(k, v):
        if k in prog:
            try:
                prog[k].value = v
            except Exception:
                pass

    dt = float(preset.get("dt", 0.3))
    p0, p1, p2, p3 = preset_params(preset)
    bmode = _BOUNDARY.get(preset.get("boundary", "toroidal"), 0)
    setu("u_size", size)
    setu("u_dt", dt)
    setu("u_boundary", bmode)
    setu("u_param0", p0); setu("u_param1", p1)
    setu("u_param2", p2); setu("u_param3", p3)

    src_t, dst_t = tex_a, tex_b
    ctx.finish()
    t0 = time.perf_counter()
    try:
        for f in range(steps):
            setu("u_frame", f)
            src_t.bind_to_image(0, read=True, write=False)
            dst_t.bind_to_image(1, read=False, write=True)
            g = (size + 7) // 8
            prog.run(g, g, g)
            src_t, dst_t = dst_t, src_t
        ctx.finish()
        elapsed = time.perf_counter() - t0
        arr = np.frombuffer(src_t.read(), dtype=np.float32).reshape(size, size, size, 4)
    except Exception as e:
        tex_a.release(); tex_b.release(); prog.release()
        return {"error": f"run: {e}"}

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    finite = arr[np.isfinite(arr)]
    stats = {
        "time_ms_per_step": 1000.0 * elapsed / steps,
        "nan": nan_count,
        "inf": inf_count,
    }
    if finite.size:
        chans = []
        for c in range(4):
            col = arr[..., c]
            col = col[np.isfinite(col)]
            if col.size == 0:
                chans.append(None)
            else:
                chans.append({
                    "min": float(col.min()),
                    "max": float(col.max()),
                    "mean": float(col.mean()),
                    "var": float(col.var()),
                })
        stats["channels"] = chans
    tex_a.release(); tex_b.release(); prog.release()
    return stats


# ---------------------------------------------------------------------------
# Summary across sizes: only flag true instabilities.
# ---------------------------------------------------------------------------
# "Stats drift between sizes" is mostly expected — 20 steps reach different
# fractions of the simulated time at different resolutions, random init hits
# different density regimes, etc. We only flag things that look like genuine
# numerical explosions: absolute values beyond any plausible saturation band,
# or a channel blowing up orders of magnitude between adjacent sizes.
EXPLOSION_THRESHOLD = 1e4   # any channel value > this is probably a blow-up
BLOWUP_RATIO        = 1e3   # variance ratio between adjacent sizes

def compare_scaling(per_size):
    """per_size = {size: result_dict}. Return list of warnings."""
    warns = []
    sizes = sorted([s for s, r in per_size.items() if "channels" in r])
    if len(sizes) < 2:
        return warns
    for c in range(4):
        # Flag absolute explosions
        for sz in sizes:
            ch = per_size[sz].get("channels", [None] * 4)[c]
            if ch is None:
                continue
            if max(abs(ch["min"]), abs(ch["max"])) > EXPLOSION_THRESHOLD:
                warns.append(
                    f"ch{c} @ size={sz}: values outside plausible range "
                    f"[{ch['min']:.3g}, {ch['max']:.3g}]"
                )
        # Flag giant variance jumps between adjacent sizes (numerical blow-up
        # that resolves at higher resolution = definite CFL-style bug)
        for i in range(len(sizes) - 1):
            a = per_size[sizes[i]].get("channels", [None] * 4)[c]
            b = per_size[sizes[i + 1]].get("channels", [None] * 4)[c]
            if a is None or b is None:
                continue
            va, vb = a["var"], b["var"]
            if va > 1.0 and vb > 1e-6 and va / vb > BLOWUP_RATIO:
                warns.append(
                    f"ch{c}: variance collapses {sizes[i]}→{sizes[i+1]} "
                    f"({va:.3g} → {vb:.3g}) — CFL-unstable at small grid?"
                )
    return warns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="32,64,128")
    ap.add_argument("--big", action="store_true",
                    help="Also test 256 & 384 & 512 (needs lots of VRAM)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--only", default="",
                    help="Comma-separated preset names to restrict to")
    ap.add_argument("--json", default="",
                    help="Write full results to this JSON path")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    if args.big:
        sizes = sorted(set(sizes + [256, 384, 512]))
    only = set(s.strip() for s in args.only.split(",") if s.strip())

    ctx = moderngl.create_standalone_context(require=430)
    print(f"GL renderer : {ctx.info.get('GL_RENDERER','?')}")
    print(f"GL vendor   : {ctx.info.get('GL_VENDOR','?')}")
    print(f"GL version  : {ctx.info.get('GL_VERSION','?')}")
    print(f"Sizes       : {sizes}")
    print(f"Steps/size  : {args.steps}")
    print(f"Presets     : {len(RULE_PRESETS)}" +
          (f" (filtered to {len(only)})" if only else ""))
    print()

    all_results = {}
    n_presets = sum(1 for n in RULE_PRESETS if not only or n in only)
    grand_t0 = time.perf_counter()
    idx = 0
    for preset_name, preset in RULE_PRESETS.items():
        if only and preset_name not in only:
            continue
        idx += 1
        per_size = {}
        line = f"[{idx:>2}/{n_presets}] {preset_name:32s}"
        sys.stdout.write(line); sys.stdout.flush()
        for sz in sizes:
            sys.stdout.write(f"  [{sz}…]"); sys.stdout.flush()
            t_one = time.perf_counter()
            try:
                r = run_once(ctx, preset_name, preset, sz, steps=args.steps)
            except Exception as e:
                r = {"error": f"harness exc: {e}"}
            per_size[sz] = r
            # Erase the "[sz…]" placeholder by overprinting
            sys.stdout.write("\b" * (len(f"  [{sz}…]")))
            if "error" in r:
                cell = f"  [{sz:>3}] ERR"
            elif r.get("nan", 0) or r.get("inf", 0):
                cell = f"  [{sz:>3}] NAN({r['nan']})/INF({r['inf']})"
            elif "skip" in r:
                cell = f"  [{sz:>3}] skip"
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

    # Exit code: non-zero if any NaN/Inf/errors detected
    bad = any(
        "error" in r or r.get("nan", 0) or r.get("inf", 0)
        for preset in all_results.values() for r in preset.values()
    )
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
