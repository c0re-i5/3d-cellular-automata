#!/usr/bin/env python3
"""
Compare crystal_growth vs crystal_dendritic debug runs.

The user's complaint: dendritic visually looks the same as growth (a round
blob) instead of producing branches. We use the GPU debug-stats history
already saved in debug_runs/*.json to test whether dendritic is actually
geometrically distinct from growth, or whether it really is the same blob.

Key geometric tests (no rendering needed):
  1. SPHERICITY = rg / rg_sphere(volume)
        rg_sphere = sqrt(3/5) * R, R = (3V/4π)^(1/3) in lattice units
        normalized so V = active_count, R in voxel units, rg in [0,1]
        of-box units --> rg_sphere_norm = sqrt(3/5)*R/size
        Sphericity = 1 -> compact ball
        Sphericity > 1.5 -> arms/branches reaching out
  2. BBOX_FILL = active_count / bbox_volume
        1.0 -> bbox is solid (compact)
        << 1 -> sparse (branched/skeletal)
  3. INTERFACE_FRAC = (cells with phi in (0.05,0.95)) / active_count
        ratio of front-cells to bulk-cells. Compact blob: surface/volume ~ N^(-1/3).
        Branched: much higher because the branched shape has more surface area.
        We can read this from the histogram.
"""
import json
import glob
import math
import argparse
import os
import sys


def sphericity(active_count, rg, size):
    """rg / rg_sphere_of_same_volume.  1 = compact ball. >1.5 = branched."""
    if active_count <= 0:
        return float("nan")
    # Volume in voxels -> radius in voxels.
    R_vox = (3.0 * active_count / (4.0 * math.pi)) ** (1.0 / 3.0)
    rg_sphere_vox = math.sqrt(3.0 / 5.0) * R_vox
    rg_sphere_norm = rg_sphere_vox / size
    return rg / rg_sphere_norm


def bbox_fill(active_count, bbox_min, bbox_max, size):
    extents = [(bbox_max[i] - bbox_min[i]) * size for i in range(3)]
    bbox_vol = max(1.0, extents[0] * extents[1] * extents[2])
    return active_count / bbox_vol


def interface_frac(hist_phi, hist_min, hist_max, n_voxels):
    """Fraction of the *non-empty* phi-histogram that's at intermediate values.

    The phi histogram has 64 bins between hist_min and hist_max. We compute
    'interface' as bins where 0.05 < phi < 0.95, and 'bulk' as phi >= 0.95.
    The denominator is bulk + interface (active cells).
    """
    nbins = len(hist_phi)
    bin_edges = [hist_min + (hist_max - hist_min) * i / nbins for i in range(nbins + 1)]
    interface = 0
    bulk = 0
    for i in range(nbins):
        # bin spans [bin_edges[i], bin_edges[i+1])
        center = 0.5 * (bin_edges[i] + bin_edges[i + 1])
        if center >= 0.95:
            bulk += hist_phi[i]
        elif center > 0.05:
            interface += hist_phi[i]
    total = interface + bulk
    if total == 0:
        return float("nan")
    return interface / total


def analyze(path):
    d = json.load(open(path))
    sz = d["size"]
    n_voxels = sz ** 3
    rule = d["rule"]
    params = d["params"]
    rows = []
    for h in d["history"]:
        active = h["active_count"]
        if active < 5:
            continue
        sph = sphericity(active, h["rg"], sz)
        bf = bbox_fill(active, h["bbox_min"], h["bbox_max"], sz)
        iff = interface_frac(h["hist"][0], h["hist_min"][0], h["hist_max"][0], n_voxels)
        rows.append(
            dict(
                step=h["step"],
                active_frac=h["active_frac"],
                rg=h["rg"],
                sphericity=sph,
                bbox_fill=bf,
                interface_frac=iff,
                phi_mean=h["mean"][0],
                u_mean=h["mean"][1],
                u_min=h["min"][1],
                u_max=h["max"][1],
            )
        )
    return rule, sz, params, rows


def fmt_row(r):
    return (
        f"  step={r['step']:5d}  af={r['active_frac']:.3f}  rg={r['rg']:.3f}  "
        f"sph={r['sphericity']:.2f}  bbfill={r['bbox_fill']:.3f}  "
        f"iface={r['interface_frac']:.3f}  phi_m={r['phi_mean']:.3f}  "
        f"u=[{r['u_min']:.2f},{r['u_max']:.2f}] um={r['u_mean']:.3f}"
    )


def summarize_run(path):
    rule, sz, params, rows = analyze(path)
    print(f"\n=== {os.path.basename(path)} ===")
    print(f"  rule={rule}  size={sz}")
    print(f"  params: U={params.get('Undercooling'):.3f}  D={params.get('Diffusion'):.3f}  "
          f"A={params.get('Anisotropy strength'):.3f}  Shape={int(params.get('Shape', 0))}")
    if not rows:
        print("  (no active cells)")
        return rows
    # Show first, mid, late, final
    n = len(rows)
    pick = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    for i in pick:
        print(fmt_row(rows[i]))
    return rows


def find_match(dendritic_rows, growth_rows, key="active_frac"):
    """For each dendritic snapshot, find the growth snapshot whose active_frac
    is closest, and tabulate what differs.
    """
    g_sorted = sorted(growth_rows, key=lambda r: r[key])
    out = []
    for d_row in dendritic_rows:
        target = d_row[key]
        best = min(growth_rows, key=lambda g: abs(g[key] - target))
        if abs(best[key] - target) > 0.05:
            continue
        out.append((d_row, best))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    growths = sorted(glob.glob(f"debug_runs/crystal_growth_{args.size}_*.json"))
    dendritics = sorted(glob.glob(f"debug_runs/crystal_dendritic_{args.size}_*.json"))

    if not growths or not dendritics:
        print(f"need growth + dendritic runs at size {args.size}", file=sys.stderr)
        return 1

    print("=" * 90)
    print(f"BASELINE: crystal_growth runs (size {args.size})")
    print("=" * 90)
    growth_rows_all = []
    for g in growths:
        rows = summarize_run(g)
        growth_rows_all.extend(rows)

    print()
    print("=" * 90)
    print(f"TEST: crystal_dendritic runs (size {args.size})")
    print("=" * 90)
    dend_runs = []
    for d in dendritics:
        rows = summarize_run(d)
        dend_runs.append((d, rows))

    print()
    print("=" * 90)
    print("MATCHED COMPARISON: at equal active_frac, do dendritic & growth differ?")
    print("=" * 90)
    print("If sphericity, bbox_fill, and interface_frac match within ~5%,")
    print("then dendritic is geometrically indistinguishable from growth.")
    print()

    for (d_path, d_rows) in dend_runs:
        if not d_rows:
            continue
        # Pick three representative active_fracs from this dendritic run.
        targets = [0.05, 0.20, 0.50]
        print(f"\n  --- {os.path.basename(d_path)} ---")
        print(f"  {'af':>6} | dendritic (sph/bbfill/iface) | growth (sph/bbfill/iface) | delta")
        for t in targets:
            d_row = min(d_rows, key=lambda r: abs(r["active_frac"] - t))
            if abs(d_row["active_frac"] - t) > 0.10:
                continue
            g_row = min(growth_rows_all, key=lambda g: abs(g["active_frac"] - d_row["active_frac"]))
            d_sph, d_bf, d_if = d_row["sphericity"], d_row["bbox_fill"], d_row["interface_frac"]
            g_sph, g_bf, g_if = g_row["sphericity"], g_row["bbox_fill"], g_row["interface_frac"]
            print(
                f"  {d_row['active_frac']:.3f}  | "
                f"{d_sph:5.2f} / {d_bf:5.3f} / {d_if:5.3f}        | "
                f"{g_sph:5.2f} / {g_bf:5.3f} / {g_if:5.3f}        | "
                f"Δsph={d_sph-g_sph:+.2f} Δbf={d_bf-g_bf:+.3f} Δif={d_if-g_if:+.3f}"
            )

    print()
    print("Reference values:")
    print("  sphericity = 1.00  -> perfect ball (rg = sqrt(3/5)*R)")
    print("  sphericity > 1.30  -> visibly aspherical / branched / planar")
    print("  bbox_fill  = 0.52  -> ball inscribed in cube")
    print("  bbox_fill  < 0.30  -> sparse / branched / dendritic")
    print("  iface_frac ~ N^-1/3 ~ 0.10-0.20 for compact blob")
    print("  iface_frac > 0.40  -> high surface/volume (branched)")


if __name__ == "__main__":
    raise SystemExit(main())
