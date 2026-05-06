"""Differential testing harness: run the audit across a parameter matrix.

Sweeps test_harness audit across a grid of {sizes × seeds × steps} so the
analyzer / smell / property layers have enough data to distinguish noise
from real bugs.

    # Default matrix: 2 sizes × 5 seeds × 1 step count = 10 audits ≈ 2 min.
    python -m ca_debug.matrix run

    # Custom matrix:
    python -m ca_debug.matrix run --sizes 32 48 64 --seeds 1 2 3 --steps 80 200

    # Dry-run prints the planned commands only:
    python -m ca_debug.matrix run --dry-run

    # Quick post-hoc summary of how many bundles cover each cell:
    python -m ca_debug.matrix coverage

The harness resets the runs/ tree by default (so the matrix is the *only*
data the smell report sees) — pass ``--keep`` to append instead. Bundles
are tagged ``matrix`` so future analyses can filter on tag.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import schema as S


def _audit_cmd(size: int, seed: int, steps: int, *,
               keep_runs: int) -> list[str]:
    """Construct the audit invocation. Inherits CA_DEBUG_KEEP_RUNS via env."""
    return [
        sys.executable, "test_harness.py",
        "--record",
        "--size", str(size),
        "--seed", str(seed),
        "--steps", str(steps),
        "audit",
    ]


def run_matrix(sizes: list[int], seeds: list[int], steps_list: list[int],
               *, keep: bool = False, dry_run: bool = False,
               keep_runs_cap: int = 5000) -> int:
    """Run audit for every (size, seed, steps) combination.

    Returns the number of audit invocations completed.
    """
    runs_dir = Path(S.DEFAULT_RUNS_ROOT)
    if not keep and runs_dir.exists() and not dry_run:
        print(f"[matrix] removing existing {runs_dir}/ ({sum(1 for _ in runs_dir.iterdir())} entries)")
        shutil.rmtree(runs_dir)

    env = os.environ.copy()
    env["CA_DEBUG_KEEP_RUNS"] = str(keep_runs_cap)

    n_total = len(sizes) * len(seeds) * len(steps_list)
    print(f"[matrix] planning {n_total} audit runs "
          f"(sizes={sizes}, seeds={seeds}, steps={steps_list})")

    n_done = 0
    t0 = time.time()
    for steps in steps_list:
        for size in sizes:
            for seed in seeds:
                cmd = _audit_cmd(size, seed, steps, keep_runs=keep_runs_cap)
                tag = f"size={size} seed={seed} steps={steps}"
                if dry_run:
                    print(f"  [dry-run] {tag}: {' '.join(cmd)}")
                    n_done += 1
                    continue
                t1 = time.time()
                print(f"  [{n_done+1}/{n_total}] {tag} ...", end=" ", flush=True)
                try:
                    subprocess.run(cmd, env=env, check=True,
                                   capture_output=True, text=True)
                    dur = time.time() - t1
                    print(f"ok ({dur:.1f}s)")
                    n_done += 1
                except subprocess.CalledProcessError as e:
                    print("FAIL")
                    print(f"    stderr tail: {e.stderr[-400:]}")
                    raise

    if not dry_run:
        total = time.time() - t0
        print(f"[matrix] done in {total:.1f}s ({n_done} audits, "
              f"{total/max(n_done,1):.1f}s avg)")
    return n_done


def coverage(runs_root: str = S.DEFAULT_RUNS_ROOT) -> None:
    """Print a coverage table: how many bundles per (rule, size) cell."""
    from . import analyzer
    df = analyzer.sql("""
        select rule, size, seed, count(*) as n
        from runs
        group by rule, size, seed
        order by rule, size, seed
    """, runs_root=runs_root)
    if df.empty:
        print(f"no runs in {runs_root}/")
        return
    pivot = df.pivot_table(
        index="rule", columns="size", values="n", aggfunc="sum",
        fill_value=0,
    )
    print(f"=== bundle coverage in {runs_root} ===")
    print(f"  {len(df['rule'].unique())} rules × {len(df['size'].unique())} sizes × "
          f"{df['seed'].nunique()} seeds = {len(df)} cells "
          f"({df['n'].sum()} bundles total)")
    print()
    print(pivot.to_string())


# ── CLI ───────────────────────────────────────────────────────────────
def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="ca_debug.matrix",
        description="Differential testing matrix runner.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="execute the matrix")
    p_run.add_argument("--sizes", type=int, nargs="+", default=[32, 64])
    p_run.add_argument("--seeds", type=int, nargs="+",
                       default=[7, 42, 101, 202, 303])
    p_run.add_argument("--steps", type=int, nargs="+", default=[80])
    p_run.add_argument("--keep", action="store_true",
                       help="append to existing runs/ (default: wipe first)")
    p_run.add_argument("--dry-run", action="store_true",
                       help="print planned commands without executing")
    p_run.add_argument("--keep-runs", type=int, default=5000,
                       help="CA_DEBUG_KEEP_RUNS cap (default: 5000)")

    sub.add_parser("coverage", help="show coverage of existing runs/")

    args = ap.parse_args()
    if args.cmd == "run":
        run_matrix(args.sizes, args.seeds, args.steps,
                   keep=args.keep, dry_run=args.dry_run,
                   keep_runs_cap=args.keep_runs)
    elif args.cmd == "coverage":
        coverage()


if __name__ == "__main__":
    _cli()
