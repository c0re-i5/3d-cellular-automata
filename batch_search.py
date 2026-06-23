#!/usr/bin/env python3
"""Interactive batch-search launcher for the 3D CA harness.

Pick rules + metrics + run parameters from a menu instead of memorising env
vars and rule names, then fan out parallel ``test_harness.py search`` jobs and
merge the winners into ``discoveries.json`` -- the exact same pipeline the
``batch_*.sh`` scripts drive.

Interactive:
    python batch_search.py

Non-interactive (scriptable, mirrors the batch_*.sh env knobs):
    python batch_search.py --rules flagships --metrics elegance,combined \
        --trials 500 --size 64 --steps 300 --top 50 --jobs 8

    python batch_search.py --rules "fcc_life,fcc_crystal" --dry-run
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.environ.get("PYTHON", sys.executable)
HARNESS = os.path.join(HERE, "test_harness.py")
MERGER = os.path.join(HERE, "batch_merge.py")

# Metric choices accepted by `test_harness.py search --metric` (score first =
# the sensible default).
METRICS = [
    "score", "combined", "elegance", "gol_coherence", "projection",
    "structure", "slice_mi", "period", "glider", "growth", "clusters",
    "symmetry", "gol_like",
]

# Ordered group predicates; first match wins, anything unmatched -> "other".
# Purely a selection aid -- you can always pick individual rules or "all".
GROUP_DEFS = [
    ("flagships", lambda n: n.startswith("flagship_")),
    ("fcc", lambda n: n.startswith("fcc_")),
    ("quantum", lambda n: n.startswith("quantum_")),
    ("crystals", lambda n: n.startswith("crystal") or "solidification" in n),
    ("lenia_nca", lambda n: "lenia" in n or n.startswith("nca_")),
    ("fractals", lambda n: n in {
        "mandelbulb_3d", "juliabulb_3d", "mandelbox_3d", "menger_3d"}),
    ("agents", lambda n: any(t in n for t in (
        "ant_colony", "wolf_sheep", "termite", "physarum", "langton",
        "smuggler", "predator_prey", "flock", "wandering_voxels"))),
    ("reaction_diffusion", lambda n: any(t in n for t in (
        "gray_scott", "schnakenberg", "brusselator", "fitzhugh", "bz_",
        "morphogen", "cyclic", "greenberg", "turing", "sine_gordon",
        "kuramoto", "active_"))),
    ("physics", lambda n: any(t in n for t in (
        "fluid", "euler", "wave", "dirac", "rayleigh", "smoke", "volcan",
        "em_wave", "fire", "erosion", "fracture", "viscous", "compressible",
        "phase_separation", "nucleation", "hopfion", "nematic"))),
    ("classic", lambda n: any(t in n for t in (
        "game_of_life", "445", "larger_than_life", "ising", "eden",
        "hodgepodge", "sandpile", "wireworld", "forest_fire", "prisoners",
        "xy_spin", "margolus", "smallworld", "life", "causal", "genome"))),
]


def load_rules():
    """Return the ordered list of rule names (source of truth: RULE_PRESETS)."""
    from simulator import RULE_PRESETS
    return list(RULE_PRESETS.keys())


def group_rules(rules):
    """Map rule list -> {group_name: [rules...]} preserving order."""
    groups: dict[str, list[str]] = {name: [] for name, _ in GROUP_DEFS}
    groups["other"] = []
    for r in rules:
        for name, pred in GROUP_DEFS:
            if pred(r):
                groups[name].append(r)
                break
        else:
            groups["other"].append(r)
    return {k: v for k, v in groups.items() if v}


# ── interactive selection ─────────────────────────────────────────────────

def _print_rule_catalog(rules, groups):
    idx = {r: i for i, r in enumerate(rules)}
    print("\nAvailable rules (pick by number, range a-b, group name, or 'all'):\n")
    for gname, grules in groups.items():
        print(f"  \033[1m{gname}\033[0m ({len(grules)})")
        line = "    "
        for r in grules:
            cell = f"[{idx[r]:>3}] {r}"
            if len(line) + len(cell) > 96:
                print(line)
                line = "    "
            line += cell + "  "
        if line.strip():
            print(line)
    print()


def parse_selection(expr, rules, groups):
    """Parse a selection expression into an ordered, de-duplicated rule list."""
    selected: list[str] = []
    seen = set()

    def add(r):
        if r not in seen:
            seen.add(r)
            selected.append(r)

    tokens = expr.replace(",", " ").split()
    for tok in tokens:
        low = tok.lower()
        if low == "all":
            for r in rules:
                add(r)
        elif low in groups:
            for r in groups[low]:
                add(r)
        elif "-" in tok and all(p.strip().isdigit() for p in tok.split("-", 1)):
            a, b = (int(p) for p in tok.split("-", 1))
            for i in range(min(a, b), max(a, b) + 1):
                if 0 <= i < len(rules):
                    add(rules[i])
        elif tok.isdigit():
            i = int(tok)
            if 0 <= i < len(rules):
                add(rules[i])
        elif tok in rules:
            add(tok)
        else:
            print(f"  ! ignored unknown token: {tok!r}")
    return selected


def select_rules_interactive(rules, groups):
    _print_rule_catalog(rules, groups)
    while True:
        expr = input("Select rules > ").strip()
        if not expr:
            print("  (nothing entered -- type e.g. 'flagships' or '0-11' or 'all')")
            continue
        chosen = parse_selection(expr, rules, groups)
        if not chosen:
            print("  (no valid rules matched, try again)")
            continue
        preview = ", ".join(chosen[:8]) + (" ..." if len(chosen) > 8 else "")
        ans = input(f"  -> {len(chosen)} rules: {preview}\n  "
                    f"Proceed? [Y]es / [e]dit > ").strip().lower()
        if ans in ("", "y", "yes"):
            return chosen


def select_metrics_interactive():
    print("\nMetrics to optimise (search runs once per rule x metric):\n")
    for i, m in enumerate(METRICS):
        print(f"  [{i:>2}] {m}")
    raw = input("\nSelect metrics (numbers/names, blank = score) > ").strip()
    if not raw:
        return ["score"]
    out, seen = [], set()
    for tok in raw.replace(",", " ").split():
        m = None
        if tok.isdigit() and 0 <= int(tok) < len(METRICS):
            m = METRICS[int(tok)]
        elif tok.lower() in METRICS:
            m = tok.lower()
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out or ["score"]


def ask_int(prompt, default):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  ! enter a whole number")


# ── orchestration ─────────────────────────────────────────────────────────

def build_jobs(rules, metrics, args, tmp_dir, log_dir):
    """Return a list of job dicts: cmd + bookkeeping per (rule, metric)."""
    jobs = []
    for metric in metrics:
        for rule in rules:
            seed = random.randint(1, (1 << 30) - 1)
            tmpfile = os.path.join(tmp_dir, f"{rule}_{metric}.json")
            logfile = os.path.join(log_dir, f"{rule}_{metric}.log")
            cmd = [
                PYTHON, HARNESS,
                "--size", str(args.size),
                "--steps", str(args.steps),
                "--seed", str(seed),
                "search", rule,
                "--trials", str(args.trials),
                "--metric", metric,
                "--top", str(args.top),
                "--save", tmpfile,
            ]
            jobs.append({"rule": rule, "metric": metric, "cmd": cmd,
                         "logfile": logfile, "desc": f"{rule}/{metric}"})
    return jobs


def run_jobs(jobs, max_jobs):
    total = len(jobs)
    queue = list(jobs)
    running = {}  # Popen -> (desc, filehandle)
    done = 0
    failed = []
    print(f"\nLaunching {total} jobs, {max_jobs} concurrent "
          f"(Ctrl-C to abort)...\n")
    try:
        while queue or running:
            while queue and len(running) < max_jobs:
                job = queue.pop(0)
                fh = open(job["logfile"], "w")
                p = subprocess.Popen(job["cmd"], stdout=fh,
                                     stderr=subprocess.STDOUT)
                running[p] = (job["desc"], fh)
                print(f"  launch  {job['desc']}")
            time.sleep(0.3)
            for p in list(running):
                rc = p.poll()
                if rc is None:
                    continue
                desc, fh = running.pop(p)
                fh.close()
                done += 1
                if rc == 0:
                    print(f"  [{done}/{total}] done  {desc}")
                else:
                    failed.append(desc)
                    print(f"  [{done}/{total}] FAIL  {desc} (rc={rc})")
    except KeyboardInterrupt:
        print("\n[abort] terminating workers...")
        for p in list(running):
            p.terminate()
        deadline = time.time() + 4
        for p in list(running):
            try:
                p.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
        for _desc, fh in running.values():
            fh.close()
        raise
    return failed


def main():
    ap = argparse.ArgumentParser(
        description="Interactive / scriptable batch-search launcher.")
    ap.add_argument("--rules", default=None,
                    help="Comma/space rule selection (names, group names, or "
                         "'all'). Skips the interactive rule picker.")
    ap.add_argument("--metrics", default=None,
                    help="Comma/space metric list. Skips the metric picker.")
    ap.add_argument("--trials", type=int, default=None, help="Trials per job")
    ap.add_argument("--size", type=int, default=None, help="Grid size")
    ap.add_argument("--steps", type=int, default=None, help="Sim steps")
    ap.add_argument("--top", type=int, default=None, help="Top-N pool per job")
    ap.add_argument("--jobs", type=int, default=None, help="Concurrent jobs")
    ap.add_argument("--out", default="discoveries.json",
                    help="Merge target (default: discoveries.json)")
    ap.add_argument("--no-merge", action="store_true",
                    help="Skip the merge step (leave per-job JSON in tmp dir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and exit without running anything.")
    args = ap.parse_args()

    # Defaults match the batch_*.sh env knobs.
    defaults = {"trials": 500, "size": 64, "steps": 300, "top": 50,
                "jobs": int(os.environ.get("MAX_JOBS", 8))}

    rules = load_rules()
    groups = group_rules(rules)
    interactive = sys.stdin.isatty() and args.rules is None

    # Resolve rule selection.
    if args.rules is not None:
        sel_rules = parse_selection(args.rules, rules, groups)
        if not sel_rules:
            sys.exit("No valid rules in --rules selection.")
    elif interactive:
        sel_rules = select_rules_interactive(rules, groups)
    else:
        sys.exit("Non-interactive: pass --rules (e.g. --rules flagships).")

    # Resolve metrics.
    if args.metrics is not None:
        sel_metrics = [m.strip() for m in args.metrics.replace(",", " ").split()
                       if m.strip() in METRICS]
        if not sel_metrics:
            sys.exit("No valid metrics in --metrics selection.")
    elif interactive:
        sel_metrics = select_metrics_interactive()
    else:
        sel_metrics = ["score"]

    # Resolve numeric params (flag > interactive prompt > default).
    for key in ("trials", "size", "steps", "top", "jobs"):
        val = getattr(args, key)
        if val is None:
            label = {"trials": "Trials per job", "size": "Grid size",
                     "steps": "Sim steps", "top": "Top-N pool",
                     "jobs": "Concurrent jobs"}[key]
            val = ask_int(label, defaults[key]) if interactive else defaults[key]
        setattr(args, key, val)

    n_jobs = len(sel_rules) * len(sel_metrics)
    print("\n" + "=" * 60)
    print("  BATCH SEARCH PLAN")
    print("=" * 60)
    print(f"  rules    : {len(sel_rules)}  ({', '.join(sel_rules[:6])}"
          f"{' ...' if len(sel_rules) > 6 else ''})")
    print(f"  metrics  : {', '.join(sel_metrics)}")
    print(f"  trials   : {args.trials}   size: {args.size}   "
          f"steps: {args.steps}   top: {args.top}")
    print(f"  jobs     : {n_jobs} total, {args.jobs} concurrent")
    print(f"  merge -> : {'(skipped)' if args.no_merge else args.out}")
    print("=" * 60)

    if args.dry_run:
        print("\n[dry-run] no jobs executed.")
        return

    if interactive:
        if input("\nStart? [Y/n] > ").strip().lower() not in ("", "y", "yes"):
            print("Aborted.")
            return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_dir = os.path.join(HERE, f"tmp_search_{stamp}")
    log_dir = os.path.join(HERE, "logs", f"search_{stamp}")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(args.out):
        with open(args.out, "w") as f:
            f.write("[]")

    t0 = time.time()
    aborted = False
    try:
        failed = run_jobs(build_jobs(sel_rules, sel_metrics, args, tmp_dir,
                                     log_dir), args.jobs)
    except KeyboardInterrupt:
        aborted = True
        failed = []

    dt = time.time() - t0
    print(f"\nFinished {n_jobs} jobs in {dt:.0f}s. "
          f"{len(failed)} failed." if not aborted else
          f"\nAborted after {dt:.0f}s.")
    if failed:
        print("  failed: " + ", ".join(failed))
        print(f"  logs:   {log_dir}")

    if not args.no_merge and not aborted:
        print("\nMerging winners...")
        subprocess.run([PYTHON, MERGER, "--tmp-dir", tmp_dir, "--out", args.out,
                        "--filter-blinkers", "--rules", " ".join(sel_rules)],
                       check=False)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Done -> {args.out}  (logs: {log_dir})")
    else:
        print(f"\nPer-job results kept in: {tmp_dir}")


if __name__ == "__main__":
    main()
