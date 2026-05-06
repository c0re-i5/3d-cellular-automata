"""Read-side helpers for inspecting run bundles produced by RunRecorder.

Designed for two audiences:

  1.  Notebooks / scripts:
          from ca_debug.analyzer import load_run, load_runs, sql
          run = load_run("runs/20260506_080124_lenia_3d_abc123")
          run.timeseries.plot.line(x="step", y="alive_fraction")

          all_lenia = load_runs("runs/*lenia*")
          best = (all_lenia.timeseries
                    .groupby("run_id").alive_fraction.max()
                    .nlargest(10))

  2.  Ad-hoc CLI:
          python -m ca_debug.analyzer ls
          python -m ca_debug.analyzer show runs/20260506_...
          python -m ca_debug.analyzer sql "select rule, count(*) from runs group by rule"

Everything is lazy where it can be:
  * `Run.timeseries` reads the parquet on first access, then caches.
  * `RunSet.timeseries` concatenates only the runs you actually iterate.
  * `sql(query)` runs DuckDB directly against `runs/**/timeseries.parquet`
    via globbing — no concatenation needed for "select where rule=..." style
    queries that benefit from predicate pushdown.

Keeps the dependency surface to (pyarrow, pandas, duckdb) — all already
required for the recorder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Iterator

from . import schema as S


# ── Single-run handle ──────────────────────────────────────────────────
@dataclass
class Run:
    """Lazy view of one runs/<run_id>/ directory.

    Properties read parquet/json on first access then memoise. Snapshot
    grids are loaded explicitly via `snapshot(step)` since they're
    typically the largest artifact and rarely needed.
    """
    path: Path

    # populated lazily; do not access directly — use the properties
    _manifest: dict[str, Any] | None = None
    _ts: Any = None  # pandas.DataFrame
    _frames: Any = None
    _events: list[dict[str, Any]] | None = None
    _derived: dict[str, Any] | None = None

    # ── basic identifiers ────────────────────────────────────────────
    @property
    def run_id(self) -> str:
        return self.path.name

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            mp = self.path / S.MANIFEST_NAME
            self._manifest = json.loads(mp.read_text()) if mp.exists() else {}
        return self._manifest

    @property
    def rule(self) -> str:
        return self.manifest.get("rule", "?")

    @property
    def producer(self) -> str:
        return self.manifest.get("producer", "?")

    @property
    def seed(self) -> int | None:
        return self.manifest.get("seed")

    @property
    def is_complete(self) -> bool:
        """True if `close()` was called (ended_at stamped)."""
        return self.manifest.get("ended_at") is not None

    # ── time-series tables ───────────────────────────────────────────
    @property
    def timeseries(self):
        """pandas.DataFrame of per-step CA metrics; empty if file missing."""
        if self._ts is None:
            self._ts = _read_parquet_df(self.path / S.TIMESERIES_NAME)
        return self._ts

    @property
    def frames(self):
        """pandas.DataFrame of per-frame profiling; empty if file missing."""
        if self._frames is None:
            self._frames = _read_parquet_df(self.path / S.FRAMES_NAME)
        return self._frames

    @property
    def events(self) -> list[dict[str, Any]]:
        """Parsed events.jsonl as a list of dicts (chronological)."""
        if self._events is None:
            ep = self.path / S.EVENTS_NAME
            if not ep.exists():
                self._events = []
            else:
                self._events = [
                    json.loads(line) for line in ep.read_text().splitlines() if line.strip()
                ]
        return self._events

    @property
    def derived(self) -> dict[str, Any]:
        """Post-hoc analyses (period, translation, clusters, ...)."""
        if self._derived is None:
            dp = self.path / S.DERIVED_NAME
            self._derived = json.loads(dp.read_text()) if dp.exists() else {}
        return self._derived

    # ── snapshots & artifacts ────────────────────────────────────────
    def snapshot_steps(self) -> list[int]:
        """Sorted list of steps for which a snapshot was written."""
        out = []
        for p in (self.path / S.SNAPSHOT_DIR_NAME).glob("t*.npz"):
            try:
                out.append(int(p.stem[1:]))  # 't000050' → 50
            except ValueError:
                continue
        return sorted(out)

    def snapshot(self, step: int):
        """Load a snapshot. Returns (voxels: ndarray, meta: dict)."""
        import numpy as np
        path = self.path / S.SNAPSHOT_DIR_NAME / S.SNAPSHOT_FMT.format(step=int(step))
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as npz:
            voxels = npz["voxels"][:]  # materialise; npz closes after with-block
            meta = json.loads(str(npz["meta"]))
        return voxels, meta

    @property
    def thumbnail_path(self) -> Path | None:
        p = self.path / S.THUMBNAIL_NAME
        return p if p.exists() else None

    @property
    def recording_path(self) -> Path | None:
        p = self.path / S.RECORDING_NAME
        return p if p.exists() else None

    # ── repr / summary ───────────────────────────────────────────────
    def __repr__(self) -> str:
        m = self.manifest
        n_steps = len(self.timeseries) if (self.path / S.TIMESERIES_NAME).exists() else 0
        n_snaps = len(self.snapshot_steps())
        flag = "" if self.is_complete else " [open]"
        return (f"<Run {self.run_id} rule={m.get('rule')} size={m.get('size')} "
                f"steps={n_steps} snaps={n_snaps}{flag}>")

    def summary(self) -> dict[str, Any]:
        """Headline numbers for `ls`-style output. Cheap (manifest + counts)."""
        ts = self.timeseries
        return {
            "run_id":     self.run_id,
            "rule":       self.rule,
            "size":       self.manifest.get("size"),
            "seed":       self.seed,
            "producer":   self.producer,
            "n_steps":    len(ts),
            "n_snaps":    len(self.snapshot_steps()),
            "n_events":   len(self.events),
            "complete":   self.is_complete,
            "created_at": self.manifest.get("created_at"),
        }


# ── Multi-run handle ───────────────────────────────────────────────────
class RunSet:
    """A collection of runs, with cross-run join helpers.

    The "joined" tables (`timeseries`, `frames`) are concatenations with a
    `run_id` column prepended, so groupby-style analyses work directly:

        rs = load_runs("runs/*")
        rs.timeseries.groupby("run_id")["alive_fraction"].max()
    """

    def __init__(self, runs: list[Run]):
        self.runs: list[Run] = runs
        self._ts_joined = None
        self._fr_joined = None

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self) -> Iterator[Run]:
        return iter(self.runs)

    def __getitem__(self, key) -> "Run | RunSet":
        if isinstance(key, slice):
            return RunSet(self.runs[key])
        if isinstance(key, int):
            return self.runs[key]
        # treat as run_id substring match
        matches = [r for r in self.runs if key in r.run_id]
        if len(matches) == 1:
            return matches[0]
        return RunSet(matches)

    def filter(self, **kw) -> "RunSet":
        """Manifest-field equality filter. e.g. `rs.filter(rule="lenia_3d")`."""
        out = []
        for r in self.runs:
            m = r.manifest
            if all(m.get(k) == v for k, v in kw.items()):
                out.append(r)
        return RunSet(out)

    @property
    def manifests(self):
        """pandas.DataFrame of one row per run with manifest fields flattened.

        Manifest params/code/env are flattened to `param.*`/`code.*`/`env.*`.
        Derived metrics (from derived.json) are merged at the top level so
        common queries like `select score from runs` work directly.
        """
        import pandas as pd
        rows = []
        for r in self.runs:
            m = dict(r.manifest)
            # flatten params/code/env into prefixed cols so they're queryable
            params = m.pop("params", {}) or {}
            code   = m.pop("code", {}) or {}
            env    = m.pop("env", {}) or {}
            for k, v in params.items(): m[f"param.{k}"] = v
            for k, v in code.items():   m[f"code.{k}"] = v
            for k, v in env.items():    m[f"env.{k}"] = v
            # Merge derived at top level. Manifest keys win on collision so
            # immutable run identity (rule, size, seed) can't be shadowed.
            try:
                d = r.derived or {}
            except Exception:
                d = {}
            for k, v in d.items():
                # Only merge JSON-scalar / list values; nested dicts would
                # confuse pandas type inference.
                if isinstance(v, (int, float, str, bool, type(None))):
                    m.setdefault(k, v)
            rows.append(m)
        return pd.DataFrame(rows)

    @property
    def timeseries(self):
        """All runs' timeseries concat'd with a `run_id` column."""
        if self._ts_joined is None:
            import pandas as pd
            parts = []
            for r in self.runs:
                df = r.timeseries
                if df is None or len(df) == 0:
                    continue
                df = df.copy()
                df.insert(0, "run_id", r.run_id)
                parts.append(df)
            self._ts_joined = (pd.concat(parts, ignore_index=True)
                               if parts else pd.DataFrame())
        return self._ts_joined

    @property
    def frames(self):
        if self._fr_joined is None:
            import pandas as pd
            parts = []
            for r in self.runs:
                df = r.frames
                if df is None or len(df) == 0:
                    continue
                df = df.copy()
                df.insert(0, "run_id", r.run_id)
                parts.append(df)
            self._fr_joined = (pd.concat(parts, ignore_index=True)
                               if parts else pd.DataFrame())
        return self._fr_joined

    def __repr__(self) -> str:
        return f"<RunSet n={len(self.runs)}>"


# ── Top-level entry points ─────────────────────────────────────────────
def load_run(path: str | Path) -> Run:
    p = Path(path)
    if not (p / S.MANIFEST_NAME).exists():
        raise FileNotFoundError(f"No manifest at {p / S.MANIFEST_NAME}")
    return Run(path=p)


def load_runs(pattern: str = "runs/*") -> RunSet:
    """Load every run matching a glob.

    Silently skips directories without a manifest (in-progress dirs that
    crashed before their first flush, leftover scratch dirs).
    """
    paths = sorted(glob(pattern))
    runs = []
    for path in paths:
        p = Path(path)
        if (p / S.MANIFEST_NAME).exists():
            runs.append(Run(path=p))
    return RunSet(runs)


def sql(query: str, *, runs_root: str = S.DEFAULT_RUNS_ROOT):
    """Run a DuckDB query against the run bundles.

    Three virtual tables are exposed:

      * `runs`        — one row per run, manifest fields (params flattened).
      * `timeseries`  — UNION ALL of every runs/*/timeseries.parquet,
                        with a `run_id` column added.
      * `frames`      — same for frames.parquet.

    Returns a pandas.DataFrame. DuckDB pushes filters into parquet, so
    `select * from timeseries where run_id like '%lenia%'` only reads the
    matching files.
    """
    import duckdb
    con = duckdb.connect(":memory:")

    # `runs` view via the analyzer's own manifest reader (simpler than
    # cramming nested JSON parsing into a SQL CTE).
    rs = load_runs(f"{runs_root}/*")
    if len(rs) > 0:
        con.register("runs", rs.manifests)

    # Parquet views — globs let DuckDB skip files via predicate pushdown.
    # Each view is only created if at least one matching file exists, so a
    # runs/ tree with only headless runs (no frames.parquet) doesn't error.
    ts_glob = f"{runs_root}/*/{S.TIMESERIES_NAME}"
    fr_glob = f"{runs_root}/*/{S.FRAMES_NAME}"
    if glob(ts_glob):
        con.execute(f"""
            CREATE VIEW timeseries AS
            SELECT regexp_extract(filename, '{runs_root}/([^/]+)/', 1) AS run_id, *
            FROM read_parquet('{ts_glob}', filename=true, union_by_name=true)
        """)
    if glob(fr_glob):
        con.execute(f"""
            CREATE VIEW frames AS
            SELECT regexp_extract(filename, '{runs_root}/([^/]+)/', 1) AS run_id, *
            FROM read_parquet('{fr_glob}', filename=true, union_by_name=true)
        """)
    return con.execute(query).df()


# ── internals ──────────────────────────────────────────────────────────
def _read_parquet_df(path: Path):
    """Read a parquet file as pandas DataFrame; empty DF if missing."""
    import pandas as pd
    if not path.exists():
        return pd.DataFrame()
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pandas()


# ── CLI ────────────────────────────────────────────────────────────────
def _cli() -> None:
    import argparse, sys, textwrap
    ap = argparse.ArgumentParser(prog="ca_debug.analyzer",
        description="Inspect run bundles in runs/")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls", help="list runs (most recent first)")
    p_ls.add_argument("pattern", nargs="?", default="runs/*")
    p_ls.add_argument("--limit", type=int, default=20)

    p_show = sub.add_parser("show", help="dump summary + tail of a run")
    p_show.add_argument("path")
    p_show.add_argument("--tail", type=int, default=5,
                        help="show last N timeseries rows")

    p_sql = sub.add_parser("sql", help="run a DuckDB query")
    p_sql.add_argument("query")
    p_sql.add_argument("--runs-root", default=S.DEFAULT_RUNS_ROOT)

    args = ap.parse_args()

    if args.cmd == "ls":
        rs = load_runs(args.pattern)
        rows = [r.summary() for r in rs.runs[::-1][:args.limit]]
        if not rows:
            print(f"no runs match {args.pattern!r}")
            return
        # crude column-aligned print to avoid pulling tabulate
        cols = ["run_id", "rule", "size", "seed", "producer",
                "n_steps", "n_snaps", "n_events", "complete"]
        widths = [max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols]
        print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
        print("  ".join("-" * w for w in widths))
        for r in rows:
            print("  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))

    elif args.cmd == "show":
        run = load_run(args.path)
        print(f"=== {run.run_id} ===")
        for k, v in run.summary().items():
            print(f"  {k:>12s}: {v}")
        print(f"  manifest:")
        for k in ("rule", "label", "params", "code", "env", "tags"):
            print(f"    {k}: {run.manifest.get(k)}")
        ts = run.timeseries
        if len(ts):
            print(f"\n  last {min(args.tail, len(ts))} timeseries rows:")
            cols = [c for c in ("step", "alive_fraction", "activity",
                                "surface_ratio", "rg") if c in ts.columns]
            print(textwrap.indent(ts[cols].tail(args.tail).to_string(index=False), "    "))
        if run.events:
            print(f"\n  events ({len(run.events)}):")
            for e in run.events[-args.tail:]:
                print(f"    step={e.get('step')!s:>6}  {e['kind']}  "
                      f"{ {k: v for k, v in e.items() if k not in ('t','step','kind')} }")
        if run.derived:
            print(f"\n  derived: {run.derived}")

    elif args.cmd == "sql":
        df = sql(args.query, runs_root=args.runs_root)
        print(df.to_string(index=False) if len(df) else "(no rows)")


if __name__ == "__main__":
    _cli()
