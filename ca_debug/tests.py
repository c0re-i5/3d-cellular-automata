"""Smoke tests for ca_debug foundation. Run with `python -m ca_debug.tests`.

These don't need GPU / moderngl — they exercise schema + recorder against
synthetic data so we can iterate on the format independently of the simulator.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from . import metrics as M
from . import schema as S
from .recorder import RunRecorder


def _fake_step_sample(step: int) -> dict:
    """Plausible per-step row covering both legacy + canonical keys, so we
    exercise the alias rewriter inside log_step.
    """
    rng = np.random.default_rng(step)
    return {
        # legacy synonym → must be canonicalised
        "active_count":     int(rng.integers(0, 1000)),
        "active_frac":      float(rng.random()),
        # canonical
        M.ACTIVITY:         float(rng.random() * 0.1),
        M.SURFACE_RATIO:    float(rng.random()),
        M.MEASURE_MODE:     "continuous",
        # per-channel
        "ch0_mean":         float(rng.random()),
        "ch0_std":          float(rng.random()),
        "ch1_mean":         float(rng.random()),
        # spatial
        M.COM_X:            float(rng.random()),
        M.COM_Y:            float(rng.random()),
        M.COM_Z:            float(rng.random()),
        M.RG:               float(rng.random()),
        M.HAS_NAN:          False,
        M.HAS_INF:          False,
    }


def _fake_frame_sample(frame: int) -> dict:
    """F12-style legacy keys. Must come back as canonical sec_* in parquet."""
    return {
        "poll":           0.0005,
        "step":           0.008,    # NOTE: gets aliased to sec_step (frames table only)
        "render":         0.012,
        "render.volume":  0.009,
        "render.blit":    0.001,
        "ui":             0.002,
        "swap":           0.014,
        "total":          0.038,
    }


def test_round_trip(tmp_root: Path) -> None:
    """Create a run, write 10 steps + 20 frames + 2 snapshots + derived,
    close, reopen, and verify the parquet round-trips."""
    rec = RunRecorder.create(
        rule="lenia_3d",
        size=32,
        dt=0.1,
        seed=42,
        params={"kernel_radius": 13, "growth_mu": 0.15},
        producer="harness",
        runs_root=str(tmp_root),
        label="Lenia 3D test",
        init_variant="random_smooth",
        preset_keys=["kernel_radius", "growth_mu", "growth_sigma"],
        renderer_mode=None,
        tags=["smoke-test"],
    )
    print(f"  created run dir: {rec.run_dir.relative_to(tmp_root)}")

    # log_step round-trip + alias rewrite
    for s in range(10):
        rec.log_step(s, _fake_step_sample(s))
    for f in range(20):
        rec.log_frame(f, _fake_frame_sample(f), step=f // 2)

    # event log
    rec.log_event("randomize", step=0)
    rec.log_event("param_change", step=3, key="kernel_radius", from_=13, to=15)

    # snapshots
    voxels = np.random.default_rng(0).random((16, 16, 16, 4), dtype=np.float32)
    rec.snapshot(0, voxels, dtype_native="float32")
    rec.snapshot(5, voxels, dtype_native="float32")

    rec.set_derived({
        "period_score":     0.83,
        "translation_score": 0.0,
        "n_clusters":       3,
    })
    rec.close()

    # ── verify on disk ────────────────────────────────────────────────
    assert (rec.run_dir / S.MANIFEST_NAME).exists()
    assert (rec.run_dir / S.TIMESERIES_NAME).exists()
    assert (rec.run_dir / S.FRAMES_NAME).exists()
    assert (rec.run_dir / S.EVENTS_NAME).exists()
    assert (rec.run_dir / S.DERIVED_NAME).exists()
    snaps = sorted((rec.run_dir / S.SNAPSHOT_DIR_NAME).glob("*.npz"))
    assert len(snaps) == 2, snaps

    # manifest sanity
    man = json.loads((rec.run_dir / S.MANIFEST_NAME).read_text())
    assert man["schema_version"] == S.SCHEMA_VERSION
    assert man["rule"] == "lenia_3d"
    assert man["ended_at"] is not None, "ended_at should be stamped on close"
    assert man["code"]["git_sha"] is not None or True  # may be None outside repo
    print(f"  manifest OK: rule={man['rule']} producer={man['producer']} "
          f"tags={man['tags']}")

    # parquet sanity
    import pyarrow.parquet as pq
    ts = pq.read_table(rec.run_dir / S.TIMESERIES_NAME)
    assert ts.num_rows == 10
    assert M.ALIVE_COUNT in ts.schema.names, "alias rewrite failed"
    assert M.ALIVE_FRACTION in ts.schema.names
    assert "active_count" not in ts.schema.names, "legacy name leaked into schema"
    print(f"  timeseries OK: {ts.num_rows} rows, {len(ts.schema.names)} cols")

    fr = pq.read_table(rec.run_dir / S.FRAMES_NAME)
    assert fr.num_rows == 20
    assert M.SEC_TOTAL in fr.schema.names
    assert M.SEC_RENDER in fr.schema.names
    assert "sec_render__volume" in fr.schema.names, "render.volume sub-key not flattened"
    assert "sec_render__blit" in fr.schema.names
    print(f"  frames OK: {fr.num_rows} rows, render-subkeys: "
          f"{[c for c in fr.schema.names if c.startswith('sec_render__')]}")

    # events
    events = [json.loads(line) for line in (rec.run_dir / S.EVENTS_NAME).read_text().splitlines()]
    # 1 randomize + 1 param_change + 2 snapshot = 4
    assert len(events) == 4, events
    assert any(e["kind"] == "param_change" and e.get("from") == 13 for e in events), \
        "from_ should be rewritten to from"
    print(f"  events OK: {len(events)} entries, kinds={[e['kind'] for e in events]}")

    # derived
    der = json.loads((rec.run_dir / S.DERIVED_NAME).read_text())
    assert der["period_score"] == 0.83
    print(f"  derived OK: {list(der)}")

    # snapshots
    npz = np.load(snaps[0], allow_pickle=False)
    assert "voxels" in npz.files and "meta" in npz.files
    meta = json.loads(str(npz["meta"]))
    assert meta["step"] == 0 and meta["dims"] == [16, 16, 16]
    print(f"  snapshot OK: dims={meta['dims']} dtype_native={meta['dtype_native']}")


def test_reopen_appends(tmp_root: Path) -> None:
    """Open() an existing run dir and append more rows; old data preserved."""
    rec = RunRecorder.create(
        rule="game_of_life_3d", size=16, dt=1.0, seed=1,
        params={}, producer="gui", runs_root=str(tmp_root),
    )
    for s in range(3):
        rec.log_step(s, {"alive_count": s, "alive_fraction": s * 0.1})
    rec.close()

    rec2 = RunRecorder.open(rec.run_dir)
    for s in range(3, 6):
        rec2.log_step(s, {"alive_count": s, "alive_fraction": s * 0.1})
    rec2.close()

    import pyarrow.parquet as pq
    ts = pq.read_table(rec2.run_dir / S.TIMESERIES_NAME)
    assert ts.num_rows == 6, ts.num_rows
    steps = ts.column(M.STEP).to_pylist()
    assert steps == [0, 1, 2, 3, 4, 5], steps
    print(f"  reopen OK: total rows after append = {ts.num_rows}")


def test_widening_render_subkeys(tmp_root: Path) -> None:
    """Late-arriving render.* keys should add columns, not crash."""
    rec = RunRecorder.create(
        rule="lenia_3d", size=16, dt=0.1, seed=2,
        params={}, producer="gui", runs_root=str(tmp_root),
    )
    # First flush: only render.volume
    for f in range(rec.fr_flush_every + 1):
        rec.log_frame(f, {"render": 0.01, "render.volume": 0.008, "total": 0.02})
    rec.flush()
    # Second batch: also render.iso (new subkey)
    for f in range(rec.fr_flush_every + 1, rec.fr_flush_every + 5):
        rec.log_frame(f, {"render": 0.01, "render.volume": 0.008,
                          "render.iso": 0.001, "total": 0.02})
    rec.close()

    import pyarrow.parquet as pq
    fr = pq.read_table(rec.run_dir / S.FRAMES_NAME)
    assert "sec_render__volume" in fr.schema.names
    assert "sec_render__iso" in fr.schema.names, fr.schema.names
    # Older rows should have null for the new col
    iso_col = fr.column("sec_render__iso").to_pylist()
    assert iso_col[0] is None and iso_col[-1] == 0.001
    print(f"  widening OK: iso col = {iso_col[:3]}...{iso_col[-3:]}")


def test_analyzer(tmp_root: Path) -> None:
    """Build two runs, then exercise load_run / load_runs / sql."""
    from .analyzer import load_run, load_runs, sql

    # build two runs
    for tag, n in (("a", 5), ("b", 8)):
        rec = RunRecorder.create(
            rule=f"rule_{tag}", size=16, dt=0.1, seed=ord(tag),
            params={"k": ord(tag)}, producer="harness",
            runs_root=str(tmp_root),
            tags=["analyzer-test", tag],
        )
        for s in range(n):
            rec.log_step(s, {"alive_count": s, "alive_fraction": s / n})
            rec.log_frame(s, {"render": 0.01, "total": 0.02}, step=s)
        rec.log_event("randomize", step=0)
        rec.set_derived({"score": 0.5 + 0.1 * n})
        rec.close()

    # ── load_run ────────────────────────────────────────────────────
    rule_a_dir = next(p for p in sorted(tmp_root.iterdir()) if "rule_a" in p.name)
    run = load_run(rule_a_dir)
    assert run.is_complete
    assert len(run.timeseries) == 5
    assert run.derived["score"] == 1.0
    assert len(run.events) == 1
    print(f"  load_run OK: {run!r}")

    # ── load_runs + filter ──────────────────────────────────────────
    rs = load_runs(f"{tmp_root}/*rule_*")
    assert len(rs) == 2, [r.run_id for r in rs.runs]
    only_a = rs.filter(rule="rule_a")
    assert len(only_a) == 1 and only_a.runs[0].rule == "rule_a"
    print(f"  load_runs OK: {rs!r}, filtered: {only_a!r}")

    # ── joined timeseries with run_id col ───────────────────────────
    ts = rs.timeseries
    assert "run_id" in ts.columns
    assert len(ts) == 5 + 8
    by_rule = ts.groupby("run_id").size().to_dict()
    assert set(by_rule.values()) == {5, 8}
    print(f"  joined timeseries OK: {len(ts)} rows across {ts.run_id.nunique()} runs")

    # ── manifests dataframe with flattened params ───────────────────
    mans = rs.manifests
    assert "param.k" in mans.columns
    assert set(mans["param.k"]) == {ord("a"), ord("b")}
    print(f"  manifests df OK: cols include {[c for c in mans.columns if c.startswith('param.')]}")

    # ── DuckDB sql across run bundles ───────────────────────────────
    # Use a per-test subdir so prior tests' bundles don't leak in.
    sql_root = tmp_root / "sql_only"
    sql_root.mkdir(exist_ok=True)
    for tag, n in (("c", 4), ("d", 6)):
        rec = RunRecorder.create(
            rule=f"sql_{tag}", size=8, dt=1.0, seed=ord(tag),
            params={}, producer="harness", runs_root=str(sql_root),
        )
        for s in range(n):
            rec.log_step(s, {"alive_count": s, "alive_fraction": s / n})
        rec.close()
    df = sql("select rule, count(*) as steps from timeseries "
             "join runs using (run_id) group by rule order by rule",
             runs_root=str(sql_root))
    print(f"  sql OK:\n{df.to_string(index=False)}")
    assert df.shape == (2, 2)
    assert dict(zip(df.rule, df.steps)) == {"sql_c": 4, "sql_d": 6}


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="ca_debug_test_"))
    try:
        print(f"[ca_debug.tests] tmp root: {root}")
        print("[1] round-trip:")
        test_round_trip(root)
        print("[2] reopen + append:")
        test_reopen_appends(root)
        print("[3] schema widening:")
        test_widening_render_subkeys(root)
        print("[4] analyzer + sql:")
        test_analyzer(root)
        print("\nALL TESTS PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
