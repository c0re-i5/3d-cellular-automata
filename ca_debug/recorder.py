"""RunRecorder — single writer for all CA debug data.

Used by the GUI, the test harness, the snapshot CLI, and the replay tool.
Owns a single `runs/<run_id>/` directory; everything any tool wants to
persist about a run goes through this object.

Design:
    * Writes are buffered in memory and flushed in batches. The default
      flush threshold is 256 timeseries rows / 512 frame rows / immediately
      for events. `flush()` forces all pending writes; `close()` flushes
      and stamps `ended_at` in the manifest.
    * Parquet files are written *append-style* by accumulating record
      batches and rewriting on flush. For very long runs this would matter,
      but we cap timeseries at sample_interval >= 1 step so even a 100k-step
      run is < 1 MB of parquet — a full rewrite is cheap and avoids the
      complexity of streaming parquet writers across resize boundaries.
    * Schema is fixed (see schema.timeseries_arrow_schema). Producers that
      only have a subset of columns pass partial dicts; missing columns are
      written as nulls.

Public surface:
    rec = RunRecorder.create(rule="lenia_3d", size=64, dt=0.1, seed=42, ...)
    rec.log_step(step, {"alive_count": 512, "alive_fraction": 0.5, ...})
    rec.log_frame(frame, {"sec_total": 0.016, "sec_render": 0.008, ...})
    rec.log_event("param_change", step, key="kr", from_=13, to=15)
    rec.snapshot(step, voxels_array, dtype_native="float32")
    rec.set_derived({"period_score": 0.83, ...})
    rec.write_thumbnail(rgb_array)
    rec.close()
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from . import metrics as M
from . import schema as S


def _utcnow_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class RunRecorder:
    """A single writer pointed at one runs/<run_id>/ directory.

    Construct via `create(...)` (new run) or `open(...)` (existing run, e.g.
    when the GUI launches and adopts a run dir already on disk).

    Thread-safety: a single internal lock guards the in-memory buffers. Safe
    to call any log_* / snapshot / set_derived / flush / close from any
    thread; the underlying parquet rewrite is not parallelised, so very hot
    callers (e.g. per-frame from the render thread) should still favour one
    writer thread.
    """

    # ── construction ─────────────────────────────────────────────────────
    def __init__(self, run_dir: Path, manifest: dict[str, Any]):
        self.run_dir: Path = run_dir
        self.manifest: dict[str, Any] = dict(manifest)
        self._lock = threading.Lock()
        self._closed = False

        # in-memory row buffers; flushed to parquet on threshold or close
        self._ts_rows: list[dict[str, Any]] = []
        self._fr_rows: list[dict[str, Any]] = []
        # extra render-sub keys observed; we widen the frames schema lazily
        self._render_subkeys: set[str] = set()

        # events.jsonl is append-only and tiny → no buffering, just open file
        self._events_path = run_dir / S.EVENTS_NAME
        self._events_fp = self._events_path.open("a", buffering=1)  # line-buffered

        # flush thresholds (tunable later if needed)
        self.ts_flush_every = 256
        self.fr_flush_every = 512

    @classmethod
    def create(
        cls,
        *,
        rule: str,
        size: int,
        dt: float,
        seed: int,
        params: dict[str, Any],
        producer: str = "gui",
        runs_root: str | os.PathLike[str] = S.DEFAULT_RUNS_ROOT,
        label: str | None = None,
        description: str = "",
        init_variant: str | None = None,
        preset_keys: Iterable[str] | None = None,
        renderer_mode: str | None = None,
        colormap: str | None = None,
        tags: Iterable[str] | None = None,
        gl_ctx: Any | None = None,
        run_id: str | None = None,
    ) -> "RunRecorder":
        """Allocate a fresh runs/<run_id>/ dir and write the initial manifest."""
        if producer not in M.PRODUCERS:
            raise ValueError(f"producer must be one of {M.PRODUCERS}, got {producer!r}")
        run_id = run_id or S.make_run_id(rule)
        run_dir = Path(runs_root) / run_id
        (run_dir / S.SNAPSHOT_DIR_NAME).mkdir(parents=True, exist_ok=True)

        manifest = {
            "schema_version": S.SCHEMA_VERSION,
            "run_id":         run_id,
            "created_at":     _utcnow_iso(),
            "ended_at":       None,
            "producer":       producer,
            "rule":           rule,
            "label":          label or rule,
            "description":    description,
            "size":           int(size),
            "dt":             float(dt),
            "seed":           int(seed),
            "init_variant":   init_variant,
            "params":         dict(params),
            "preset_keys":    list(preset_keys or ()),
            "renderer_mode":  renderer_mode,
            "colormap":       colormap,
            "code":           S.capture_code_metadata(),
            "env":            S.capture_env_metadata(gl_ctx),
            "tags":           list(tags or ()),
        }
        rec = cls(run_dir, manifest)
        rec._write_manifest()
        return rec

    @classmethod
    def open(cls, run_dir: str | os.PathLike[str]) -> "RunRecorder":
        """Re-open an existing run directory (e.g. for replay or appending)."""
        path = Path(run_dir)
        manifest = json.loads((path / S.MANIFEST_NAME).read_text())
        return cls(path, manifest)

    # ── manifest mutation ────────────────────────────────────────────────
    def update_manifest(self, **kw) -> None:
        """Merge keys into manifest and rewrite. Use sparingly (e.g. set tags
        once a run is classified, or update label after a rename).
        """
        with self._lock:
            self.manifest.update(kw)
            self._write_manifest()

    def _write_manifest(self) -> None:
        tmp = self.run_dir / (S.MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2, default=_json_default))
        tmp.replace(self.run_dir / S.MANIFEST_NAME)

    # ── per-step timeseries ──────────────────────────────────────────────
    def log_step(self, step: int, sample: dict[str, Any], *, t_wall: float | None = None) -> None:
        """Append one per-step row. Unknown legacy keys are normalised via
        `metrics.to_canonical_timeseries` so harness/debug-overlay/snapshot
        callers can pass whatever they already produce.
        """
        if self._closed:
            raise RuntimeError("RunRecorder is closed")
        canon = M.to_canonical_timeseries(sample)
        canon[M.STEP] = int(step)
        canon[M.T_WALL] = float(t_wall if t_wall is not None else time.time())
        with self._lock:
            self._ts_rows.append(canon)
            if len(self._ts_rows) >= self.ts_flush_every:
                self._flush_timeseries_locked()

    def log_frame(self, frame: int, sample: dict[str, Any], *,
                  step: int | None = None, t_wall: float | None = None) -> None:
        """Append one per-frame profiling row.

        `sample` may contain bare 'render', 'poll', etc. (legacy F12 names) or
        canonical 'sec_render' / 'sec_poll'. Render sub-keys ('render.foo')
        are rewritten to 'sec_render__foo' so they're parquet-safe and can
        coexist as separate columns.
        """
        if self._closed:
            raise RuntimeError("RunRecorder is closed")
        canon = M.to_canonical_frame(sample)
        canon[M.FRAME] = int(frame)
        canon[M.FRAME_T_WALL] = float(t_wall if t_wall is not None else time.time())
        if step is not None:
            canon[M.FRAME_STEP] = int(step)
        # remember any new sub-keys for the schema widening
        for k in canon:
            if k.startswith(M.SEC_RENDER_SUB_PREFIX):
                self._render_subkeys.add(k[len(M.SEC_RENDER_SUB_PREFIX):])
        with self._lock:
            self._fr_rows.append(canon)
            if len(self._fr_rows) >= self.fr_flush_every:
                self._flush_frames_locked()

    # ── events ───────────────────────────────────────────────────────────
    def log_event(self, kind: str, step: int | None = None, **fields) -> None:
        """Append one event line. `kind` should be one of metrics.EVENT_KINDS;
        unknown kinds are accepted (logged with a warning field) so adding a
        new event type doesn't require a schema change.
        """
        if self._closed:
            raise RuntimeError("RunRecorder is closed")
        ev: dict[str, Any] = {
            "t":    time.time(),
            "step": int(step) if step is not None else None,
            "kind": str(kind),
        }
        # `from` is reserved in Python so callers pass `from_`; rewrite.
        if "from_" in fields:
            fields["from"] = fields.pop("from_")
        ev.update(fields)
        if kind not in M.EVENT_KINDS:
            ev["_unknown_kind"] = True
        line = json.dumps(ev, default=_json_default)
        with self._lock:
            self._events_fp.write(line + "\n")

    # ── snapshots (full voxel grids) ─────────────────────────────────────
    def snapshot(self, step: int, voxels, *, dtype_native: str | None = None,
                 extra_meta: dict[str, Any] | None = None) -> Path:
        """Write voxels (W,H,D,C) to snapshots/tNNNNNN.npz and log an event.

        Stored in float16 to keep size down; pass `dtype_native` to record the
        true GPU dtype (e.g. 'float32') in the per-snapshot meta.
        """
        import numpy as np
        path = self.run_dir / S.SNAPSHOT_DIR_NAME / S.SNAPSHOT_FMT.format(step=int(step))
        meta = {
            "schema_version": S.SCHEMA_VERSION,
            "run_id":         self.manifest["run_id"],
            "rule":           self.manifest["rule"],
            "step":           int(step),
            "seed":           self.manifest["seed"],
            "size":           self.manifest["size"],
            "dims":           list(voxels.shape[:3]),
            "channels":       int(voxels.shape[3]) if voxels.ndim == 4 else 1,
            "dtype_native":   dtype_native or str(voxels.dtype),
            "params":         self.manifest["params"],
            "init_variant":   self.manifest.get("init_variant"),
            "timestamp":      _utcnow_iso(),
        }
        if extra_meta:
            meta.update(extra_meta)
        arr = voxels.astype(np.float16, copy=False)
        np.savez_compressed(path, voxels=arr, meta=json.dumps(meta, default=_json_default))
        self.log_event("snapshot", step=step, path=str(path.relative_to(self.run_dir)))
        return path

    # ── derived analyses & artifacts ─────────────────────────────────────
    def set_derived(self, derived: dict[str, Any]) -> None:
        """Overwrite derived.json with the supplied dict. Idempotent."""
        path = self.run_dir / S.DERIVED_NAME
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(derived, indent=2, default=_json_default))
        tmp.replace(path)

    def write_thumbnail(self, rgb) -> Path:
        """Write a small PNG preview. `rgb` is an HxWx3 uint8 array."""
        from PIL import Image
        path = self.run_dir / S.THUMBNAIL_NAME
        Image.fromarray(rgb).save(path)
        return path

    def recording_path(self) -> Path:
        """Canonical path for the run's video. Caller is responsible for
        actually writing the file; we just expose the location so the
        existing ffmpeg pipeline can target it.
        """
        return self.run_dir / S.RECORDING_NAME

    # ── flush / close ────────────────────────────────────────────────────
    def flush(self) -> None:
        with self._lock:
            self._flush_timeseries_locked()
            self._flush_frames_locked()
            self._events_fp.flush()

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._flush_timeseries_locked()
            self._flush_frames_locked()
            try:
                self._events_fp.flush()
                self._events_fp.close()
            except Exception:  # noqa: BLE001  teardown, never fatal
                pass
            self.manifest["ended_at"] = _utcnow_iso()
            self._write_manifest()
            self._closed = True

    def __enter__(self):  # context-manager sugar for short-lived runs
        return self
    def __exit__(self, *exc):
        self.close()
        return False

    # ── parquet flushers (called under self._lock) ───────────────────────
    def _flush_timeseries_locked(self) -> None:
        if not self._ts_rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = S.timeseries_arrow_schema()
        new_table = _rows_to_table(self._ts_rows, schema)
        path = self.run_dir / S.TIMESERIES_NAME
        if path.exists():
            existing = pq.read_table(path)
            # cast existing to current schema in case columns were added
            existing = existing.cast(schema, safe=False) if existing.schema != schema else existing
            new_table = pa.concat_tables([existing, new_table])
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(new_table, tmp, compression="zstd")
        tmp.replace(path)
        self._ts_rows.clear()

    def _flush_frames_locked(self) -> None:
        if not self._fr_rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = S.frames_arrow_schema(sorted(self._render_subkeys))
        new_table = _rows_to_table(self._fr_rows, schema)
        path = self.run_dir / S.FRAMES_NAME
        if path.exists():
            existing = pq.read_table(path)
            # widen older table if schema picked up new render subkeys
            if existing.schema != schema:
                existing = _widen_table(existing, schema)
            new_table = pa.concat_tables([existing, new_table])
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(new_table, tmp, compression="zstd")
        tmp.replace(path)
        self._fr_rows.clear()


# ── helpers ────────────────────────────────────────────────────────────
def _rows_to_table(rows: list[dict[str, Any]], schema):
    """Build an arrow Table from row dicts, padding missing columns with None.

    We go field-by-field so a row missing an optional column doesn't raise;
    pyarrow's from_pylist requires homogenous keys, which we don't have.
    """
    import pyarrow as pa
    cols = {field.name: [r.get(field.name) for r in rows] for field in schema}
    return pa.Table.from_pydict(cols, schema=schema)


def _widen_table(table, target_schema):
    """Add any columns present in target_schema but missing from `table`,
    filled with nulls. Used when a new render-sub key appears mid-run."""
    import pyarrow as pa
    have = set(table.schema.names)
    missing = [f for f in target_schema if f.name not in have]
    if not missing:
        return table.cast(target_schema, safe=False)
    n = table.num_rows
    for f in missing:
        nulls = pa.nulls(n, type=f.type)
        table = table.append_column(f, nulls)
    # reorder to match target
    return table.select([f.name for f in target_schema]).cast(target_schema, safe=False)


def _json_default(o):
    """Make numpy scalars + Path objects JSON-serialisable."""
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return list(o)
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")
