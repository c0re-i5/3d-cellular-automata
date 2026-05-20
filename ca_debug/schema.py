"""Schema constants + lightweight helpers for the run-bundle format.

Owns:
  * SCHEMA_VERSION  — bumped on any breaking change to manifest/parquet layout.
  * Filenames inside a run dir.
  * Helpers for capturing code/env metadata at run start.
  * `make_run_id`  — collision-resistant directory name.

Kept deliberately small. Most field-name knowledge lives in `metrics.py`.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import re
import secrets
import subprocess
import sys
from typing import Any

SCHEMA_VERSION = 1

# ── Filenames inside runs/<run_id>/ ─────────────────────────────────────
MANIFEST_NAME      = "manifest.json"
TIMESERIES_NAME    = "timeseries.parquet"
FRAMES_NAME        = "frames.parquet"
EVENTS_NAME        = "events.jsonl"
DERIVED_NAME       = "derived.json"
THUMBNAIL_NAME     = "thumbnail.png"
RECORDING_NAME     = "recording.mp4"
SNAPSHOT_DIR_NAME  = "snapshots"
SNAPSHOT_FMT       = "t{step:06d}.npz"  # zero-padded so filename sort = step sort

DEFAULT_RUNS_ROOT  = "runs"


# ── Run id ─────────────────────────────────────────────────────────────
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")

def _slug(s: str) -> str:
    return _SLUG_RE.sub("_", s).strip("_") or "run"

def make_run_id(rule: str, *, when: _dt.datetime | None = None) -> str:
    """Stable, collision-resistant directory name.

    Format:  YYYYMMDD_HHMMSS_<rule>_<6hex>
    The 6-hex tail prevents collisions when two runs of the same rule start
    in the same second (e.g. parallel harness workers). Sortable by name = by
    start time, with rule grouping intact for tab-completion.
    """
    when = when or _dt.datetime.now()
    return f"{when.strftime('%Y%m%d_%H%M%S')}_{_slug(rule)}_{secrets.token_hex(3)}"


# ── Code & env capture ─────────────────────────────────────────────────
def capture_code_metadata(repo_path: str | None = None) -> dict[str, Any]:
    """Return git sha + dirty flag + branch for the repo containing this file.

    Best-effort: returns zeros if git unavailable or not a repo. Never raises.
    """
    cwd = repo_path or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out: dict[str, Any] = {"git_sha": None, "git_dirty": None, "branch": None}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
        out["git_sha"] = sha
    except Exception:  # noqa: BLE001  git probe; missing .git or git binary is OK
        return out
    try:
        diff = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd,
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
        out["git_dirty"] = bool(diff)
    except Exception:  # noqa: BLE001  git probe; missing .git or git binary is OK
        pass
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
        out["branch"] = branch
    except Exception:  # noqa: BLE001  git probe; missing .git or git binary is OK
        pass
    return out


def capture_env_metadata(gl_ctx: Any | None = None) -> dict[str, Any]:
    """Return python/platform info plus optional GL info from a moderngl ctx.

    `gl_ctx` is an optional moderngl Context. We pull vendor/renderer/version
    from it if present; otherwise the GL fields are None (e.g. snapshot CLI
    that already finished and released its context).
    """
    out: dict[str, Any] = {
        "python":       sys.version.split()[0],
        "platform":     platform.platform(),
        "machine":      platform.machine(),
        "gpu_vendor":   None,
        "gpu_renderer": None,
        "gl_version":   None,
    }
    if gl_ctx is not None:
        try:  # moderngl Context.info → dict-like with these keys
            info = gl_ctx.info
            out["gpu_vendor"]   = info.get("GL_VENDOR")
            out["gpu_renderer"] = info.get("GL_RENDERER")
            out["gl_version"]   = info.get("GL_VERSION")
        except Exception:  # noqa: BLE001  GL info probe, may be unsupported
            pass
    return out


# ── pyarrow schema builders ────────────────────────────────────────────
# Built lazily so importing this module doesn't pull in pyarrow on systems
# that only need to read manifests (e.g. lightweight CI scripts).
def timeseries_arrow_schema():
    """pyarrow.Schema for timeseries.parquet — superset of all columns.

    A single fixed schema lets us concat parquet files across runs without
    type drift. Optional columns are nullable; producers that don't compute
    them write nulls.
    """
    import pyarrow as pa
    from . import metrics as M

    fields = [
        pa.field(M.STEP, pa.int64(), nullable=False),
        pa.field(M.T_WALL, pa.float64(), nullable=False),
        pa.field(M.ALIVE_COUNT, pa.int64(), nullable=True),
        pa.field(M.ALIVE_FRACTION, pa.float64(), nullable=True),
        pa.field(M.ACTIVITY, pa.float64(), nullable=True),
        pa.field(M.SURFACE_RATIO, pa.float64(), nullable=True),
        pa.field(M.MEASURE_MODE,
                 pa.dictionary(pa.int8(), pa.string()), nullable=True),
        pa.field(M.HAS_NAN, pa.bool_(), nullable=True),
        pa.field(M.HAS_INF, pa.bool_(), nullable=True),
        pa.field(M.COM_X, pa.float64(), nullable=True),
        pa.field(M.COM_Y, pa.float64(), nullable=True),
        pa.field(M.COM_Z, pa.float64(), nullable=True),
        pa.field(M.BBOX_MIN_X, pa.float64(), nullable=True),
        pa.field(M.BBOX_MIN_Y, pa.float64(), nullable=True),
        pa.field(M.BBOX_MIN_Z, pa.float64(), nullable=True),
        pa.field(M.BBOX_MAX_X, pa.float64(), nullable=True),
        pa.field(M.BBOX_MAX_Y, pa.float64(), nullable=True),
        pa.field(M.BBOX_MAX_Z, pa.float64(), nullable=True),
        pa.field(M.RG, pa.float64(), nullable=True),
        pa.field(M.BOUNDARY_COUNT, pa.int64(), nullable=True),
        pa.field(M.BOUNDARY_FRACTION, pa.float64(), nullable=True),
    ]
    # Per-channel scalars: 4 channels × {mean,std,min,max,var,nan,inf,finite}
    for stat, typ in (
        ("mean", pa.float64()), ("std", pa.float64()),
        ("min",  pa.float64()), ("max", pa.float64()),
        ("var",  pa.float64()),
        ("nan",  pa.int64()),   ("inf", pa.int64()),
        ("finite", pa.int64()),
    ):
        for col in M.per_channel(stat):
            fields.append(pa.field(col, typ, nullable=True))
    return pa.schema(fields)


def frames_arrow_schema(extra_render_subkeys: list[str] | None = None):
    """pyarrow.Schema for frames.parquet.

    Render sub-section keys are open-ended (different rules expose different
    sub-timings); pass any extras observed at recorder-init time. Unknown
    columns appearing later are still accepted by the writer — they just
    won't show in the documented schema.
    """
    import pyarrow as pa
    from . import metrics as M

    fields = [
        pa.field(M.FRAME, pa.int64(), nullable=False),
        pa.field(M.FRAME_T_WALL, pa.float64(), nullable=False),
        pa.field(M.FRAME_STEP, pa.int64(), nullable=True),
        pa.field(M.SEC_POLL, pa.float64(), nullable=True),
        pa.field(M.SEC_STEP, pa.float64(), nullable=True),
        pa.field(M.SEC_SCORE, pa.float64(), nullable=True),
        pa.field(M.SEC_RENDER, pa.float64(), nullable=True),
        pa.field(M.SEC_UI, pa.float64(), nullable=True),
        pa.field(M.SEC_SWAP, pa.float64(), nullable=True),
        pa.field(M.SEC_TOTAL, pa.float64(), nullable=True),
    ]
    for sub in extra_render_subkeys or ():
        fields.append(pa.field(M.SEC_RENDER_SUB_PREFIX + sub,
                               pa.float64(), nullable=True))
    return pa.schema(fields)
