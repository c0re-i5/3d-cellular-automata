"""Canonical metric names — single source of truth.

Three things this module owns:

1.  CANONICAL: the name we store on disk, in parquet columns and JSON keys.
    Pick once, use everywhere. Any code that emits metrics imports from here.

2.  ALIASES: maps legacy names from each producer to their canonical name.
    `to_canonical(d)` rewrites a dict in place; loaders use this so old
    `debug_runs/*.json` and `discoveries.json` files stay readable.

3.  Schema enums (`MEASURE_MODES`, `EVENT_KINDS`, etc.) used by parquet
    dictionary encoding and event validation.

The previous fragmentation was the actual debugging blocker — same quantity
under three different names blocked any cross-tool query. So the rule now is:
*if it has a synonym anywhere in the codebase, it lives here, exactly once.*
"""

from __future__ import annotations

# ── Per-step canonical names (timeseries.parquet columns) ──────────────────
#
# Naming convention:
#   • snake_case, no abbreviations except `ch{N}` for channel index.
#   • counts are int (`*_count`), fractions float in [0,1] (`*_fraction`),
#     ratios float in [0,inf) (`*_ratio`), seconds float (`*_sec`).
#   • per-channel scalars get a `chN_` prefix; channel-agnostic scalars do
#     not. So mean-of-active-channel is `mean_active`, but mean of channel 0
#     is `ch0_mean`.

# --- liveness / activity (channel-agnostic, drives most analysis) ---
STEP                 = "step"           # int64, simulation step index
T_WALL               = "t_wall"         # float64, unix epoch sec when sample taken
ALIVE_COUNT          = "alive_count"    # int64, cells of measured channel above threshold
ALIVE_FRACTION       = "alive_fraction" # float64, alive_count / voxel_count, [0,1]
ACTIVITY             = "activity"       # float64, fraction of cells changed since last step
SURFACE_RATIO        = "surface_ratio"  # float64, boundary_count / max(alive_count, 1)
MEASURE_MODE         = "measure_mode"   # string, see MEASURE_MODES

# --- spatial decomposition of alive cells (gui debug today; bring to harness) ---
COM_X                = "com_x"          # float64, normalised [0,1]
COM_Y                = "com_y"
COM_Z                = "com_z"
BBOX_MIN_X           = "bbox_min_x"     # float64, normalised [0,1]; NaN if no alive
BBOX_MIN_Y           = "bbox_min_y"
BBOX_MIN_Z           = "bbox_min_z"
BBOX_MAX_X           = "bbox_max_x"
BBOX_MAX_Y           = "bbox_max_y"
BBOX_MAX_Z           = "bbox_max_z"
RG                   = "rg"             # float64, gyration radius / size
BOUNDARY_COUNT       = "boundary_count" # int64, alive cells with >=1 dead neighbour
BOUNDARY_FRACTION    = "boundary_fraction"

# --- per-channel scalars (4 channels = chan{0..3}) ---
def per_channel(prefix: str, n: int = 4) -> list[str]:
    """Generate ['ch0_<p>', 'ch1_<p>', ...] for n channels."""
    return [f"ch{i}_{prefix}" for i in range(n)]

# convenience lists
CH_MEAN     = per_channel("mean")     # float64
CH_STD      = per_channel("std")      # float64
CH_MIN      = per_channel("min")      # float64 (NaN if no finite)
CH_MAX      = per_channel("max")      # float64
CH_VAR      = per_channel("var")      # float64
CH_NAN      = per_channel("nan")      # int64
CH_INF      = per_channel("inf")      # int64
CH_FINITE   = per_channel("finite")   # int64

# --- aggregate health flags ---
HAS_NAN     = "has_nan"   # bool, any channel had a NaN this step
HAS_INF     = "has_inf"   # bool

TIMESERIES_REQUIRED = (STEP, T_WALL, ALIVE_COUNT, ALIVE_FRACTION)
TIMESERIES_OPTIONAL = (
    ACTIVITY, SURFACE_RATIO, MEASURE_MODE,
    COM_X, COM_Y, COM_Z,
    BBOX_MIN_X, BBOX_MIN_Y, BBOX_MIN_Z,
    BBOX_MAX_X, BBOX_MAX_Y, BBOX_MAX_Z,
    RG, BOUNDARY_COUNT, BOUNDARY_FRACTION,
    HAS_NAN, HAS_INF,
    *CH_MEAN, *CH_STD, *CH_MIN, *CH_MAX, *CH_VAR,
    *CH_NAN, *CH_INF, *CH_FINITE,
)

# ── Per-frame profiling (frames.parquet columns) ──────────────────────────
FRAME              = "frame"        # int64, monotonic frame index
FRAME_T_WALL       = "t_wall"       # float64, unix epoch sec
FRAME_STEP         = "step"         # int64, sim step current at frame end
SEC_POLL           = "sec_poll"     # float64, glfw + imgui input
SEC_STEP           = "sec_step"     # float64, CA shader dispatch
SEC_SCORE          = "sec_score"    # float64, score reduction (0 if not run)
SEC_RENDER         = "sec_render"   # float64, total render
SEC_UI             = "sec_ui"       # float64, imgui draw + Python panel code
SEC_SWAP           = "sec_swap"     # float64, swap_buffers (vsync wait)
SEC_TOTAL          = "sec_total"    # float64, full frame
# render sub-section keys arrive as `render.foo` from the simulator; we
# rewrite them to `sec_render__foo` for parquet. The double-underscore is
# parquet-safe and reversible: `col.removeprefix("sec_render__")`.
SEC_RENDER_SUB_PREFIX = "sec_render__"

FRAMES_REQUIRED = (FRAME, FRAME_T_WALL, SEC_TOTAL)


# ── Manifest fields (manifest.json) ───────────────────────────────────────
# Top-level keys in the run's manifest.json. Anything optional becomes None
# rather than being dropped, so the schema is stable across producers.
MANIFEST_FIELDS = (
    "schema_version",   # int, see schema.SCHEMA_VERSION
    "run_id",           # str, directory name
    "created_at",       # str, ISO8601
    "ended_at",         # str, ISO8601 or None while open
    "producer",         # one of PRODUCERS
    "rule",             # str
    "label",            # str, human-friendly
    "description",      # str, may be empty
    "size",             # int, cubic grid edge
    "dt",               # float
    "seed",             # int
    "init_variant",     # str or None
    "params",           # dict[str, number]
    "preset_keys",      # list[str], for replay validation
    "renderer_mode",    # str or None (None = headless)
    "colormap",         # str or None
    "code",             # dict: git_sha, git_dirty, branch
    "env",              # dict: python, platform, gpu_vendor, gpu_renderer, gl_version
    "tags",             # list[str], free-form labels (e.g. ["search", "lenia"])
)

PRODUCERS = ("gui", "harness", "snapshot-cli", "replay")


# ── Event log (events.jsonl) ──────────────────────────────────────────────
# One JSON object per line. Required keys: t (float, unix sec), step (int),
# kind (one of EVENT_KINDS). Other keys are kind-specific.
EVENT_KINDS = (
    "param_change",     # key, from, to
    "rule_change",      # from, to
    "renderer_mode",    # from, to
    "randomize",        # (no extra)
    "restart",          # (no extra)
    "snapshot",         # path
    "recording_start",  # path, fps, resolution
    "recording_stop",   # path, frames
    "discovery_save",   # path or index
    "anomaly",          # detail (e.g. "nan in ch2")
    "note",             # text (manual annotation)
)


# ── Measurement modes (per-step `measure_mode` column) ────────────────────
MEASURE_MODES = (
    "discrete",          # GoL-style 0/1 cells; alive = nonzero
    "continuous",        # Lenia-style; alive = ch > threshold
    "wave",              # signed field; alive = abs(ch) > threshold
    "deviation",         # alive = |ch - baseline| > threshold
    "phase_coherence",   # complex/orientation field; alive via coherence
    "element",           # element CA; alive = id != 0 and id != wall
)


# ── Aliases: legacy name → canonical name ─────────────────────────────────
# The two big synonym pairs that block cross-tool queries today.
# Add any historical name here so old debug_runs / discoveries / snapshots
# load cleanly into the new schema.
TIMESERIES_ALIASES = {
    # gui-debug used `active_*`, harness used `alive_*`. Canonical: `alive_*`.
    "active_count":   ALIVE_COUNT,
    "active_frac":    ALIVE_FRACTION,
    "alive_ratio":    ALIVE_FRACTION,
    "alive_frac":     ALIVE_FRACTION,
    # historical typo / earlier naming
    "boundary_frac":  BOUNDARY_FRACTION,
    # snapshot_3d.py series cols are already ch{N}_{stat} — match.
    "ch0_alive":      "ch0_alive_fraction",  # if ever stored, route to its own col
    "ch1_alive":      "ch1_alive_fraction",
    "ch2_alive":      "ch2_alive_fraction",
    "ch3_alive":      "ch3_alive_fraction",
}

FRAMES_ALIASES = {
    # F12 perf overlay used bare names; we prefix with sec_ so they don't
    # collide with timeseries columns when joined.
    "poll":   SEC_POLL,
    "step":   SEC_STEP,   # NB: collides with sim-step `step`; only valid in frames table
    "score":  SEC_SCORE,
    "render": SEC_RENDER,
    "ui":     SEC_UI,
    "swap":   SEC_SWAP,
    "total":  SEC_TOTAL,
}

# Discovery aliases. We're keeping discoveries.json structurally the same for
# now (it's an aggregate; the full data lives in the run bundle) — these are
# just for loaders that want to normalise against canonical per-step names.
DISCOVERY_AGGREGATE_ALIASES = {
    "final_alive":     "final_" + ALIVE_FRACTION,
    "final_activity":  "final_" + ACTIVITY,
    "final_surface":   "final_" + SURFACE_RATIO,
}


def to_canonical_timeseries(sample: dict) -> dict:
    """Rewrite legacy keys in a per-step sample dict to canonical names.

    Keys not in TIMESERIES_ALIASES pass through unchanged, so this is safe to
    call on already-canonical samples (idempotent) or on mixed legacy data.
    """
    out = {}
    for k, v in sample.items():
        out[TIMESERIES_ALIASES.get(k, k)] = v
    return out


def to_canonical_frame(sample: dict) -> dict:
    """Rewrite legacy section names + render.* sub-keys to canonical."""
    out = {}
    for k, v in sample.items():
        if k.startswith("render."):
            out[SEC_RENDER_SUB_PREFIX + k[len("render."):]] = v
        else:
            out[FRAMES_ALIASES.get(k, k)] = v
    return out
