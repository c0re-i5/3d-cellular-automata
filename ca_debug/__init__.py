"""Unified debug & data-capture infrastructure for the CA simulator.

This package replaces three previously-disjoint debug systems:
    - GUI F11 debug overlay (`debug_runs/*.json`)
    - GUI F12 perf overlay  (`perf_runs/*.json`)
    - test_harness trial logs + discoveries.json
    - snapshot_3d.py voxel dumps (`snapshots/*.npz`)

with a single "run bundle" on disk:

    runs/<run_id>/
        manifest.json          one-time run-level facts
        timeseries.parquet     per-step CA metrics (canonical schema)
        frames.parquet         per-frame profiling (gui only)
        snapshots/tNNNNNN.npz  full (W,H,D,4) voxel dumps
        events.jsonl           chronological state changes
        derived.json           post-hoc dynamics analyses
        thumbnail.png          final-state preview
        recording.mp4          optional video

Modules:
    metrics    - canonical metric names + legacy-name aliases
    schema     - parquet/manifest schemas + version constants
    recorder   - RunRecorder: the single writer used by GUI + harness + CLI
    analyses   - pure-numpy structure / dynamics analyses
"""

from . import metrics, schema, analyses  # noqa: F401
from .recorder import RunRecorder  # noqa: F401

__all__ = ["metrics", "schema", "analyses", "RunRecorder"]
