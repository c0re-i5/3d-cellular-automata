# scratch — informal debug & audit tooling

Personal workflow scripts accumulated during development. Unlike the
formal probes one level up (`shader_lint`, `symmetry`, `coupling`,
`conservation`, `limits`) these are not maintained, not part of CI,
and may break when their target rule's preset changes.

Kept in the public repo as **examples of investigation methodology** —
how a specific bug was hunted, what statistics were probed, how a
new rule was validated against analytic expectations.

## What's here

| Script | Purpose |
|---|---|
| `density_audit.py` | Mirrors the GPU view shader's density math on the CPU and flags rules whose voxels saturate / vanish under their declared `vis_mode` |
| `particle_debug.py` | Reads the particle SSBO + deposit texture each frame to debug feedback-loop blow-ups |
| `audit_*.py` | Ad-hoc audits: fp16 precision, ghost cube artefacts, preset-tuning regressions, rule purposefulness |
| `validate_*.py` | One-off correctness checks against analytic / lattice solutions for specific rules (Ising, Langton, Margolus, fluid projection, predator/prey, etc.) |
| `probe_*.py` / `diag_*.py` / `compare_*.py` | Per-rule investigation scripts (mostly crystal growth dendritic-tip kinetics) |
| `sweep_crystals.py` | Parameter sweep harness for the five crystal-growth presets |
| `flag_strip.py` / `visualize_flagged.py` | Quick visual triage of audit-flagged rules |
| `ca_dashboard.py` | Rolling health dashboard (alive ratio / activity / NaN watch) over a discovery batch |
| `batch_debug_audit.py` / `batch_merge.py` | Batch orchestration for headless audit runs |
| `apply_preset_tuning.py` / `preset_tuning_audit.py` | Apply / audit a preset-overrides JSON layer (predates the inline-presets-are-authoritative refactor) |
| `bench_regression.py` | Step-time regression test against a baseline JSON |

## Running

Most assume the repo root is on `PYTHONPATH` and the `.venv` is active.
Read the docstring at the top of each file for usage — there is no
unified CLI.
