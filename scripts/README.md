# Debugging & diagnostic tools

Reusable tools for chasing native crashes (segfaults, heap corruption,
use-after-free) in `simulator.py`. Kept in the tree on purpose — these paid
off once and will again. `faulthandler` is armed inside `simulator.py`, so
Python-level dumps always land in `logs/crash.log`.

## Case study: the intermittent GUI segfault (fixed 2026-07-04)

Browsing the discovery list — especially rapid `<` / `>` arrow clicks —
would occasionally segfault (or throw a nonsense `UnboundLocalError` on
`self`) somewhere in `_draw_ui`.

**Root cause:** a UI button called `_load_discovery` → `_change_rule` →
`_compile_compute` **while the imgui frame was open**. `_compile_compute`
`glDeleteProgram`s the old shader while it is still `GL_CURRENT_PROGRAM`.
imgui's pure-Python PyOpenGL backend saved that program id at the start of
`render()` and calls `glUseProgram(<deleted>)` when restoring GL state at the
end → `GLError 1281`, and intermittently a use-after-free of a freed PyObject
(the freed 64-byte block got recycled into a discovery-list `PyFloat`, whose
`ob_type` pointer was then poison `0xdd…`).

**Fix:** a deferred-action queue. `run()` sets `self._in_imgui_frame = True`
around `_draw_ui`, clears it before `imgui.render()`, and drains
`self._deferred_frame_actions` **after** `imgui_renderer.render()`. The seven
UI entry points that recompile shaders / realloc GL resources
(`_load_discovery`, `_change_rule`, `_change_size`, `_reset`,
`_randomize_params`, `_mutate_params`, `_load_best_for_current_rule`) enqueue
themselves instead of running when called mid-frame.

**What actually caught it:** `CA_HEAP_MODE=pydebug` (pymalloc redzones +
`0xdd` free-poison) run under `stdbuf -o0 -e0`, plus a `gdb` core dump showing
`obj->ob_type == 0xdddddddddddddddd`. valgrind was too slow (~2 FPS) to reach
the trigger interactively.

**Rule of thumb learned:** never mutate GL / free GL resources mid-imgui-frame
while they're GL-current. Defer to after `render()`.

---

## `run_debug_simulator.sh` — heap-corruption detector + core dumps

Launches the GUI with a sharp allocator so corruption faults *at* the bad
operation, and enables core dumps for `gdb`.

```bash
./scripts/run_debug_simulator.sh --size 96                 # glibc malloc mode
CA_HEAP_MODE=pydebug stdbuf -o0 -e0 \
    ./scripts/run_debug_simulator.sh --size 96             # pymalloc redzones
```

- `CA_HEAP_MODE=malloc` (default): `PYTHONMALLOC=malloc` + `MALLOC_CHECK_=3`
  + `MALLOC_PERTURB_`. Aborts at the corrupting `free()`/`realloc()`.
- `CA_HEAP_MODE=pydebug`: `PYTHONMALLOC=debug`. Redzones every Python alloc
  and poisons freed memory with `0xdd`; ~2× slowdown but fully interactive.
  **Run under `stdbuf -o0 -e0`** or the "Fatal Python error" line is lost to
  glibc block-buffering when stderr is redirected to a file.
- On crash a core lands at repo root as `./core` (or `core.<pid>`):
  `gdb .venv/bin/python ./core` then `bt full`. Cores are ~2 GB — delete
  them after inspecting.

## `CA_AUTOBROWSE=1` — hands-off in-app repro driver

In-simulator driver (in `simulator.py`'s `run()` loop) that steps through
saved discoveries automatically — the exact "browse the list" action — so a
crash can be reproduced unattended. Knobs:

| env var                 | default | meaning                            |
|-------------------------|---------|------------------------------------|
| `CA_AUTOBROWSE_RULE`    | `<all>` | filter to a single rule name       |
| `CA_AUTOBROWSE_STEPS`   | `2`     | sim steps per batch (`sim_speed`)  |
| `CA_AUTOBROWSE_DWELL`   | `200`   | sim steps spent per discovery      |
| `CA_AUTOBROWSE_SWEEPS`  | `3`     | full passes before auto-quit       |
| `CA_AUTOBROWSE_REVERSE` | `1`     | newest-first (match UI sort)       |

```bash
CA_AUTOBROWSE=1 CA_AUTOBROWSE_RULE=game_of_life_3d \
    ./scripts/run_debug_simulator.sh --size 96
```

Note: this driver calls `_load_discovery` **outside** the imgui frame, so it
did *not* reproduce the mid-frame bug above — that needed real arrow clicks.
Still useful for soak-testing rule switching and GL resource churn.

## `CA_NO_GL_FENCE=1` — fence kill switch

Disables the raw-PyOpenGL fence trio (`glFenceSync`/`glClientWaitSync`/
`glDeleteSync`) in the per-frame hot loop. Was a suspect for the segfault
(disproven — kept default-off as a quick A/B toggle for GL-sync issues).

## `run_valgrind_simulator.sh` + `nvidia.supp` — memcheck

Runs the GUI under valgrind memcheck to name the exact line of an invalid
write/read/use-after-free. ~20–50× slowdown (unusable for interactive repro
under the NVIDIA GL driver, but authoritative when a crash is easy to hit).
`nvidia.supp` filters driver/OpenBLAS/pyarrow noise; extend it with stacks
from `--gen-suppressions=all`.

## `stress_rule_switch.py` — headless rule-switch soak

Hammers `_load_discovery` → `_change_rule` (shader recompile + grid resize +
reset) headlessly under a sharp allocator to shake out GL-object
leak/double-free/use-after-free without a window.

```bash
MPLBACKEND=Agg PYTHONMALLOC=malloc PYTHONFAULTHANDLER=1 PYTHONPATH="$PWD" \
    .venv/bin/python -u scripts/stress_rule_switch.py --iters 4000
```

## `probe_debug_stats_headless.py` — debug-stats crash bisector

Reproduces the headless `_dispatch_debug_stats` / `_harvest_debug_stats`
(PyOpenGL fence + moderngl SSBO read) path behind fault markers to isolate
which native call segfaults (the `refine.py` pass-A exit `-11`).

```bash
.venv/bin/python -u scripts/probe_debug_stats_headless.py
```
