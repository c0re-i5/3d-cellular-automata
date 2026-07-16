#!/usr/bin/env bash
# Launch simulator.py with heap-corruption detection + native crash capture.
#
# The intermittent segfault is heap corruption from a C extension
# (moderngl / imgui-bundle / numpy): faulthandler only shows the Python
# frame where the interpreter happened to be, not the native call that
# corrupted the heap. This wrapper makes the corruption fault *at the
# corrupting operation* and leaves a core dump so gdb can show the real
# C-level backtrace.
#
#   PYTHONMALLOC=malloc  — use glibc malloc; pymalloc's arena reuse hides
#                          overflows/use-after-free and misattributes them.
#   MALLOC_CHECK_=3      — glibc aborts (with a message) at the corrupting
#                          free()/realloc() instead of crashing randomly later.
#   MALLOC_PERTURB_      — fill freed memory with a poison byte so
#                          use-after-free reads produce obviously-wrong data
#                          and fault sooner.
#   ulimit -c unlimited  — write a core dump on crash for gdb.
#
# After a crash (core is written to the repo root as ./core):
#   gdb .venv/bin/python ./core
#   (gdb) bt full          # native backtrace -> the extension at fault
#
# Python-level dumps still land in logs/crash.log (faulthandler is armed
# inside simulator.py).

set -euo pipefail
cd "$(dirname "$0")/.."

ulimit -c unlimited || true

# Heap-corruption detector selection via CA_HEAP_MODE:
#   malloc  (default) — glibc malloc + MALLOC_CHECK_=3 + MALLOC_PERTURB_.
#                       Catches corruption at free()/realloc(). pymalloc's
#                       arena reuse is bypassed so overflows aren't hidden.
#   pydebug           — PYTHONMALLOC=debug: keep pymalloc but add forbidden
#                       -byte redzones before/after every Python allocation,
#                       checked on each alloc/free. Aborts with a precise
#                       "bad trailing pad byte" diagnostic the moment a
#                       Python object's block is overrun (e.g. a float64 OOB
#                       write past an adjacent object). ~2x slowdown, fully
#                       interactive — unlike valgrind's ~2 FPS.
CA_HEAP_MODE="${CA_HEAP_MODE:-malloc}"
if [[ "$CA_HEAP_MODE" == "pydebug" ]]; then
    export PYTHONMALLOC=debug
    unset MALLOC_CHECK_ MALLOC_PERTURB_ 2>/dev/null || true
    echo "[debug-run] CA_HEAP_MODE=pydebug (PYTHONMALLOC=debug, pymalloc redzones)" >&2
else
    export PYTHONMALLOC=malloc
    export MALLOC_CHECK_=3
    export MALLOC_PERTURB_=$((RANDOM % 255 + 1))
    echo "[debug-run] CA_HEAP_MODE=malloc PYTHONMALLOC=malloc MALLOC_CHECK_=3 MALLOC_PERTURB_=${MALLOC_PERTURB_}" >&2
fi
export PYTHONFAULTHANDLER=1

# Hypothesis test: set CA_NO_GL_FENCE=1 to disable the raw-PyOpenGL fence
# trio (glFenceSync/glClientWaitSync/glDeleteSync) — the only non-moderngl
# GL calls in the per-frame hot loop and the prime heap-corruption suspect.
# Invoke as:  CA_NO_GL_FENCE=1 ./scripts/run_debug_simulator.sh
# If extended browsing no longer segfaults, the fence mix is the corruptor.
export CA_NO_GL_FENCE="${CA_NO_GL_FENCE:-0}"

echo "[debug-run] CA_NO_GL_FENCE=${CA_NO_GL_FENCE}" >&2
echo "[debug-run] core dumps enabled (ulimit -c = $(ulimit -c))" >&2
echo "[debug-run] on crash: gdb .venv/bin/python ./core  then  bt full" >&2

exec .venv/bin/python -u simulator.py "$@"
