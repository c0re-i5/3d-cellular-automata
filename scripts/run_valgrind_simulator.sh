#!/usr/bin/env bash
# Run simulator.py under Valgrind memcheck to catch the float64 heap
# out-of-bounds write that corrupts a Python-object pointer (intermittent
# GUI segfault at _draw_ui, always surfacing in the discovery-list loop).
#
# WHY valgrind and not the core dump: the SIGSEGV fires later, wherever the
# interpreter next dereferences the poisoned pointer (decoupled from the
# corrupting write). Memcheck reports the WRITE itself — the exact line — the
# instant it lands in a redzone or freed/quarantined block.
#
# HOW TO USE:
#   ./scripts/run_valgrind_simulator.sh
#   ...then browse the CA list / switch rules until the crash WOULD happen.
#   Watch logs/valgrind.<pid>.log for the FIRST:
#       "Invalid write of size 8"   <-- this stack names the corruptor
#   (also look for "Invalid read", "Use of uninitialised", or a write whose
#    address is "inside a block of size N free'd" => use-after-free.)
#
# Expect ~20-50x slowdown. That's fine: memcheck flags the bad write on its
# first occurrence, usually well before the app would have crashed.
#
# Noise from the NVIDIA driver / OpenBLAS / pyarrow is filtered via
# scripts/nvidia.supp. If new noise floods the log, add stacks with
# --gen-suppressions=all (already on) copied into that file.

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

# Route ALL Python allocations through glibc malloc so valgrind can see them
# (pymalloc's arena pooling hides overflows/use-after-free from memcheck).
export PYTHONMALLOC=malloc
# Collapse worker thread-pools: fewer threads = far less valgrind noise and
# nondeterminism, and our bug is single-threaded (main thread) anyway.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1
export PYTHONFAULTHANDLER=1

LOG="logs/valgrind.%p.log"
echo "[valgrind] logging to logs/valgrind.<pid>.log" >&2
echo "[valgrind] watch for: 'Invalid write of size 8'  (the corruptor)" >&2
echo "[valgrind] browse the CA list until the crash would normally hit." >&2

exec valgrind \
    --tool=memcheck \
    --error-limit=no \
    --num-callers=40 \
    --redzone-size=128 \
    --freelist-vol=200000000 \
    --leak-check=no \
    --errors-for-leak-kinds=none \
    --gen-suppressions=all \
    --suppressions=scripts/nvidia.supp \
    --log-file="$LOG" \
    .venv/bin/python -u simulator.py "$@"
