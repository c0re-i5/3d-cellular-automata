#!/usr/bin/env bash
# Fresh catalogue rebuild: brief search across EVERY searchable preset.
#
# Context (2026-07-16): discoveries.json was purged (backup:
# discoveries_backup_pre_purge_*.json) after the catalogue audit found only
# 32% of 12,320 entries hash-valid, all crystal_* physics-invalid (harness
# pair-2 binding bug, since fixed), and lenia_multi 100% stale. This script
# rebuilds a small-but-trustworthy baseline: every entry it produces carries
# the current rule_code_hash and ran with correct pair-2 physics.
#
# Brief by design: TRIALS=40 per rule (vs 400+ for a deep search). Rules that
# deserve deep searches can get them individually afterwards.
#
# Lenia-family rules are searched with --metric lenia_organism (structure/
# motility-aware); everything else uses the default interestingness score.
#
# Usage:
#   bash batch_fresh_sweep.sh
#   TRIALS=80 STEPS=300 bash batch_fresh_sweep.sh
#   TRIALS=400 STEPS=400 SIZE=64 bash batch_fresh_sweep.sh   # FULL sweep (~overnight)
#
# SIZE >= 64 recommended for full sweeps: size-32 searches reward
# volume-filling drift waves for kernel rules (lenia_catalogue_notes.md).
set -uo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-.venv/bin/python}"
TRIALS="${TRIALS:-40}"
STEPS="${STEPS:-250}"
SIZE="${SIZE:-32}"
MIN_QUALITY="${MIN_QUALITY:-0.25}"

# Lenia family: organism metric (needs dynamics capture; a bit slower).
LENIA_RULES="lenia_3d lenia_geminium lenia_multi lenia_rings flow_lenia flagship_dual_lenia"

# All other searchable rules, emitted by preset introspection so the list
# never goes stale as presets are added/removed.
OTHER_RULES=$("$PYTHON" -c "
import simulator as s
lenia = set('$LENIA_RULES'.split())
for name, p in s.RULE_PRESETS.items():
    pr = p.get('param_ranges') or {}
    if name not in lenia and any(v[0] != v[1] for v in pr.values()):
        print(name)
")

total=0; failed=0
run_search () {
    local rule="$1" metric="$2"
    total=$((total+1))
    echo "=== [$(date +%H:%M:%S)] $rule (metric=$metric trials=$TRIALS steps=$STEPS size=$SIZE) ==="
    if ! "$PYTHON" test_harness.py --steps "$STEPS" --size "$SIZE" search "$rule" \
            --metric "$metric" --trials "$TRIALS" --min_quality "$MIN_QUALITY"; then
        echo "!!! $rule FAILED (continuing)"
        failed=$((failed+1))
    fi
}

for rule in $LENIA_RULES; do
    run_search "$rule" lenia_organism
done
for rule in $OTHER_RULES; do
    run_search "$rule" score
done

echo "=== sweep complete: $total rules, $failed failed ==="
