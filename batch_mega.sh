#!/bin/bash
# Mega search: runs elegance AND combined metrics in parallel with diversity-aware search
# Varies init density, uses novelty bonus to maximize variety of discoveries
# Uses the improved quality filter that rejects global blinkers
cd /home/user/development/experiments/cellular-automata
PYTHON=/home/user/development/experiments/.venv/bin/python3
HARNESS=test_harness.py
TRIALS=500
STEPS=300
TOP=50

RULES=(
  game_of_life_3d
  445_rule
  smoothlife_3d
  reaction_diffusion_3d
  gray_scott_worms
  wave_3d
  crystal_growth
  crystal_dendritic
  crystal_faceted
  lenia_3d
  lenia_geminium
  lenia_multi
  predator_prey_3d
  kuramoto_3d
  bz_spiral_waves
  bz_turbulence
  bz_excitable
  morphogen_spots
  flocking_3d
  phase_separation
  nucleation
  erosion
  mycelium
  em_wave
  viscous_fingers
  fire
  physarum
  fracture
  galaxy
  lichen
)

METRICS=(elegance combined)

# Clean up old tmp files
rm -f mega_*.tmp.json

TOTAL=$((${#RULES[@]} * ${#METRICS[@]}))
echo "=== MEGA SEARCH ==="
echo "${#RULES[@]} rules × ${#METRICS[@]} metrics × $TRIALS trials × $STEPS steps"
echo "Total: $TOTAL parallel jobs, $(( ${#RULES[@]} * ${#METRICS[@]} * TRIALS )) trials"
echo ""

# Launch all jobs
PIDS=()
for metric in "${METRICS[@]}"; do
  for rule in "${RULES[@]}"; do
    outfile="mega_${metric}_${rule}.tmp.json"
    echo "  Launching: $rule ($metric) → $outfile"
    $PYTHON $HARNESS --seed $RANDOM --steps $STEPS search "$rule" \
      --trials $TRIALS --metric "$metric" --top $TOP --save "$outfile" &
    PIDS+=($!)
  done
done

echo ""
echo "All $TOTAL searches running (PIDs: ${PIDS[*]})"
echo "Waiting for completion..."

# Wait for all
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait $pid; then
    FAILED=$((FAILED + 1))
  fi
done

if [ $FAILED -gt 0 ]; then
  echo "WARNING: $FAILED searches failed"
fi

echo ""
echo "Merging results..."

# Merge all tmp files into discoveries.json, applying quality filter
$PYTHON -c "
import json, glob, os

main_file = 'discoveries.json'
existing = []
if os.path.exists(main_file):
    with open(main_file) as f:
        existing = json.load(f)

old_count = len(existing)

# Load all tmp files
new_entries = []
for path in sorted(glob.glob('mega_*.tmp.json')):
    with open(path) as f:
        entries = json.load(f)
    new_entries.extend(entries)
    os.remove(path)

# Filter out global blinkers from new entries
kept = []
rejected = 0
for d in new_entries:
    alive = d.get('final_alive', 0)
    activity = d.get('final_activity', 0)
    if alive > 0.2 and activity > 0.5:
        act_ratio = activity / max(alive, 0.01)
        if act_ratio > 1.5:
            rejected += 1
            continue
    kept.append(d)

existing.extend(kept)

with open(main_file, 'w') as f:
    json.dump(existing, f, indent=1)

print(f'New entries: {len(new_entries)} (kept {len(kept)}, rejected {rejected} blinkers)')
print(f'Total discoveries: {old_count} → {len(existing)}')
print()

# Summary by rule
from collections import Counter
c = Counter(x['rule'] for x in existing)
for rule, count in c.most_common():
    rule_d = [x for x in existing if x['rule'] == rule]
    best_s = max(x.get('score', 0) for x in rule_d)
    has_dyn = [x for x in rule_d if 'period_score' in x]
    if has_dyn:
        best_p = max(x.get('period_score', 0) for x in has_dyn)
        best_g = max(x.get('translation_score', 0) for x in has_dyn)
        best_c = max(x.get('cluster_score', 0) for x in has_dyn)
        best_y = max(x.get('symmetry_score', 0) for x in has_dyn)
        print(f'  {rule:25s}: {count:4d}  S={best_s:.2f} Per={best_p:.2f} Glide={best_g:.2f} Clust={best_c:.2f} Sym={best_y:.2f}')
    else:
        print(f'  {rule:25s}: {count:4d}  S={best_s:.2f}')
"

echo ""
echo "Done!"
