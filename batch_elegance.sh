#!/bin/bash
# Parallel search for elegant CA structures (oscillators, gliders, guns, symmetry)
# Uses 300 steps per trial for dynamics capture and the "elegance" compound metric
cd /home/user/development/experiments/cellular-automata
PYTHON=/home/user/development/experiments/.venv/bin/python3
HARNESS=test_harness.py
TRIALS=200
STEPS=300

RULES=(
  445_rule
  smoothlife_3d
  reaction_diffusion_3d
  gray_scott_worms
  wave_3d
  crystal_growth
  lenia_3d
  game_of_life_3d
)

# Clean up old per-rule files
rm -f elegance_*.tmp.json

echo "Starting parallel ELEGANCE search: ${#RULES[@]} rules × $TRIALS trials × $STEPS steps"
echo "Metrics: period, glider, growth, clusters, symmetry → elegance score"
echo ""

# Run all searches in parallel
PIDS=()
for rule in "${RULES[@]}"; do
  outfile="elegance_${rule}.tmp.json"
  echo "  Launching: $rule → $outfile"
  $PYTHON $HARNESS --seed $RANDOM --steps $STEPS search "$rule" \
    --trials $TRIALS --metric elegance --top 20 --save "$outfile" &
  PIDS+=($!)
done

echo ""
echo "All ${#RULES[@]} searches running (PIDs: ${PIDS[*]})"
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

# Merge all per-rule files into discoveries.json
$PYTHON -c "
import json, glob, os

# Load existing
main_file = 'discoveries.json'
existing = []
if os.path.exists(main_file):
    with open(main_file) as f:
        existing = json.load(f)

# Load all tmp files
new_count = 0
for path in sorted(glob.glob('elegance_*.tmp.json')):
    with open(path) as f:
        entries = json.load(f)
    existing.extend(entries)
    new_count += len(entries)
    os.remove(path)

with open(main_file, 'w') as f:
    json.dump(existing, f, indent=2)

print(f'Merged {new_count} new elegance discoveries into {main_file}')
print(f'Total: {len(existing)}')
print()
from collections import Counter
c = Counter(x['rule'] for x in existing)
for rule, count in c.most_common():
    rule_d = [x for x in existing if x['rule'] == rule]
    best_s = max(x.get('score', 0) for x in rule_d)
    best_p = max(x.get('period_score', 0) for x in rule_d)
    best_g = max(x.get('translation_score', 0) for x in rule_d)
    best_c = max(x.get('cluster_score', 0) for x in rule_d)
    best_y = max(x.get('symmetry_score', 0) for x in rule_d)
    print(f'  {rule:25s}: {count:3d}  S={best_s:.2f} Per={best_p:.2f} Glide={best_g:.2f} Clust={best_c:.2f} Sym={best_y:.2f}')
"

echo ""
echo "Done!"
