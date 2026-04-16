#!/bin/bash
cd /home/user/development/experiments/cellular-automata
PYTHON=/home/user/development/experiments/.venv/bin/python3
HARNESS=test_harness.py
TRIALS=500

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
rm -f discoveries_*.tmp.json

echo "Starting parallel search: ${#RULES[@]} rules × $TRIALS trials each"
echo ""

# Run all searches in parallel, each saving to its own file
PIDS=()
for rule in "${RULES[@]}"; do
  outfile="discoveries_${rule}.tmp.json"
  echo "  Launching: $rule → $outfile"
  $PYTHON $HARNESS --seed $RANDOM search "$rule" --trials $TRIALS --metric combined --top 20 \
    --save "$outfile" &
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
for path in sorted(glob.glob('discoveries_*.tmp.json')):
    with open(path) as f:
        entries = json.load(f)
    existing.extend(entries)
    new_count += len(entries)
    os.remove(path)

with open(main_file, 'w') as f:
    json.dump(existing, f, indent=2)

print(f'Merged {new_count} new discoveries into {main_file}')
print(f'Total: {len(existing)}')
print()
from collections import Counter
c = Counter(x['rule'] for x in existing)
for rule, count in c.most_common():
    rule_d = [x for x in existing if x['rule'] == rule]
    best_s = max(x.get('score', 0) for x in rule_d)
    best_g = max(x.get('gol_coherence', 0) for x in rule_d)
    best_p = max(x.get('projection_complexity', 0) for x in rule_d)
    best_m = max(x.get('slice_mi', 0) for x in rule_d)
    print(f'  {rule:25s}: {count:3d}  S={best_s:.2f} G={best_g:.2f} P={best_p:.2f} M={best_m:.3f}')
"

echo ""
echo "Done!"
