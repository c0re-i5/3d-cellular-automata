#!/bin/bash
# Ordered discovery generator: all jobs run in parallel, merged in rule order
# Results are grouped by rule in discoveries.json so < > buttons step through each CA in order
cd /home/user/development/experiments/cellular-automata
PYTHON=/home/user/development/experiments/.venv/bin/python3
HARNESS=test_harness.py
TRIALS=750
STEPS=300
TOP=100
MAX_JOBS=8  # concurrent GPU processes (tune for your GPU memory)

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
OUTFILE="discoveries.json"

NJOBS=$(( ${#RULES[@]} * ${#METRICS[@]} ))
echo "=== ORDERED DISCOVERY GENERATOR ==="
echo "${#RULES[@]} rules × ${#METRICS[@]} metrics × $TRIALS trials × $STEPS steps"
echo "$NJOBS total jobs, $MAX_JOBS concurrent"
echo "Output: $OUTFILE"
echo ""

# Append to existing discoveries (accumulates across runs)
if [ ! -f "$OUTFILE" ]; then
  echo "[]" > "$OUTFILE"
fi

# Clean old tmp files
rm -f ordered_*_*.tmp.json

# ── Launch all jobs with a concurrency limiter ─────────────────────────
RUNNING=0
PIDS=()
JOBS_DESC=()

for rule in "${RULES[@]}"; do
  for metric in "${METRICS[@]}"; do
    # Wait if we're at the job limit
    while (( RUNNING >= MAX_JOBS )); do
      # Wait for any child to finish
      wait -n 2>/dev/null
      RUNNING=$(jobs -rp | wc -l)
    done

    tmpfile="ordered_${rule}_${metric}.tmp.json"
    echo "  Launching: $rule ($metric) → $tmpfile"
    $PYTHON $HARNESS --seed $RANDOM --steps $STEPS search "$rule" \
      --trials $TRIALS --metric "$metric" --top $TOP --min_quality 0.1 --save "$tmpfile" &
    PIDS+=($!)
    JOBS_DESC+=("$rule/$metric")
    RUNNING=$(jobs -rp | wc -l)
  done
done

echo ""
echo "All $NJOBS jobs launched (max $MAX_JOBS concurrent)"
echo "Waiting for completion..."

# Wait for everything
FAILED=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "  FAILED: ${JOBS_DESC[$i]}"
    FAILED=$((FAILED + 1))
  fi
done

if [ $FAILED -gt 0 ]; then
  echo "WARNING: $FAILED/$NJOBS jobs failed"
fi

# ── Merge in rule order ────────────────────────────────────────────────
echo ""
echo "Merging results in rule order..."

$PYTHON -c "
import json, glob, os

rules = '''${RULES[*]}'''.split()
outfile = '$OUTFILE'

# Load existing discoveries to append to
existing = []
if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
    with open(outfile) as f:
        existing = json.load(f)

# Build set of existing keys for dedup
existing_keys = set()
for d in existing:
    key = (d.get('rule', ''), d.get('seed', 0), str(sorted(d.get('params', {}).items())))
    existing_keys.add(key)

new_total = 0
for rule in rules:
    # Collect all tmp files for this rule
    new_entries = []
    for path in sorted(glob.glob(f'ordered_{rule}_*.tmp.json')):
        with open(path) as f:
            entries = json.load(f)
        new_entries.extend(entries)
        os.remove(path)

    if not new_entries:
        print(f'  {rule:25s}: 0 new (no results)')
        continue

    # Deduplicate against existing and within batch
    unique = []
    for d in new_entries:
        key = (d.get('rule', ''), d.get('seed', 0), str(sorted(d.get('params', {}).items())))
        if key not in existing_keys:
            existing_keys.add(key)
            unique.append(d)

    # Sort this rule's new discoveries by score descending
    unique.sort(key=lambda d: d.get('score', 0), reverse=True)
    existing.extend(unique)
    new_total += len(unique)

    scores = [d.get('score', 0) for d in unique] if unique else [0]
    print(f'  {rule:25s}: {len(unique):3d} new  best={max(scores):.2f}  avg={sum(scores)/len(scores):.2f}')

# Re-sort entire file: group by rule (in RULES order), then by score within each rule
rule_order = {r: i for i, r in enumerate(rules)}
existing.sort(key=lambda d: (rule_order.get(d.get('rule',''), 999), -d.get('score', 0)))

with open(outfile, 'w') as f:
    json.dump(existing, f, indent=1)

print(f'\nAdded {new_total} new discoveries (total now: {len(existing)} in {outfile})')
"

echo ""
echo "Done!"
echo "Launch simulator to browse: ../.venv/bin/python simulator.py"
echo "  Use < > buttons with 'All rules' to step through each CA in order"
