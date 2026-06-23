#!/usr/bin/env bash
# run_ablations.sh — sequentially run TerrainDreamer ablation conditions for the
# T-IV paper. Each condition runs the same 5-seed sweep as run_seeds.sh,
# but with one TerrainDreamer contribution disabled.
#
# Conditions (in order):
#   B1: DreamerV3-vanilla (no interrupts, no distillation, no demo anchor)
#   A1: -interrupts only        (full TerrainDreamer minus safety filter)
#   A2: -distillation only      (full TerrainDreamer minus reverse-mapping)
#   A3: -demo anchor only       (full TerrainDreamer minus BC anchor)
#
# Each condition: TERRAIN × 5 seeds × 50 episodes ≈ 25 h sim wall-time.
# Default: only EXTREME terrain. Set TERRAIN_LIST="rugged extreme" to
# run multiple terrains.
#
# Usage:
#   ./scripts/run_ablations.sh                       # run all 4 conditions
#   ABLATIONS="B1" ./scripts/run_ablations.sh         # one condition only
#   TERRAIN_LIST="extreme" ./scripts/run_ablations.sh
#
# Each condition writes its 5 metrics.json files to
# experiments/crater/iter_<N>/. The label file
# experiments/crater/iter_<N>/baseline_label.txt
# records the condition name so /scripts/build_results_table.py can
# group them later.

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TERRAIN_LIST="${TERRAIN_LIST:-extreme}"
ABLATIONS="${ABLATIONS:-B1 A1 A2 A3}"
HEADLESS="${HEADLESS:-1}"
N_SEEDS="${N_SEEDS:-5}"
EPISODES_PER_ITER="${EPISODES_PER_ITER:-50}"

log() { printf '\033[1;33m[ablation %s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }

# Map condition name → env-var overrides
ablation_env() {
    case "$1" in
        B1)  echo "ABLATE_INTERRUPTS=1 ABLATE_DISTILLATION=1 ABLATE_DEMO_ANCHOR=1" ;;
        A1)  echo "ABLATE_INTERRUPTS=1" ;;
        A2)  echo "ABLATE_DISTILLATION=1" ;;
        A3)  echo "ABLATE_DEMO_ANCHOR=1" ;;
        *)   echo "" ;;
    esac
}

for terrain in $TERRAIN_LIST; do
  for cond in $ABLATIONS; do
    envstr=$(ablation_env "$cond")
    if [ -z "$envstr" ]; then
      log "unknown condition '$cond' — skipping"
      continue
    fi
    log "=== TERRAIN=$terrain CONDITION=$cond ($envstr) ==="
    # Stash a label that build_results_table.py can pick up.
    label_pre="$cond"

    # Run the standard run_seeds with the ablation env injected.
    # Each iter dir gets a baseline_label.txt automatically via
    # the .next_label file (see run_seeds.sh).
    echo "${cond}_${terrain}" > "$PROJECT_ROOT/experiments/crater/.next_label"

    env $envstr \
      TERRAIN="$terrain" \
      HEADLESS="$HEADLESS" \
      N_SEEDS="$N_SEEDS" \
      EPISODES_PER_ITER="$EPISODES_PER_ITER" \
      bash "$PROJECT_ROOT/scripts/run_seeds.sh" 2>&1 \
      | tee "$PROJECT_ROOT/experiments/crater/.last_ablation.log"

    log "$cond on $terrain done."
  done
done

log "all ablations done — running build_results_table.py"
python3 "$PROJECT_ROOT/scripts/build_results_table.py"
