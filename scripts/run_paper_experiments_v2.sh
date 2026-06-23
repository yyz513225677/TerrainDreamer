#!/usr/bin/env bash
# run_paper_experiments_v2.sh — updated paper experiment queue.
# Adds a Rugged 5-seed re-run after Landscape, because the algorithm
# has changed since the prior Rugged result and we want all three
# terrains evaluated under the same final code.
#
# Queue order (each ~25 h):
#   1. Landscape 5-seed (TerrainDreamer full)
#   2. Rugged    5-seed (TerrainDreamer full)   <-- NEW: re-run under updated algo
#   3. B1 vanilla baseline (extreme)
#   4. A1 -interrupts (extreme)
#   5. A2 -distillation (extreme)
#   6. A3 -demo anchor (extreme)
#
# Total wall-time: ~150 h.

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Where to START in the queue. Useful for resuming after a partial run.
#   START_STEP=1 (default) runs all 6.
#   START_STEP=2 skips landscape, starts from rugged.
#   START_STEP=3 skips both, starts from B1 vanilla.
START_STEP="${START_STEP:-1}"

log() { printf '\033[1;36m[paper-queue %s]\033[0m %s\n' "$(date +'%F %T')" "$*"; }

wait_for_runner_done() {
    local label="$1"
    log "waiting for $label to finish..."
    while pgrep -f "scripts/run_seeds.sh|scripts/run_ablations.sh" > /dev/null 2>&1; do
        sleep 60
    done
    sleep 5
    log "$label complete."
}

clean_state() {
    pkill -9 -f "scripts/run_seeds.sh|scripts/run_ablations.sh|clean_and_train.sh|train_lunar.sh|gz sim|train_crater_ros" 2>/dev/null
    sleep 10
}

log "queue v2 starting — START_STEP=$START_STEP"
clean_state

sed -i 's/^SEEDS=.*/SEEDS=(42 1337 2024 7 31415)/' "$PROJECT_ROOT/scripts/run_seeds.sh"

#-------------------------------------------------------------------
# Step 1: Landscape 5-seed (TerrainDreamer full)
#-------------------------------------------------------------------
if [ "$START_STEP" -le 1 ]; then
log "STEP 1/6: Landscape 5-seed (TerrainDreamer full)"
TERRAIN=default HEADLESS=1 VIEWER=0 N_SEEDS=5 EPISODES_PER_ITER=50 \
    nohup bash "$PROJECT_ROOT/scripts/run_seeds.sh" \
      > /tmp/seed_runner_landscape.log 2>&1 &
sleep 5
wait_for_runner_done "Landscape 5-seed"
clean_state
fi

#-------------------------------------------------------------------
# Step 2: Rugged 5-seed (TerrainDreamer full)  -- re-run under updated algo
#-------------------------------------------------------------------
if [ "$START_STEP" -le 2 ]; then
log "STEP 2/6: Rugged 5-seed (TerrainDreamer full, re-run under updated algorithm)"
TERRAIN=rugged HEADLESS=1 VIEWER=0 N_SEEDS=5 EPISODES_PER_ITER=50 \
    nohup bash "$PROJECT_ROOT/scripts/run_seeds.sh" \
      > /tmp/seed_runner_rugged.log 2>&1 &
sleep 5
wait_for_runner_done "Rugged 5-seed"
clean_state

# ── All TerrainDreamer-full terrains complete. Auto-update paper Table 1 +
#    regenerate all paper figures using the freshly-collected metrics.
log "all TerrainDreamer-full terrains done — updating Table 1 + regenerating figures"
python3 "$PROJECT_ROOT/scripts/update_paper_metrics.py" \
    2>&1 | tee "$PROJECT_ROOT/paper/.last_update.log"
python3 "$PROJECT_ROOT/scripts/generate_paper_figures.py" \
    2>&1 | tee "$PROJECT_ROOT/paper/.last_figures.log"
log "paper Table 1 + figures refreshed."
fi

#-------------------------------------------------------------------
# Step 3: B1 vanilla baseline on extreme
#-------------------------------------------------------------------
if [ "$START_STEP" -le 3 ]; then
log "STEP 3/6: B1 vanilla baseline on extreme"
ABLATIONS="B1" TERRAIN_LIST="extreme" HEADLESS=1 \
    nohup bash "$PROJECT_ROOT/scripts/run_ablations.sh" \
      > /tmp/seed_runner_B1.log 2>&1 &
sleep 5
wait_for_runner_done "B1 vanilla"
clean_state
fi

#-------------------------------------------------------------------
# Step 4: A1 -interrupts on extreme
#-------------------------------------------------------------------
if [ "$START_STEP" -le 4 ]; then
log "STEP 4/6: A1 -interrupts on extreme"
ABLATIONS="A1" TERRAIN_LIST="extreme" HEADLESS=1 \
    nohup bash "$PROJECT_ROOT/scripts/run_ablations.sh" \
      > /tmp/seed_runner_A1.log 2>&1 &
sleep 5
wait_for_runner_done "A1 -interrupts"
clean_state
fi

#-------------------------------------------------------------------
# Step 5: A2 -distillation on extreme
#-------------------------------------------------------------------
if [ "$START_STEP" -le 5 ]; then
log "STEP 5/6: A2 -distillation on extreme"
ABLATIONS="A2" TERRAIN_LIST="extreme" HEADLESS=1 \
    nohup bash "$PROJECT_ROOT/scripts/run_ablations.sh" \
      > /tmp/seed_runner_A2.log 2>&1 &
sleep 5
wait_for_runner_done "A2 -distillation"
clean_state
fi

#-------------------------------------------------------------------
# Step 6: A3 -demo anchor on extreme
#-------------------------------------------------------------------
if [ "$START_STEP" -le 6 ]; then
log "STEP 6/6: A3 -demo-anchor on extreme"
ABLATIONS="A3" TERRAIN_LIST="extreme" HEADLESS=1 \
    nohup bash "$PROJECT_ROOT/scripts/run_ablations.sh" \
      > /tmp/seed_runner_A3.log 2>&1 &
sleep 5
wait_for_runner_done "A3 -demo-anchor"
clean_state
fi

#-------------------------------------------------------------------
log "ALL DONE. Running build_results_table.py for final consolidated table."
python3 "$PROJECT_ROOT/scripts/build_results_table.py"
log "queue v2 exit."
