#!/usr/bin/env bash
# run_paper_experiments.sh — sequential experiment queue for the T-IV
# paper. Runs the remaining sweeps in priority order. Each sweep is
# ~25 hours of sim time. The script blocks until each sub-task completes
# before starting the next, so you can leave it overnight.
#
# Queue (highest priority first):
#   1. Landscape 5-seed (TerrainDreamer full)        ~25 h
#   2. B1 vanilla baseline on extreme         ~25 h  ← critical for paper
#   3. A1 -interrupts ablation on extreme     ~25 h
#   4. A2 -distillation ablation on extreme   ~25 h
#   5. A3 -demo-anchor ablation on extreme    ~25 h
#
# Total wall-time: ~125 h.
#
# Logs:
#   /tmp/paper_queue.log              high-level scheduler log
#   /tmp/seed_runner_<step>.log       per-step sub-process log
#
# Usage:
#   nohup bash scripts/run_paper_experiments.sh > /tmp/paper_queue.log 2>&1 &
#   tail -f /tmp/paper_queue.log

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

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

# Kill anything currently running so the queue starts from a clean state.
log "queue starting — cleaning prior sim processes."
pkill -9 -f "scripts/run_seeds.sh|scripts/run_ablations.sh|clean_and_train.sh|train_lunar.sh|gz sim|train_crater_ros" 2>/dev/null
sleep 10

# Ensure SEEDS list is the standard 5.
sed -i 's/^SEEDS=.*/SEEDS=(42 1337 2024 7 31415)/' "$PROJECT_ROOT/scripts/run_seeds.sh"

#-------------------------------------------------------------------
# Step 1: Landscape 5-seed (TerrainDreamer full)
#-------------------------------------------------------------------
log "STEP 1/5: Landscape 5-seed (TerrainDreamer full)"
TERRAIN=default HEADLESS=1 VIEWER=0 N_SEEDS=5 EPISODES_PER_ITER=50 \
    nohup bash "$PROJECT_ROOT/scripts/run_seeds.sh" \
      > /tmp/seed_runner_landscape.log 2>&1 &
sleep 5
wait_for_runner_done "Landscape 5-seed"

#-------------------------------------------------------------------
# Step 2: B1 vanilla baseline on extreme  (CRITICAL for paper)
#-------------------------------------------------------------------
log "STEP 2/5: B1 vanilla baseline on extreme"
ABLATIONS="B1" TERRAIN_LIST="extreme" HEADLESS=1 \
    nohup bash "$PROJECT_ROOT/scripts/run_ablations.sh" \
      > /tmp/seed_runner_B1.log 2>&1 &
sleep 5
wait_for_runner_done "B1 vanilla"

#-------------------------------------------------------------------
# Step 3: A1 -interrupts on extreme
#-------------------------------------------------------------------
log "STEP 3/5: A1 -interrupts on extreme"
ABLATIONS="A1" TERRAIN_LIST="extreme" HEADLESS=1 \
    nohup bash "$PROJECT_ROOT/scripts/run_ablations.sh" \
      > /tmp/seed_runner_A1.log 2>&1 &
sleep 5
wait_for_runner_done "A1 -interrupts"

#-------------------------------------------------------------------
# Step 4: A2 -distillation on extreme
#-------------------------------------------------------------------
log "STEP 4/5: A2 -distillation on extreme"
ABLATIONS="A2" TERRAIN_LIST="extreme" HEADLESS=1 \
    nohup bash "$PROJECT_ROOT/scripts/run_ablations.sh" \
      > /tmp/seed_runner_A2.log 2>&1 &
sleep 5
wait_for_runner_done "A2 -distillation"

#-------------------------------------------------------------------
# Step 5: A3 -demo anchor on extreme
#-------------------------------------------------------------------
log "STEP 5/5: A3 -demo-anchor on extreme"
ABLATIONS="A3" TERRAIN_LIST="extreme" HEADLESS=1 \
    nohup bash "$PROJECT_ROOT/scripts/run_ablations.sh" \
      > /tmp/seed_runner_A3.log 2>&1 &
sleep 5
wait_for_runner_done "A3 -demo-anchor"

#-------------------------------------------------------------------
log "ALL DONE. Running build_results_table.py for final consolidated table."
python3 "$PROJECT_ROOT/scripts/build_results_table.py"
log "queue exit."
