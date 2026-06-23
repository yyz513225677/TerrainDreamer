#!/usr/bin/env bash
# queue_interceptor.sh — wait for the currently running paper queue
# (run_paper_experiments.sh) to complete STEP 1 (Landscape), then kill
# its remaining steps and chain into run_paper_experiments_v2.sh
# from STEP 2 (Rugged), so the final result set includes a
# Rugged 5-seed under the updated algorithm.
#
# Detection rule: STEP 1 complete = "STEP 2/5: B1 vanilla baseline"
# appears in /tmp/paper_queue.log AND the run_seeds.sh process from
# landscape has exited (briefly).
#
# Usage:
#   nohup bash scripts/queue_interceptor.sh > /tmp/queue_interceptor.log 2>&1 &

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_LOG="/tmp/paper_queue.log"
INTERCEPTED=0

log() { printf '\033[1;33m[interceptor %s]\033[0m %s\n' "$(date +'%F %T')" "$*"; }

log "started; watching $QUEUE_LOG for landscape completion."

while true; do
    sleep 60
    # Landscape complete when queue moves to STEP 2 in the OLD script.
    if grep -q "STEP 2/5: B1 vanilla baseline" "$QUEUE_LOG" 2>/dev/null; then
        log "landscape STEP 1 done. Killing old queue (it would run B1 now)."
        # Kill the old queue + any baseline runner it just spawned.
        pkill -9 -f "scripts/run_paper_experiments.sh" 2>/dev/null
        pkill -9 -f "scripts/run_ablations.sh"        2>/dev/null
        pkill -9 -f "scripts/run_seeds.sh"            2>/dev/null
        pkill -9 -f "clean_and_train.sh"              2>/dev/null
        pkill -9 -f "train_lunar.sh"                  2>/dev/null
        pkill -9 -f "train_crater_ros"                2>/dev/null
        pkill -9 -f "gz sim"                          2>/dev/null
        sleep 20

        log "launching v2 queue from STEP 2 (Rugged 5-seed)."
        cd "$PROJECT_ROOT"
        START_STEP=2 nohup bash scripts/run_paper_experiments_v2.sh \
            > /tmp/paper_queue_v2.log 2>&1 &
        log "v2 queue PID=$!"
        INTERCEPTED=1
        break
    fi
done

log "interceptor handed off to v2 queue. exiting."
