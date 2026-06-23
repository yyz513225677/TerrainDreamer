#!/usr/bin/env bash
# paper_post_ablation_watcher.sh — fires update_paper_metrics.py + figures
# every time the v2 queue advances past an ablation step (B1, A1, A2, A3).
#
# Trigger conditions (any of):
#   * v2 queue log shows next STEP started (STEP 4/6, STEP 5/6, STEP 6/6,
#     or "ALL DONE").
#   * Backup: at least 5 iters exist in the expected range for the ablation
#     we are currently waiting on (B1=137-147, A1=148-157, A2=158-167,
#     A3=168-177).
#
# Each trigger re-runs update_paper_metrics.py (patches Tables 1 & 2) and
# regenerates the paper figures. State markers are written under paper/
# to prevent re-firing for an ablation that has already been processed.

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_LOG="/tmp/paper_queue_v2.log"
PAPER_DIR="$PROJECT_ROOT/paper"
EXP_DIR="$PROJECT_ROOT/experiments/crater"

log() { printf '\033[1;33m[post-ablation %s]\033[0m %s\n' "$(date +'%F %T')" "$*"; }

# Each entry: "KEY:NEXT_STEP_PATTERN:LO:HI"
# We fire when either the queue advances OR enough iters exist in [LO,HI].
declare -a STEPS=(
    "B1:STEP 4/6:137:147"
    "A1:STEP 5/6:148:157"
    "A2:STEP 6/6:158:167"
    "A3:ALL DONE:168:177"
)

iters_in_range() {
    local lo=$1 hi=$2 cnt=0
    for it in $(seq "$lo" "$hi"); do
        [ -f "$EXP_DIR/iter_$it/metrics.json" ] || continue
        n=$(python3 -c "import json,sys; print(json.load(open('$EXP_DIR/iter_$it/metrics.json')).get('n_episodes',0))" 2>/dev/null)
        [ "${n:-0}" -ge 40 ] 2>/dev/null && cnt=$((cnt+1))
    done
    echo "$cnt"
}

run_update() {
    log "running update_paper_metrics.py ..."
    python3 "$PROJECT_ROOT/scripts/update_paper_metrics.py" 2>&1 \
        | tee "$PAPER_DIR/.last_update.log"
    log "running generate_paper_figures.py ..."
    python3 "$PROJECT_ROOT/scripts/generate_paper_figures.py" 2>&1 \
        | tee "$PAPER_DIR/.last_figures.log"
}

log "started; watching B1/A1/A2/A3 completion."

while true; do
    sleep 60
    all_done=1
    for entry in "${STEPS[@]}"; do
        IFS=":" read -r key pattern lo hi <<< "$entry"
        marker="$PAPER_DIR/.${key}_done"
        if [ -f "$marker" ]; then continue; fi
        all_done=0
        fired=0
        if grep -qF "$pattern" "$QUEUE_LOG" 2>/dev/null; then
            log "queue advanced past $key (matched '$pattern')."
            fired=1
        fi
        n_ready=$(iters_in_range "$lo" "$hi")
        if [ "$n_ready" -ge 5 ]; then
            log "$key has 5 completed iters in [$lo,$hi]."
            fired=1
        fi
        if [ "$fired" -eq 1 ]; then
            run_update
            touch "$marker"
            log "patched Table 1 & 2 after $key; marker written."
        fi
    done
    if [ "$all_done" -eq 1 ]; then
        log "all ablations processed. exiting."
        break
    fi
done
