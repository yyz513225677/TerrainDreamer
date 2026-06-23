#!/usr/bin/env bash
# paper_post_rugged_watcher.sh — independent watcher that fires once,
# after the Rugged 5-seed step of run_paper_experiments_v2.sh completes.
#
# Trigger condition: v2 queue log contains "STEP 3/6" (i.e. v2 queue
# has finished step 2 and started step 3 B1 baseline) OR
# 5 rugged iters appear in experiments/crater since iter_126.
#
# Actions on trigger:
#   1. python3 scripts/update_paper_metrics.py  -> patch Table 1
#   2. python3 scripts/generate_paper_figures.py -> regen all PNG figures
#   3. write a marker file paper/.rugged_done so this watcher
#      exits gracefully and never re-fires.
#
# Usage:
#   nohup bash scripts/paper_post_rugged_watcher.sh \
#     > /tmp/paper_post_rugged.log 2>&1 &

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_LOG="/tmp/paper_queue_v2.log"
MARKER="$PROJECT_ROOT/paper/.rugged_done"

log() { printf '\033[1;32m[post-rugged %s]\033[0m %s\n' "$(date +'%F %T')" "$*"; }

if [ -f "$MARKER" ]; then
    log "marker exists; nothing to do (already updated)."
    exit 0
fi

log "started; watching for Rugged completion."

while true; do
    sleep 60
    # Two independent triggers — either suffices.
    triggered=0
    if grep -q "STEP 3/6" "$QUEUE_LOG" 2>/dev/null; then
        log "v2 queue advanced to STEP 3 — Rugged done."
        triggered=1
    fi
    # Backup trigger: count rugged iter metrics directly.
    n_rugged=$(ls -d "$PROJECT_ROOT/experiments/crater/iter_"* 2>/dev/null | \
        while read d; do
            it=$(basename "$d" | sed 's/iter_//')
            [ "$it" -lt 126 ] 2>/dev/null && continue
            meta="$d/run_metadata.json"
            [ -f "$meta" ] || continue
            tr=$(python3 -c "import json,sys; print(json.load(open('$meta')).get('env',{}).get('TERRAIN',''))" 2>/dev/null)
            [ "$tr" = "rugged" ] && [ -f "$d/metrics.json" ] && echo "$it"
        done | wc -l)
    if [ "$n_rugged" -ge 5 ]; then
        log "5+ rugged metrics found ($n_rugged) — Rugged done."
        triggered=1
    fi
    [ "$triggered" -eq 1 ] && break
done

log "running update_paper_metrics.py ..."
python3 "$PROJECT_ROOT/scripts/update_paper_metrics.py" 2>&1 \
    | tee "$PROJECT_ROOT/paper/.last_update.log"

log "running generate_paper_figures.py ..."
python3 "$PROJECT_ROOT/scripts/generate_paper_figures.py" 2>&1 \
    | tee "$PROJECT_ROOT/paper/.last_figures.log"

touch "$MARKER"
log "paper Table 1 + figures refreshed. marker written. exiting."
