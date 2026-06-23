#!/usr/bin/env bash
# mem_watchdog.sh — background sentinel that kills the current trainer
# if system memory exceeds a threshold. Designed to run alongside
# run_seeds.sh so a leaking trainer doesn't OOM-kill the whole desktop
# (including VSCode / shells) in long overnight sessions.
#
# Behavior:
#   - Polls /proc/meminfo every 30 s.
#   - When (MemTotal - MemAvailable) / MemTotal > THRESHOLD, kills the
#     active train_crater_ros / gz sim processes.
#   - run_seeds.sh's poll loop detects the trainer death and advances
#     to its post-seed cleanup, then starts the next seed fresh.
#
# Usage (parallel to run_seeds.sh):
#   nohup bash scripts/mem_watchdog.sh > /tmp/mem_watchdog.log 2>&1 &

set -uo pipefail
THRESHOLD="${MEM_THRESHOLD:-90}"   # percent used to trigger kill
POLL_S="${POLL_S:-30}"
log() { printf '\033[1;31m[mem-watchdog %s]\033[0m %s\n' "$(date +'%F %T')" "$*"; }

log "started; threshold=${THRESHOLD}% poll=${POLL_S}s"
while true; do
    info=$(free -m)
    total=$(echo "$info" | awk '/^Mem:/{print $2}')
    avail=$(echo "$info" | awk '/^Mem:/{print $7}')
    used_pct=$(( 100 - 100 * avail / total ))
    if [ "$used_pct" -ge "$THRESHOLD" ]; then
        log "MEM USED=${used_pct}% (avail=${avail}MB / ${total}MB) — killing trainer + sim"
        pkill -9 -f "train_crater_ros" 2>/dev/null || true
        pkill -9 -f "gz sim"           2>/dev/null || true
        pkill -9 -f "ruby.*gz"         2>/dev/null || true
        log "kill issued. run_seeds.sh poll loop will detect death and reset."
        sleep 30
    fi
    sleep "$POLL_S"
done
