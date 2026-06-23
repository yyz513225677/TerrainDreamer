#!/usr/bin/env bash
# auto_loop.sh — autonomous TerrainDreamer training loop.
#
# Endlessly:
#   1. Waits for current trainer to hit EPISODES_PER_ITER episodes.
#   2. Stops trainer + viewer (kept clean by clean_and_train.sh).
#   3. Snapshots run to experiments/crater/iter_<N>/
#   4. Runs auto_improve_iteration.py — decides next mutation, writes
#      experiments/crater/next_env.sh.
#   5. Sources next_env.sh and re-launches via clean_and_train.sh
#      (cold sim restart + fresh checkpoints).
#
# Run in the background:
#   nohup ./scripts/auto_loop.sh > /tmp/auto_loop.log 2>&1 &
#
# Stop the loop by ``pkill -f auto_loop.sh`` or ``Ctrl-C``.

set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPISODES_PER_ITER="${EPISODES_PER_ITER:-100}"
LOG_FILE="${LOG_FILE:-/tmp/dreamer_train.log}"
POLL_INTERVAL_S=60

log() {
  printf '\033[1;33m[auto_loop %s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"
}

trap 'log "stopping loop (signal)"; exit 0' INT TERM

iter=0
log "starting auto-loop (target $EPISODES_PER_ITER episodes per iter)"

# First launch: clean start if no trainer already running
if ! pgrep -f "python3 scripts/train_crater_ros" >/dev/null 2>&1; then
  log "no trainer running — launching iter 0"
  ( cd "$PROJECT_ROOT" && setsid nohup ./scripts/clean_and_train.sh > /tmp/train_lunar_launcher.log 2>&1 < /dev/null & disown )
  sleep 90   # wait for cold sim boot
fi

while true; do
  count=$(grep -cE "=== Episode" "$LOG_FILE" 2>/dev/null || echo 0)

  # Safety: detect dead trainer
  if [ "$count" -gt 0 ] && ! pgrep -f "python3 scripts/train_crater_ros" >/dev/null 2>&1; then
    log "trainer died at episode $count — restarting"
    sleep 5
    ( cd "$PROJECT_ROOT" && setsid nohup ./scripts/clean_and_train.sh > /tmp/train_lunar_launcher.log 2>&1 < /dev/null & disown )
    sleep 90
    continue
  fi

  if [ "$count" -ge "$EPISODES_PER_ITER" ]; then
    log "=== reached $count episodes; running iter $iter analysis ==="
    python3 "$PROJECT_ROOT/scripts/auto_improve_iteration.py" 2>&1 | tee -a /tmp/auto_loop.log

    log "applying mutation and relaunching ..."
    # Source mutation env so next iter sees it
    if [ -f "$PROJECT_ROOT/experiments/crater/next_env.sh" ]; then
      # shellcheck disable=SC1091
      source "$PROJECT_ROOT/experiments/crater/next_env.sh"
    fi

    # clean_and_train.sh does the kill + restart-sim
    ( cd "$PROJECT_ROOT" && setsid nohup ./scripts/clean_and_train.sh > /tmp/train_lunar_launcher.log 2>&1 < /dev/null & disown )
    iter=$((iter + 1))
    log "iter $iter launched — waiting 120s for cold sim ..."
    sleep 120
    continue
  fi

  sleep "$POLL_INTERVAL_S"
done
