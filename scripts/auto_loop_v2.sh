#!/usr/bin/env bash
# auto_loop_v2.sh — robust iteration loop with explicit memory cleanup.
#
# Cycle:
#   1. Cold-start trainer via run_baseline.sh / clean_and_train.sh
#   2. Monitor /tmp/dreamer_train.log for EPISODES_PER_ITER episodes
#   3. Kill EVERYTHING (trainer, sim, viewer, bridges, controllers)
#   4. Free memory (drop caches, wait for kernel to recover)
#   5. Snapshot + analyze the run, write next mutation
#   6. Loop
#
# Env knobs:
#   BASELINE=crater|vanilla|no_demo|no_hier|no_memory
#   EPISODES_PER_ITER=50          # default 50
#   MAX_ITERATIONS=0              # 0 = forever
#   TERRAIN=rugged|landscape|...
#   HEADLESS=1
#
# Unlike auto_loop.sh, this version DOES NOT background the trainer in a
# subshell that gets stuck waiting on child IO. Instead it explicitly
# spawns the launcher via ``nohup`` with stdin/stdout redirected, then
# polls for episode count with sleep-based timing.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${BASELINE:-crater}"
EPISODES_PER_ITER="${EPISODES_PER_ITER:-50}"
MAX_ITERATIONS="${MAX_ITERATIONS:-0}"
POLL_INTERVAL_S=45
COLD_BOOT_WAIT_S=180     # gz sim needs this long to come up fresh
LOG=/tmp/dreamer_train.log
META_LOG=/tmp/auto_loop_v2.log

log() {
  printf '\033[1;33m[loop %s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$META_LOG"
}

kill_everything() {
  log "killing all TerrainDreamer processes ..."
  for p in \
    "python3 scripts/train_crater_ros" \
    "python3 scripts/td_viewer.py" \
    "train_lunar.sh" \
    "clean_and_train.sh" \
    "run_baseline.sh" \
    "ruby.*gz sim" \
    "gz sim " \
    "ros2 launch lunar_south_pole_gazebo" \
    "ros2 launch ros_gz_sim" \
    "spawn_clearpath_jackal" \
    "robot_state_publisher.*j100_0001" \
    "controller_manager.*j100_0001" \
    "ekf_node" \
    "imu_filter_madgwick" \
    "twist_mux" \
    "joy_linux_node" \
    "teleop_twist_joy_node" \
    "parameter_bridge" \
    "dem_metadata_node" \
    "terrain_manager_node" \
    "hazard_status_node" \
    "static_transform_publisher"
  do
    pkill -9 -f "$p" 2>/dev/null || true
  done
  sleep 5
}

free_memory() {
  log "memory before: $(free -m | awk '/Mem:/ {print $3"MB used / "$2"MB total"}')"
  sync
  # Best-effort cache drop; doesn't require sudo on most systems.
  ( echo 3 > /proc/sys/vm/drop_caches ) 2>/dev/null || true
  sleep 2
  log "memory after:  $(free -m | awk '/Mem:/ {print $3"MB used / "$2"MB total"}')"
}

snapshot_and_analyze() {
  local iter_n
  iter_n=$(cat "$PROJECT_ROOT/experiments/crater/.current_iter" 2>/dev/null || echo "")
  if [ -n "$iter_n" ]; then
    local iter_dir="$PROJECT_ROOT/experiments/crater/iter_${iter_n}"
    [ -f "$LOG" ] && cp -L "$LOG" "$iter_dir/dreamer_train.log" 2>/dev/null
    log "running analysis for iter $iter_n"
    python3 "$PROJECT_ROOT/scripts/analyze_training.py" 2>&1 \
      | tee "$iter_dir/analysis.txt"
    python3 "$PROJECT_ROOT/scripts/auto_improve_iteration.py" 2>&1 \
      | tee -a "$META_LOG"
  fi
}

launch_iter() {
  log "launching $BASELINE in fresh sim ..."
  cd "$PROJECT_ROOT"
  # The launcher exec-chains into train_lunar.sh which keeps the trainer
  # in a tee pipeline. Use nohup with fully-redirected fds so the parent
  # shell (this script) doesn't get blocked on the pipe.
  nohup ./scripts/run_baseline.sh "$BASELINE" \
      > /tmp/train_lunar_launcher.log 2>&1 < /dev/null &
  disown
  log "waiting $COLD_BOOT_WAIT_S s for cold sim boot ..."
  sleep "$COLD_BOOT_WAIT_S"
}

trap 'log "stopping (signal)"; kill_everything; exit 0' INT TERM

iter=0
log "auto_loop_v2 start: baseline=$BASELINE  ep/iter=$EPISODES_PER_ITER  max_iter=${MAX_ITERATIONS:-∞}"
launch_iter

while true; do
  count=$(grep -cE "=== Episode" "$LOG" 2>/dev/null || echo 0)

  # Trainer died unexpectedly?
  if ! pgrep -f "python3 scripts/train_crater_ros" > /dev/null 2>&1; then
    log "trainer not running (count=$count) — restarting"
    kill_everything
    free_memory
    launch_iter
    continue
  fi

  if [ "$count" -ge "$EPISODES_PER_ITER" ]; then
    log "=== iter $iter reached $count episodes — cycling ==="
    kill_everything
    free_memory
    snapshot_and_analyze
    iter=$((iter + 1))
    if [ "$MAX_ITERATIONS" != "0" ] && [ "$iter" -ge "$MAX_ITERATIONS" ]; then
      log "MAX_ITERATIONS=$MAX_ITERATIONS reached; exiting"
      exit 0
    fi
    # auto_improve_iteration.py wrote next_env.sh; source it for next iter
    if [ -f "$PROJECT_ROOT/experiments/crater/next_env.sh" ]; then
      # shellcheck disable=SC1091
      source "$PROJECT_ROOT/experiments/crater/next_env.sh"
    fi
    launch_iter
    continue
  fi

  sleep "$POLL_INTERVAL_S"
done
