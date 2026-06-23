#!/usr/bin/env bash
# clean_and_train.sh — wrap train_lunar.sh with full cleanup before launch.
#
# Run this instead of train_lunar.sh whenever starting a new training run
# (especially after long-running sessions where gz transport degrades or
# trainer crashed mid-episode and left stale processes / log files).
#
# What it does:
#   1. Kills every trainer / viewer / sim / Clearpath / bridge process
#      from the previous session.
#   2. Clears checkpoint dir, /tmp/*.log files.
#   3. Forwards env vars (HEADLESS, TERRAIN, BASE_POLICY_MODE, BC_WEIGHT, ...)
#      to train_lunar.sh.
#   4. Always uses --restart-sim so gz comes up fresh — no >100-episode
#      gz transport degradation.
#
# Usage:
#   ./scripts/clean_and_train.sh                                  # defaults
#   HEADLESS=1 TERRAIN=rugged ./scripts/clean_and_train.sh
#   BASE_POLICY_MODE=reactive ./scripts/clean_and_train.sh
#
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log() { printf '\033[1;36m[clean+train]\033[0m %s\n' "$*"; }

log "=== 1. kill all related processes ==="
PATTERNS=(
  "python3 scripts/train_crater_ros"
  "python3 scripts/train_terrain_dreamer_ros"
  "python3 scripts/td_viewer.py"
  "ruby.*gz sim"
  "gz sim "
  "ros2 launch lunar_south_pole_gazebo"
  "ros2 launch ros_gz_sim"
  "spawn_clearpath_jackal"
  "robot_state_publisher.*j100_0001"
  "controller_manager.*j100_0001"
  "ekf_node"
  "imu_filter_madgwick"
  "interactive_marker_twist_server"
  "twist_mux"
  "joy_linux_node"
  "teleop_twist_joy_node"
  "parameter_bridge"
  "dem_metadata_node"
  "terrain_manager_node"
  "hazard_status_node"
  "static_transform_publisher"
)
for p in "${PATTERNS[@]}"; do
  pkill -9 -f "$p" 2>/dev/null || true
done
sleep 4
remaining=$(pgrep -af "train_crater|gz sim|ros2 launch.*lunar_south_pole|j100_0001" 2>/dev/null | wc -l)
log "remaining related procs: $remaining"

log "=== 2. clear training artifacts ==="
CKPT_DIR="$PROJECT_ROOT/checkpoints_auto/crater"
mkdir -p "$CKPT_DIR"
rm -f "$CKPT_DIR"/*.pt
log "checkpoints: $(ls "$CKPT_DIR" | wc -l) files left"

for f in /tmp/dreamer_train.log /tmp/lunar_sim.log /tmp/clearpath_spawn.log \
         /tmp/td_viewer.log /tmp/train_lunar_launcher.log /tmp/clock_bridge.log; do
  : > "$f" 2>/dev/null || true
done
log "/tmp log files cleared"

log "=== 3. launch train_lunar.sh --restart-sim ==="
cd "$PROJECT_ROOT"
# Forward whatever env the user set; default to safe values.
: "${HEADLESS:=1}"
: "${TERRAIN:=rugged}"
: "${BASE_POLICY_MODE:=reactive}"
export HEADLESS TERRAIN BASE_POLICY_MODE
[ -n "${BC_WEIGHT:-}" ]            && export BC_WEIGHT
[ -n "${DREAMER_ACTOR_WEIGHT:-}" ] && export DREAMER_ACTOR_WEIGHT
[ -n "${TIMEOUT_PENALTY:-}" ]      && export TIMEOUT_PENALTY
[ -n "${SUB_R_MAX:-}" ]            && export SUB_R_MAX

log "env: HEADLESS=$HEADLESS  TERRAIN=$TERRAIN  BASE_POLICY_MODE=$BASE_POLICY_MODE"

# Snapshot run metadata for reproducibility (git sha, env, config).
NEXT_ITER=$(ls "$PROJECT_ROOT/experiments/crater" 2>/dev/null \
  | grep -E '^iter_[0-9]+$' | sed 's/iter_//' | sort -n | tail -1)
NEXT_ITER=$(( ${NEXT_ITER:-0} + 1 ))
ITER_DIR="$PROJECT_ROOT/experiments/crater/iter_${NEXT_ITER}"
mkdir -p "$ITER_DIR"
log "snapshotting run metadata → $ITER_DIR"
python3 "$PROJECT_ROOT/scripts/snapshot_run.py" --iter-dir "$ITER_DIR" || true
echo "$NEXT_ITER" > "$PROJECT_ROOT/experiments/crater/.current_iter"
# If run_baseline.sh wrote a label, propagate it.
LABEL_FILE="$PROJECT_ROOT/experiments/crater/.next_label"
if [ -f "$LABEL_FILE" ]; then
  cp "$LABEL_FILE" "$ITER_DIR/baseline_label.txt"
  rm -f "$LABEL_FILE"
  log "iter $NEXT_ITER labeled '$(cat $ITER_DIR/baseline_label.txt)'"
fi

# Persist log inside the iter dir (NOT /tmp) so systemd-tmpfiles can't
# clean it mid-run. Symlink /tmp/dreamer_train.log to it for back-compat
# with analyze_training.py and viewer scripts.
PERSISTENT_LOG="$ITER_DIR/dreamer_train.log"
touch "$PERSISTENT_LOG"
ln -sf "$PERSISTENT_LOG" /tmp/dreamer_train.log
log "log persistent at $PERSISTENT_LOG (symlinked from /tmp/dreamer_train.log)"

log "starting train_lunar.sh --restart-sim ..."
exec ./train_lunar.sh --restart-sim
