#!/usr/bin/env bash
# run_auto.sh — one-shot launcher for TerrainDreamer autonomous training.
#
# Starts Gazebo (headless or GUI) with the moon world + Jackal, then runs the
# Dreamer training loop. Handles all ROS env-var plumbing so you don't have to.
#
# Usage:
#   ./run_auto.sh                       # default: mare world, headless
#   ./run_auto.sh --env boulder_field   # pick a different moon env
#   ./run_auto.sh --gui                 # show Gazebo GUI (slower)
#   ./run_auto.sh --resume ckpt.pt      # resume training
#
# Requires:
#   - ROS 1 Noetic on /opt/ros/noetic
#   - catkin workspace built in ros_ws/
#   - venv in venv/ with project deps installed
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ---------------------------- parse args -----------------------------------
ENV_NAME="mare"
GUI="false"
RESUME=""
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)      ENV_NAME="$2"; shift 2 ;;
    --gui)      GUI="true";    shift ;;
    --headless) GUI="false";   shift ;;
    --resume)   RESUME="$2";   shift 2 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *) EXTRA_TRAIN_ARGS+=("$1"); shift ;;
  esac
done

echo "[run_auto] env=$ENV_NAME  gui=$GUI"

# ---------------------------- ROS plumbing ---------------------------------
source /opt/ros/noetic/setup.bash

if [[ ! -f "$PROJECT_DIR/ros_ws/devel/setup.bash" ]]; then
  echo "[run_auto] ros_ws not built — running catkin_make …"
  (cd "$PROJECT_DIR/ros_ws" && catkin_make)
fi
source "$PROJECT_DIR/ros_ws/devel/setup.bash"

# Inject our Velodyne + p3d extras into the Jackal URDF. The <env> tag inside
# moon_jackal.launch doesn't reliably propagate to child xacro invocations, so
# we export it at the shell level — this is the one that actually works.
export JACKAL_URDF_EXTRAS="$(rospack find terrain_dreamer_bringup)/urdf/jackal_vlp32.urdf.xacro"
export GAZEBO_MODEL_PATH="$(rospack find terrain_dreamer_bringup)/models:${GAZEBO_MODEL_PATH:-}"

# Headless display for GPU LiDAR raycast when gui=false.
if [[ "$GUI" == "false" ]]; then
  if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "[run_auto] xvfb-run missing — install with: sudo apt install xvfb"
    exit 1
  fi
  DISPLAY_WRAP=(xvfb-run -a --server-args="-screen 0 1280x720x24")
else
  DISPLAY_WRAP=()
fi

# ---------------------------- launch Gazebo --------------------------------
GAZEBO_LOG="$PROJECT_DIR/checkpoints_auto/gazebo.log"
mkdir -p "$(dirname "$GAZEBO_LOG")"

echo "[run_auto] launching Gazebo → $GAZEBO_LOG"
"${DISPLAY_WRAP[@]}" roslaunch terrain_dreamer_bringup moon_jackal.launch \
    env:="$ENV_NAME" gui:="$GUI" \
    > "$GAZEBO_LOG" 2>&1 &
GAZEBO_PID=$!

cleanup() {
  echo "[run_auto] cleaning up …"
  if kill -0 "$GAZEBO_PID" 2>/dev/null; then
    kill -INT "$GAZEBO_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$GAZEBO_PID" 2>/dev/null || true
  fi
  pkill -9 gzserver 2>/dev/null || true
  pkill -9 gzclient 2>/dev/null || true
  pkill -9 rosmaster 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for Gazebo to publish its critical topics.
echo "[run_auto] waiting for /ground_truth/odom …"
for i in $(seq 1 60); do
  if rostopic list 2>/dev/null | grep -q "/ground_truth/odom"; then
    echo "[run_auto] Gazebo up (after ${i}s)."
    break
  fi
  sleep 1
  if [[ $i -eq 60 ]]; then
    echo "[run_auto] timed out waiting for Gazebo. See $GAZEBO_LOG"
    exit 1
  fi
done

# ---------------------------- launch training ------------------------------
source "$PROJECT_DIR/venv/bin/activate" 2>/dev/null || \
  echo "[run_auto] venv not found — using system Python."

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

TRAIN_CMD=(python3 -u scripts/train_dreamer_auto.py)
if [[ -n "$RESUME" ]]; then
  TRAIN_CMD+=(--resume "$RESUME")
fi
TRAIN_CMD+=("${EXTRA_TRAIN_ARGS[@]}")

echo "[run_auto] starting: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"
