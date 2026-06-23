#!/usr/bin/env bash
# run.sh — TerrainDreamer single-launcher (replaces run_human.sh / run_auto.sh).
#
# Two-window UI (gz sim runs HEADLESS — no second OGRE pipeline for the GUI):
#   Window 1 (left half of primary monitor): RViz2 with chase camera locked
#                                              behind the Jackal, lunar
#                                              terrain mesh, no-go zones,
#                                              live LiDAR cloud, odom trail
#   Window 2 (right half of primary monitor): matplotlib LiDAR + IMU + trail
#                                              (+ control hints in human mode)
#
# Modes:
#   --mode human   pynput key-driving + demo recording (default)
#   --mode auto    autonomous Dreamer training
#
# Usage:
#   ./run.sh                          # human, varied world, RViz2 + mpl
#   ./run.sh --mode auto              # autonomous training
#   ./run.sh --mode auto --resume ckpt_latest.pt
#   ./run.sh --env mare --headless    # no UI windows at all
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ---------------------------- args -----------------------------------------
MODE="human"
ENV_NAME="varied"
GUI="true"
OUT_DIR="demos"
RESUME=""
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)     MODE="$2"; shift 2 ;;
    --env)      ENV_NAME="$2"; shift 2 ;;
    --gui)      GUI="true";    shift ;;
    --headless) GUI="false";   shift ;;
    --out)      OUT_DIR="$2";  shift 2 ;;
    --resume)   RESUME="$2";   shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *) EXTRA_TRAIN_ARGS+=("$1"); shift ;;
  esac
done
case "$MODE" in
  human|auto) ;;
  *) echo "[run] unknown mode '$MODE' (must be human|auto)"; exit 1 ;;
esac
echo "[run] mode=$MODE  env=$ENV_NAME  gui=$GUI"

# ---------------------------- env scrub ------------------------------------
unset CMAKE_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION VIRTUAL_ENV
hash -r
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | \
      grep -vE "^(/snap/|/opt/ros/noetic)" | paste -sd: -)
  export LD_LIBRARY_PATH
fi

set +u
source /opt/ros/jazzy/setup.bash
if [[ ! -f "$PROJECT_DIR/ros_ws/install/setup.bash" ]]; then
  echo "[run] ros_ws not built — running colcon build …"
  (cd "$PROJECT_DIR/ros_ws" && colcon build --packages-select terrain_dreamer_bringup)
fi
source "$PROJECT_DIR/ros_ws/install/setup.bash"
set -u

if ! command -v gz >/dev/null 2>&1; then
  echo "[run] gz (Gazebo Sim) not found. Install gz-harmonic + ros-jazzy-ros-gz."
  exit 1
fi

# ---------------------------- kill stale procs -----------------------------
pkill -9 -f "ros2 launch terrain_dreamer|gz sim|ruby.*gz_tools|parameter_bridge|robot_state_publisher|relay /odom|scripts/realtime_viewer.py|scripts/human_drive.py|scripts/train_dreamer_auto.py" 2>/dev/null || true
sleep 1

mkdir -p "$PROJECT_DIR/checkpoints_auto" "$PROJECT_DIR/$OUT_DIR"
GAZEBO_LOG="$PROJECT_DIR/checkpoints_auto/gazebo.log"

# ---------------------------- launch sim -----------------------------------
# gz sim runs HEADLESS regardless of $GUI. The GUI/headless flag now controls
# whether the two user-facing windows (RViz2 + matplotlib) are opened —
# headless mode skips them.
echo "[run] launching gz sim (server-only) + bridge + terrain markers → $GAZEBO_LOG"
# __EGL_VENDOR_LIBRARY_FILENAMES forces the glvnd EGL loader to pick the
# NVIDIA vendor JSON only — without this, gz-sim's headless EGL context
# creation tries the Mesa loader first, fails with "OpenGL 3.3 not supported"
# and segfaults. __GLX_VENDOR_LIBRARY_NAME does the same for GLX.
env -i HOME="$HOME" USER="$USER" \
    DISPLAY="${DISPLAY:-:0}" XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
    PATH=/usr/local/bin:/usr/bin:/bin \
    TERM="${TERM:-xterm-256color}" LANG="${LANG:-en_US.UTF-8}" \
    TD_PROJECT_ROOT="$PROJECT_DIR" \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    bash -c "
        set +u
        source /opt/ros/jazzy/setup.bash
        source '$PROJECT_DIR/ros_ws/install/setup.bash'
        ros2 launch terrain_dreamer_bringup moon_jackal.launch.py \
            env:='$ENV_NAME' gui:='$GUI'
    " > "$GAZEBO_LOG" 2>&1 &
GAZEBO_PID=$!

# ---------------------------- cleanup --------------------------------------
# All PID slots declared up front so `set -u` doesn't trip when cleanup runs
# before some of the optional sub-processes were launched (e.g. on early
# timeout, RViz2 / layout / UI haven't started yet).
UI_PID=""
LAYOUT_PID=""
RVIZ_PID=""

cleanup() {
  echo "[run] cleaning up …"
  for pid in "$UI_PID" "$LAYOUT_PID" "$RVIZ_PID" "$GAZEBO_PID"; do
    [[ -n "$pid" ]] && kill -INT "$pid" 2>/dev/null || true
  done
  sleep 2
  pkill -9 -f "gz sim|ruby.*gz_tools|parameter_bridge|robot_state_publisher|relay /odom|ros2 launch terrain_dreamer|scripts/human_drive.py|scripts/train_dreamer_auto.py|scripts/realtime_viewer.py|scripts/terrain_marker_publisher.py|rviz2" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---------------------------- wait for sim ---------------------------------
# Empirically on cold OGRE caches the gz-sim Sensors plugin needs ~5-10 min
# to compile every Hlms PBS shader and finish the first scene render.
# Warm-cache runs do it in ~15-30 s. 360 iterations × ~2 s = ~720 s ceiling,
# so first launch after a reboot won't time out.
echo "[run] waiting for /imu/data to publish (cold-cache first run can take 8-10 min) …"
# `timeout 3` (not 1): ros2 topic echo's DDS discovery + node init takes
# ~1.2 s on this stack — `timeout 1` always killed it before subscription
# was established, so the wait loop never broke out even when /imu/data was
# in fact publishing.
for i in $(seq 1 240); do
  if timeout 3 ros2 topic echo --once /imu/data > /dev/null 2>&1; then
    echo "[run] Gazebo Sim up (after ~${i}×4 s)."
    break
  fi
  sleep 1
  if [[ $i -eq 240 ]]; then
    echo "[run] timed out waiting for Gazebo Sim. See $GAZEBO_LOG"
    exit 1
  fi
done

# ---------------------------- launch RViz2 (window 1) ----------------------
if [[ "$GUI" == "true" ]] && command -v rviz2 >/dev/null 2>&1; then
  RVIZ_CFG="$PROJECT_DIR/ros_ws/install/terrain_dreamer_bringup/share/terrain_dreamer_bringup/rviz/training_chase.rviz"
  if [[ -f "$RVIZ_CFG" ]]; then
    echo "[run] launching RViz2 chase view (window 1 of 2)"
    # env -i to keep VS Code's snap libpthread out of RViz2's process (same
    # bug that crashed gz sim with "symbol lookup error: __libc_pthread_init").
    env -i HOME="$HOME" USER="$USER" \
        DISPLAY="${DISPLAY:-:0}" XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
        PATH=/usr/local/bin:/usr/bin:/bin \
        TERM="${TERM:-xterm-256color}" LANG="${LANG:-en_US.UTF-8}" \
        __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
        __GLX_VENDOR_LIBRARY_NAME=nvidia \
        bash -c "
            set +u
            source /opt/ros/jazzy/setup.bash
            source '$PROJECT_DIR/ros_ws/install/setup.bash'
            exec rviz2 -d '$RVIZ_CFG'
        " > "$PROJECT_DIR/checkpoints_auto/rviz.log" 2>&1 &
    RVIZ_PID=$!
  else
    echo "[run] RViz2 config missing at $RVIZ_CFG — skipping RViz2."
  fi
fi

# ---------------------------- side-by-side windows -------------------------
# Split the primary monitor 50/50: RViz2 left, matplotlib right.
if [[ "$GUI" == "true" ]] && command -v wmctrl >/dev/null 2>&1 \
                          && command -v xdotool >/dev/null 2>&1; then
  ( bash -c '
      sleep 6   # wait for both windows to open before resizing
      geom=$(xrandr | awk "/ connected primary / {print \$4; exit}")
      pri_w=$(echo "$geom" | cut -d"x" -f1)
      pri_h=$(echo "$geom" | cut -d"x" -f2 | cut -d"+" -f1)
      pri_x=$(echo "$geom" | cut -d"+" -f2)
      pri_y=$(echo "$geom" | cut -d"+" -f3)
      half=$(( pri_w / 2 ))
      gap=20
      half_left=$(( half - gap ))
      half_right=$(( half - gap ))
      usable_h=$(( pri_h - 140 ))
      top=$(( pri_y + 60 ))

      rviz_id=$(wmctrl -l | grep -iE "rviz|RViz" | head -1 | awk "{print \$1}")
      mpl_id=$(wmctrl -l | grep -iE "TerrainDreamer.*human|TerrainDreamer.*live|live IMU" | head -1 | awk "{print \$1}")
      for id in "$rviz_id" "$mpl_id"; do
        [[ -n "$id" ]] && wmctrl -ir "$id" -b remove,maximized_vert,maximized_horz 2>/dev/null
      done
      [[ -n "$rviz_id" ]] && wmctrl -ir "$rviz_id" -e "0,${pri_x},${top},${half_left},${usable_h}"
      mpl_x=$(( pri_x + half + gap/2 ))
      [[ -n "$mpl_id" ]] && wmctrl -ir "$mpl_id" -e "0,${mpl_x},${top},${half_right},${usable_h}"
      echo "[layout] left half: rviz2=$rviz_id, right half: mpl=$mpl_id"
  ' &
  ) > "$PROJECT_DIR/checkpoints_auto/layout.log" 2>&1
  LAYOUT_PID=$!
fi

# ---------------------------- launch UI ------------------------------------
source "$PROJECT_DIR/venv/bin/activate" 2>/dev/null || \
  echo "[run] venv not found — using system Python."
PY_SITE="/opt/ros/jazzy/lib/python3.12/site-packages"
export PYTHONPATH="$PROJECT_DIR/src:$PY_SITE:${PYTHONPATH:-}"
export TD_ENV_NAME="$ENV_NAME"

if [[ "$MODE" == "human" ]]; then
  echo "[run] launching human-drive UI (window 2 of 2)"
  python3 -u scripts/human_drive.py --out "$OUT_DIR"
else
  echo "[run] launching realtime IMU + LiDAR viewer (window 2 of 2)"
  python3 -u scripts/realtime_viewer.py \
      > "$PROJECT_DIR/checkpoints_auto/viewer.log" 2>&1 &
  UI_PID=$!

  TRAIN_CMD=(python3 -u scripts/train_dreamer_auto.py)
  [[ -n "$RESUME" ]] && TRAIN_CMD+=(--resume "$RESUME")
  TRAIN_CMD+=("${EXTRA_TRAIN_ARGS[@]}")
  echo "[run] starting trainer: ${TRAIN_CMD[*]}"
  "${TRAIN_CMD[@]}"
fi
