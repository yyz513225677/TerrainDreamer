#!/usr/bin/env bash
# train_lunar.sh — one-shot launcher for the TerrainDreamer training stack.
#
# Brings up the four things it needs (if they're not already running):
#   1. gz sim with the lunar_south_pole DEM world  (lunar_south_pole.launch.py)
#   2. /clock bridge (gz -> ROS) for use_sim_time   (ros_gz_bridge)
#   3. Clearpath J100 spawn                         (spawn_clearpath_jackal.launch.py)
#   4. The TerrainDreamer training driver           (scripts/train_crater_ros.py)
#
# Ctrl-C stops only the training driver and leaves the sim + Clearpath running,
# so you can re-run this script and pick up where you left off without
# re-cooking the OGRE shader cache.
#
# Algorithm module: crater/  (replaces the older lunar_dreamer_algorithm/
# which is kept only for the obs-builder utility functions used by the wrapper).
#
# Usage:
#   ./train_lunar.sh                  # start everything that isn't already up
#   ./train_lunar.sh --restart-sim    # kill + restart sim and Clearpath first
#   ./train_lunar.sh --tail           # if a trainer is already running, just tail

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$PROJECT_ROOT/lunar_south_pole_gazebo/ros2_ws"
LOG_DIR="${LOG_DIR:-/tmp}"
SIM_LOG="$LOG_DIR/lunar_sim.log"
SPAWN_LOG="$LOG_DIR/clearpath_spawn.log"
TRAIN_LOG="$LOG_DIR/dreamer_train.log"
TRAINER_SCRIPT="${TRAINER_SCRIPT:-scripts/train_crater_ros.py}"
TRAINER_PROC_NAME="${TRAINER_PROC_NAME:-train_crater_ros.py}"

ROS_SETUP="/opt/ros/jazzy/setup.bash"
WS_SETUP="$WS/install/setup.bash"

# TERRAIN=default → original landscape; TERRAIN=rugged → steep hills + river bed + boulders
TERRAIN="${TERRAIN:-default}"
# HEADLESS=1 → run gz sim server-only (no GUI window). Saves a lot of RAM/CPU
# and avoids triggering systemd-oomd against VSCode under heavy load.
HEADLESS="${HEADLESS:-0}"
case "$TERRAIN" in
  rugged)
    GZ_WORLD="lunar_south_pole_rugged"
    SIM_LAUNCH_FILE="lunar_south_pole_rugged.launch.py" ;;
  extreme)
    GZ_WORLD="lunar_south_pole_extreme"
    SIM_LAUNCH_FILE="lunar_south_pole_extreme.launch.py" ;;
  *)
    GZ_WORLD="lunar_south_pole"
    SIM_LAUNCH_FILE="lunar_south_pole.launch.py" ;;
esac
GZ_READY_TOPIC="/world/$GZ_WORLD/scene/info"

RESTART_SIM=false
TAIL_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --restart-sim) RESTART_SIM=true ;;
    --tail)        TAIL_ONLY=true ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "[train_lunar] unknown arg: $arg"; exit 1 ;;
  esac
done

log() { printf '\033[1;34m[train_lunar]\033[0m %s\n' "$*"; }

# ── Optional: tear down sim if asked ─────────────────────────────────────
if $RESTART_SIM; then
  log "Tearing down sim + Clearpath stack..."
  pkill -9 -f "$TRAINER_PROC_NAME" 2>/dev/null || true
  pkill -9 -f "train_lunar_dreamer_ros.py" 2>/dev/null || true   # legacy
  pkill -9 -f "spawn_clearpath_jackal|ros2 launch lunar_south_pole_gazebo" 2>/dev/null || true
  pkill -9 -f "j100_0001|controller_manager|robot_state_publisher|ekf_node|twist_mux|joy_linux|teleop_twist_joy|interactive_marker_twist_server|imu_filter_madgwick" 2>/dev/null || true
  pkill -9 -f "gz sim|ruby.*gz" 2>/dev/null || true
  pkill -9 -f "lunar_south_pole_gazebo/lib|ros_gz_bridge|static_transform_publisher" 2>/dev/null || true
  sleep 4
fi

# ── 1. gz sim ────────────────────────────────────────────────────────────
ensure_sim() {
  if timeout 4 gz topic -l 2>/dev/null | grep -q "^$GZ_READY_TOPIC$"; then
    log "gz sim is already up."
    return
  fi
  log "Starting gz sim..."
  : > "$SIM_LOG"
  # setsid puts the gz sim stack in its own session/process group, so
  # SIGHUP/SIGINT propagated to *this* script's session can't reach it.
  setsid env -i \
    HOME="$HOME" USER="$USER" DISPLAY="${DISPLAY:-:0}" \
    XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
    XDG_RUNTIME_DIR="/run/user/$(id -u)" \
    PATH=/opt/ros/jazzy/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/ros/jazzy/opt/rviz_ogre_vendor/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/lib \
    AMENT_PREFIX_PATH=/opt/ros/jazzy \
    PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages \
    ROS_DISTRO=jazzy \
    GZ_HEADLESS="$HEADLESS" \
    GZ_SIM_RESOURCE_PATH="$WS/install/lunar_south_pole_gazebo/share/lunar_south_pole_gazebo/models" \
    GZ_GUI_RESOURCE_PATH="$WS/src/lunar_south_pole_gazebo/gui" \
    nohup bash -c "source $ROS_SETUP; source $WS_SETUP; \
      exec ros2 launch lunar_south_pole_gazebo $SIM_LAUNCH_FILE \
        use_dem:=true use_jackal:=false use_bridge:=false \
        use_goal_markers:=false use_route_visualizer:=false \
        use_route_recorder:=false use_dreamer:=false" \
    >> "$SIM_LOG" 2>&1 < /dev/null &
  disown
  log "waiting for $GZ_READY_TOPIC (cold shader cache can take a few minutes)..."
  for _ in $(seq 1 180); do
    if timeout 4 gz topic -l 2>/dev/null | grep -q "^$GZ_READY_TOPIC$"; then
      break
    fi
    sleep 2
  done
  sleep 3   # let plugins finish initialising
  log "gz sim up."
}

# ── 1b. Bridge gz /clock → ROS /clock ───────────────────────────────────
# Without this the Clearpath controller_manager sees use_sim_time:=true but
# never receives a clock, so the diff_drive controller's update() never
# runs and /j100_0001/platform/odom stays at 0 Hz despite the publisher
# being registered. The Clearpath launch does NOT include this bridge.
ensure_clock_bridge() {
  if pgrep -f "parameter_bridge.*world/${GZ_WORLD}/clock" >/dev/null 2>&1; then
    log "clock bridge is already up."
    return
  fi
  log "Starting /clock bridge (gz -> ROS)..."
  setsid nohup bash -c "
    set +u
    source $ROS_SETUP
    exec ros2 run ros_gz_bridge parameter_bridge \
      /world/$GZ_WORLD/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock \
      --ros-args -r /world/$GZ_WORLD/clock:=/clock
  " > /tmp/clock_bridge.log 2>&1 < /dev/null &
  disown
  sleep 2
}

# ── 2. Clearpath J100 ────────────────────────────────────────────────────
ensure_clearpath() {
  # We require BOTH robot_state_publisher AND the gz-side controller_manager
  # to be alive. A lone robot_state_publisher from a previous failed/crashed
  # run is a common zombie that survives kill cycles and used to make this
  # check pass falsely; checking for the controller_manager spawner avoids
  # that. We also try to clean such zombies before deciding.
  local rsp_pid cm_pid
  rsp_pid=$(pgrep -f "robot_state_publisher.*__ns:=/j100_0001" 2>/dev/null | head -1)
  cm_pid=$(pgrep -f "controller_manager.*__ns:=/j100_0001" 2>/dev/null | head -1)
  if [[ -n "$rsp_pid" && -n "$cm_pid" ]]; then
    log "Clearpath stack is already running (rsp=$rsp_pid, cm=$cm_pid)."
    return
  fi
  if [[ -n "$rsp_pid" && -z "$cm_pid" ]]; then
    log "Found orphan robot_state_publisher (pid=$rsp_pid) without controller_manager — killing it."
    kill -9 "$rsp_pid" 2>/dev/null
    sleep 2
  fi
  log "Spawning Clearpath J100..."
  : > "$SPAWN_LOG"
  setsid nohup bash -c '
    # Clearpath generator imports python3-apt from /usr/lib/python3/dist-packages;
    # any project venv that hides system site-packages breaks it.
    unset VIRTUAL_ENV PYTHONHOME
    export PATH="/usr/bin:/usr/local/bin:/opt/ros/jazzy/bin:$PATH"
    export PYTHONPATH="/usr/lib/python3/dist-packages:'"$WS"'/install/lunar_south_pole_gazebo/lib/python3.12/site-packages:/opt/ros/jazzy/lib/python3.12/site-packages"
    source '"$ROS_SETUP"'
    source '"$WS_SETUP"'
    exec ros2 launch lunar_south_pole_gazebo spawn_clearpath_jackal.launch.py \
      setup_path:='"$HOME"'/clearpath/ \
      world:='"$GZ_WORLD"' x:=0.0 y:=0.0 z:=4.25 yaw:=0.0 rviz:=false
  ' >> "$SPAWN_LOG" 2>&1 < /dev/null &
  disown

  log "waiting for controllers..."
  for _ in $(seq 1 60); do
    if grep -qE "Configured and activated platform_velocity_controller|Failed to configure|Traceback" "$SPAWN_LOG" 2>/dev/null; then
      break
    fi
    sleep 3
  done
  if grep -qE "Failed to configure|Traceback" "$SPAWN_LOG"; then
    log "Clearpath spawn failed — see $SPAWN_LOG"
    exit 1
  fi
  # Unpause in case the world started paused.
  timeout 4 gz service -s "/world/$GZ_WORLD/control" \
    --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean \
    --timeout 2000 --req "pause: false" >/dev/null 2>&1 || true
  log "Clearpath controllers active."
}

# ── 3. Training driver ──────────────────────────────────────────────────
launch_training() {
  if pgrep -f "python3.*$TRAINER_PROC_NAME" >/dev/null 2>&1; then
    if $TAIL_ONLY; then
      log "Training driver already running — tailing $TRAIN_LOG"
      exec tail -F "$TRAIN_LOG"
    fi
    log "Training driver is already running (pid $(pgrep -f "python3.*$TRAINER_PROC_NAME" | head -1)). Tailing log; Ctrl-C just detaches."
    exec tail -F "$TRAIN_LOG"
  fi
  if $TAIL_ONLY; then
    log "No trainer running. Exiting."
    exit 0
  fi

  log "Launching Dreamer training driver — logging to $TRAIN_LOG"
  : > "$TRAIN_LOG"
  trap 'log "stopping trainer..."; pkill -TERM -f "$TRAINER_PROC_NAME" 2>/dev/null; sleep 2; pkill -9 -f "$TRAINER_PROC_NAME" 2>/dev/null; exit 0' INT TERM
  # ROS setup.bash references AMENT_TRACE_SETUP_FILES and other optional vars,
  # so set -u must be off while sourcing it.
  (
    set +u
    source "$ROS_SETUP"
    source "$WS_SETUP"
    cd "$PROJECT_ROOT"
    export GZ_WORLD="$GZ_WORLD"        # trainer picks the obstacle mask
    exec python3 "$TRAINER_SCRIPT"
  ) 2>&1 | tee -a "$TRAIN_LOG"
}

# ── 4. Real-time viewer (LiDAR / IMU / path) ────────────────────────────
# Independent process; survives trainer restarts. Set VIEWER=0 to skip.
# HEADLESS=1 implies VIEWER=0 (no GUI at all).
VIEWER="${VIEWER:-1}"
if [ "$HEADLESS" = "1" ]; then VIEWER=0; fi
ensure_viewer() {
  if [ "$VIEWER" != "1" ]; then
    log "viewer disabled (VIEWER=$VIEWER)"
    return
  fi
  if pgrep -f "python3.*scripts/td_viewer.py" >/dev/null 2>&1; then
    log "viewer already running."
    return
  fi
  if [ -z "${DISPLAY:-}" ]; then
    log "no DISPLAY — skipping viewer (set VIEWER=0 to silence)"
    return
  fi
  log "Launching td_viewer (LiDAR / IMU / path)..."
  setsid nohup bash -c "
    set +u
    source $ROS_SETUP
    cd $PROJECT_ROOT
    exec python3 scripts/td_viewer.py
  " > /tmp/td_viewer.log 2>&1 < /dev/null &
  disown
  sleep 1
}

ensure_sim
ensure_clock_bridge
ensure_clearpath
ensure_viewer
launch_training
