#!/usr/bin/env bash
# Run the dreamer_interface_node against the running Jackal sim.
#
# Why a script: the node needs BOTH `rclpy` (system Python @ /opt/ros/jazzy)
# and `torch` + `terrain_dreamer.*` (project venv). Neither python alone
# has both, so we point the venv's interpreter at the system ROS site-packages
# via PYTHONPATH.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
PKG_DIR="${REPO_ROOT}/lunar_south_pole_gazebo"
VENV_PY="${REPO_ROOT}/venv/bin/python3"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS_PREFIX="/opt/ros/${ROS_DISTRO}"
ROS_PY_SITE="${ROS_PREFIX}/lib/python3.12/site-packages"

CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints_auto/best_model.pt}"
GOALS_YAML="${GOALS_YAML:-${PKG_DIR}/config/lunar_goals.yaml}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "[run_dreamer] venv python not found at $VENV_PY" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[run_dreamer] checkpoint not found at $CHECKPOINT" >&2
  exit 2
fi
# ROS setup scripts reference unset vars internally — disable nounset
# for the duration of the source. We re-enable strict mode afterward.
set +u
if ! source "${ROS_PREFIX}/setup.bash"; then
  echo "[run_dreamer] failed to source $ROS_PREFIX/setup.bash" >&2
  exit 3
fi
WS_SETUP="${PKG_DIR}/ros2_ws/install/setup.bash"
if [[ -f "$WS_SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$WS_SETUP" || true
fi
set -u

NODE_PY="${PKG_DIR}/ros2_ws/src/lunar_south_pole_gazebo/lunar_south_pole_gazebo/dreamer_interface_node.py"

export TERRAIN_DREAMER_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="${PKG_DIR}/ros2_ws/src/lunar_south_pole_gazebo:${ROS_PY_SITE}:${PYTHONPATH:-}"

echo "[run_dreamer] venv python : $VENV_PY"
echo "[run_dreamer] checkpoint  : $CHECKPOINT"
echo "[run_dreamer] goals YAML  : $GOALS_YAML"
echo "[run_dreamer] PYTHONPATH  : $PYTHONPATH"
echo

exec "$VENV_PY" "$NODE_PY" --ros-args \
  -p checkpoint_path:="$CHECKPOINT" \
  -p goals_yaml:="$GOALS_YAML" \
  "$@"
