"""bridge.launch.py — start ros_gz_bridge with the right YAML for the
current ROS distro.

bridge_config:=auto picks bridge_jazzy.yaml for ROS_DISTRO in
{jazzy, rolling}, and bridge_humble.yaml for {humble, foxy, …}.
"""
from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PKG = "lunar_south_pole_gazebo"


def _resolve_config(context) -> str:
    choice = LaunchConfiguration("bridge_config").perform(context)
    if choice not in ("auto", "jazzy", "humble"):
        print(f"[bridge] unknown bridge_config={choice}; defaulting to auto",
              flush=True)
        choice = "auto"
    if choice == "auto":
        distro = os.environ.get("ROS_DISTRO", "").strip().lower()
        choice = "humble" if distro in {"humble", "foxy", "galactic", "iron"} \
                 else "jazzy"

    fname = f"bridge_{choice}.yaml"
    # Prefer the installed copy under share/<pkg>/config.
    try:
        share = Path(get_package_share_directory(PKG))
        installed = share / "config" / fname
        if installed.is_file():
            return str(installed)
    except Exception:
        pass
    # Source-tree fallback: launch file is at
    # ros2_ws/src/<pkg>/launch/bridge.launch.py; config/ is the
    # project root's config/ directory.
    here = Path(__file__).resolve().parent
    src_candidate = here.parents[3] / "config" / fname  # project/config
    if src_candidate.is_file():
        return str(src_candidate)
    # Last-resort: env var, then CWD.
    if env_dir := os.environ.get("LUNAR_BRIDGE_CONFIG_DIR"):
        cfg = Path(env_dir) / fname
        if cfg.is_file():
            return str(cfg)
    return str(Path.cwd() / "config" / fname)


def _launch_setup(context, *args, **kwargs):
    cfg = _resolve_config(context)
    print(f"[bridge] using config: {cfg}", flush=True)
    return [
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            output="screen",
            parameters=[{"config_file": cfg}],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("bridge_config", default_value="auto",
                              choices=["auto", "jazzy", "humble"]),
        OpaqueFunction(function=_launch_setup),
    ])
