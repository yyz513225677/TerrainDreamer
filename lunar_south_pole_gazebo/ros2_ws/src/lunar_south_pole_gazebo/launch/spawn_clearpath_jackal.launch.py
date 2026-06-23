"""spawn_clearpath_jackal.launch.py

Spawn an OFFICIAL Clearpath J100 Jackal (generated from
$setup_path/robot.yaml) into a Gazebo Sim world that is ALREADY
RUNNING. Does NOT start a new gz process and does NOT touch the
existing map geometry — only calls `clearpath_gz robot_spawn.launch.py`
with the user-supplied pose.

Launch args (per project spec):
  setup_path   default: $HOME/clearpath/                 — directory holding robot.yaml
  world        default: $GZ_WORLD or lunar_south_pole     — must match the running world
  x, y, z, yaw default: 0, 0, 1.0, 0
  rviz         default: false                            — start RViz alongside

Discover the running world's name (so you can pass --world correctly):
    gz topic -l | grep '/world/'

Build target for ROS_DISTRO is jazzy by default; export ROS_DISTRO=humble
to use the same launch on Humble (the Clearpath spawn launch is
distro-agnostic).
"""
from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    setup_path = LaunchConfiguration("setup_path")
    world = LaunchConfiguration("world")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")
    yaw = LaunchConfiguration("yaw")
    rviz = LaunchConfiguration("rviz")

    clearpath_gz_share = FindPackageShare("clearpath_gz")

    return LaunchDescription([
        DeclareLaunchArgument(
            "setup_path",
            default_value=[EnvironmentVariable("HOME"), "/clearpath/"],
            description="Directory containing robot.yaml (Clearpath config)."),
        DeclareLaunchArgument(
            "world",
            default_value="lunar_south_pole",
            description="Name of the already-running Gazebo Sim world."),
        DeclareLaunchArgument("x",   default_value="0.0"),
        DeclareLaunchArgument("y",   default_value="0.0"),
        DeclareLaunchArgument("z",   default_value="1.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument(
            "rviz", default_value="false",
            description="Start RViz alongside the spawn."),

        # Include Clearpath's official robot_spawn launch (DOES NOT
        # start a new gz sim — only spawns the URDF and starts the
        # ros_gz_bridge for the platform).
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                clearpath_gz_share, "/launch/robot_spawn.launch.py",
            ]),
            launch_arguments=[
                ("setup_path", setup_path),
                ("world", world),
                ("x", x), ("y", y), ("z", z), ("yaw", yaw),
                ("rviz", rviz),
            ],
        ),
    ])
