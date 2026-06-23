"""mission_loop.launch.py — random-goal + flag + nav + route-record loop
on top of an already-running Clearpath J100 spawn.

Run order:
  Terminal A: gz sim world (lunar_south_pole) — your existing launch
  Terminal B: ros2 launch lunar_south_pole_gazebo spawn_clearpath_jackal.launch.py
              setup_path:=$HOME/clearpath/ world:=lunar_south_pole z:=10.0
  Terminal C: ros2 launch lunar_south_pole_gazebo mission_loop.launch.py
"""
from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PKG = "lunar_south_pole_gazebo"


def generate_launch_description() -> LaunchDescription:
    world = LaunchConfiguration("world")
    min_r = LaunchConfiguration("min_radius_m")
    max_r = LaunchConfiguration("max_radius_m")
    reach = LaunchConfiguration("reach_radius_m")
    use_nav = LaunchConfiguration("use_fallback_nav")
    use_rec = LaunchConfiguration("use_route_recorder")
    routes_dir = LaunchConfiguration("routes_dir")

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="lunar_south_pole"),
        DeclareLaunchArgument("min_radius_m", default_value="5.0"),
        DeclareLaunchArgument("max_radius_m", default_value="15.0"),
        DeclareLaunchArgument("reach_radius_m", default_value="1.5"),
        DeclareLaunchArgument("use_fallback_nav", default_value="true"),
        DeclareLaunchArgument("use_route_recorder", default_value="true"),
        DeclareLaunchArgument(
            "routes_dir",
            default_value="/home/rickslab3/Documents/Leo/terrain_dreamer/"
                          "lunar_south_pole_gazebo/data/routes"),

        Node(
            package=PKG, executable="mission_manager_node",
            name="mission_manager_node", output="screen",
            parameters=[{
                "world": world,
                "min_radius_m": min_r,
                "max_radius_m": max_r,
                "reach_radius_m": reach,
                "settle_time_s": 3.0,
                "odom_topic": "/j100_0001/platform/odom",
            }],
        ),
        Node(
            package=PKG, executable="fallback_nav_node",
            name="fallback_nav_node", output="screen",
            parameters=[{
                "odom_topic": "/j100_0001/platform/odom",
                "scan_topic": "/j100_0001/sensors/lidar3d_0/scan",
                "cmd_topic":  "/j100_0001/cmd_vel",
                "cmd_stamped": True,
                "max_linear_mps": 0.4,
                "max_angular_radps": 0.7,
                "rate_hz": 10.0,
            }],
            condition=IfCondition(use_nav),
        ),
        Node(
            package=PKG, executable="route_recorder_node",
            name="route_recorder_node", output="screen",
            parameters=[{
                "robot_name": "j100_0001",
                "odom_topic": "/j100_0001/platform/odom",
                "imu_topic":  "/j100_0001/sensors/imu_0/data",
                "cmd_vel_topic": "/j100_0001/cmd_vel",
                "cmd_vel_stamped": True,
                "routes_dir": routes_dir,
                "route_id": "auto",
            }],
            condition=IfCondition(use_rec),
        ),
    ])
