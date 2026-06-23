"""Top-level launch — full sim: world + rover spawn + bridge + Phase-1
utility nodes + Phase-2 Jackal feature nodes (goals, chase camera,
route recorder, route visualizer).

Launch args (Phase 1):
  use_dem:=true|false           run the DEM world; otherwise empty world
  use_jackal:=true|false        spawn the rover (placeholder or official)
  use_bridge:=true|false        run ros_gz_bridge
  bridge_config:=auto|jazzy|humble
  x:=0.0  y:=0.0  yaw:=0.0      spawn pose (Z defaults to 'auto' via spawn_jackal)

Launch args (Phase 2):
  use_sensors:=true|false       (informational — sensors live in the rover SDF)
  use_goal_markers:=true|false  start goal_marker_node + goal_status_node
  use_chase_camera:=true|false  (informational — camera is in rover SDF; bridge
                                 entries are in bridge_*.yaml)
  use_route_recorder:=true|false start route_recorder_node
  record_route:=true|false      alias of use_route_recorder
  route_id:=auto                route_id passed to route_recorder_node
  use_route_visualizer:=true|false start route_visualizer_node
"""
from __future__ import annotations

from launch import LaunchDescription
import os
_HEADLESS = os.environ.get("GZ_HEADLESS", "0") == "1"
_GZ_EXTRA = " -r -s --headless-rendering" if _HEADLESS else " -r"
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


PKG = "lunar_south_pole_gazebo"


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare(PKG)

    use_dem = LaunchConfiguration("use_dem")
    use_jackal = LaunchConfiguration("use_jackal")
    use_bridge = LaunchConfiguration("use_bridge")
    bridge_config = LaunchConfiguration("bridge_config")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    yaw = LaunchConfiguration("yaw")

    use_goal_markers = LaunchConfiguration("use_goal_markers")
    use_route_recorder = LaunchConfiguration("use_route_recorder")
    use_route_visualizer = LaunchConfiguration("use_route_visualizer")
    route_id = LaunchConfiguration("route_id")

    world_path_dem = PathJoinSubstitution(
        [pkg_share, "worlds", "lunar_south_pole_dem.sdf"])
    world_path_empty = PathJoinSubstitution(
        [pkg_share, "worlds", "lunar_south_pole_empty.sdf"])
    goals_yaml = PathJoinSubstitution(
        [pkg_share, "config", "lunar_goals.yaml"])

    return LaunchDescription([
        # Phase 1 args
        DeclareLaunchArgument("use_dem", default_value="true"),
        DeclareLaunchArgument("use_jackal", default_value="true"),
        DeclareLaunchArgument("use_bridge", default_value="true"),
        DeclareLaunchArgument("bridge_config", default_value="auto"),
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),

        # Phase 2 args
        DeclareLaunchArgument("use_sensors", default_value="true",
                              description="informational — sensors live in rover SDF"),
        DeclareLaunchArgument("use_chase_camera", default_value="true",
                              description="informational — camera is in rover SDF"),
        DeclareLaunchArgument("use_goal_markers", default_value="true"),
        DeclareLaunchArgument("use_route_recorder", default_value="false"),
        DeclareLaunchArgument("record_route", default_value="false",
                              description="alias of use_route_recorder"),
        DeclareLaunchArgument("route_id", default_value="auto"),
        DeclareLaunchArgument("use_route_visualizer", default_value="true"),
        # Dreamer needs torch (only in the project venv). Off by default;
        # run via scripts/run_dreamer_on_jackal.sh.
        DeclareLaunchArgument("use_dreamer", default_value="false"),

        AppendEnvironmentVariable(
            name="GZ_SIM_RESOURCE_PATH",
            value=PathJoinSubstitution([pkg_share, "models"])),

        # Gazebo Sim
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py",
            ]),
            launch_arguments=[("gz_args", [world_path_dem, _GZ_EXTRA])],
            condition=IfCondition(use_dem),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py",
            ]),
            launch_arguments=[("gz_args", [world_path_empty, _GZ_EXTRA])],
            condition=UnlessCondition(use_dem),
        ),

        # Bridge
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                pkg_share, "/launch/bridge.launch.py",
            ]),
            launch_arguments=[("bridge_config", bridge_config)],
            condition=IfCondition(use_bridge),
        ),

        # Rover spawn (Phase 1 + Phase 2 sensors in SDF)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                pkg_share, "/launch/spawn_jackal.launch.py",
            ]),
            launch_arguments=[("x", x), ("y", y), ("yaw", yaw)],
            condition=IfCondition(use_jackal),
        ),

        # Phase 1 utility nodes
        Node(package=PKG, executable="dem_metadata_node",
             name="dem_metadata_node", output="screen"),
        Node(package=PKG, executable="terrain_manager_node",
             name="terrain_manager_node", output="screen"),
        Node(package=PKG, executable="hazard_status_node",
             name="hazard_status_node", output="screen"),
        Node(package=PKG, executable="dreamer_interface_node",
             name="dreamer_interface_node", output="screen",
             condition=IfCondition(LaunchConfiguration("use_dreamer"))),

        # Phase 2: goal markers + goal status (paired via the same YAML)
        Node(package=PKG, executable="goal_marker_node",
             name="goal_marker_node", output="screen",
             parameters=[{"goals_yaml": goals_yaml}],
             condition=IfCondition(use_goal_markers)),
        Node(package=PKG, executable="goal_status_node",
             name="goal_status_node", output="screen",
             parameters=[{"goals_yaml": goals_yaml}],
             condition=IfCondition(use_goal_markers)),

        # Phase 2: route recorder — gated by either use_route_recorder
        # OR record_route (alias kept per spec wording, combined here so
        # only ONE recorder is ever spawned).
        Node(package=PKG, executable="route_recorder_node",
             name="route_recorder_node", output="screen",
             parameters=[{"route_id": route_id}],
             condition=IfCondition(PythonExpression([
                 "'", use_route_recorder, "' == 'true' or '",
                 LaunchConfiguration("record_route"), "' == 'true'"]))),

        # Phase 2: route visualizer (live Path from /odom)
        Node(package=PKG, executable="route_visualizer_node",
             name="route_visualizer_node", output="screen",
             condition=IfCondition(use_route_visualizer)),
    ])
