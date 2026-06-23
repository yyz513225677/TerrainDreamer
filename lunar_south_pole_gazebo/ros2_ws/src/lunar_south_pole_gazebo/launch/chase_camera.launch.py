"""chase_camera.launch.py — start the ros_gz_bridge entries for the
chase camera topics.

The chase camera itself is a `<sensor type="camera">` inside the rover
SDF (so it follows the robot rigidly — exactly the racing-game
third-person behaviour). This launch only wires the gz→ROS image and
camera_info bridges. The mapping is also present in
`config/bridge_jazzy.yaml`; this launch is a convenience entrypoint
for users who want the camera bridge without the full system.
"""
from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    img_topic = LaunchConfiguration("image_topic")
    info_topic = LaunchConfiguration("camera_info_topic")

    return LaunchDescription([
        DeclareLaunchArgument(
            "image_topic",
            default_value="/lunar_jackal/chase_camera/image"),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/lunar_jackal/chase_camera/camera_info"),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="chase_camera_bridge",
            output="screen",
            arguments=[
                [img_topic, "@sensor_msgs/msg/Image[gz.msgs.Image"],
                [info_topic, "@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"],
            ],
        ),
    ])
