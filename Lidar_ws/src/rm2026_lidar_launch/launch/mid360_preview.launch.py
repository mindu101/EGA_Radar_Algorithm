"""Preview a single Mid360 as PointCloud2 without the field localization pipeline."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("rm2026_lidar_launch")
    config_path = os.path.join(package_share, "config", "MID360_config.json")
    rviz_path = os.path.join(package_share, "config", "mid360_preview.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=config_path),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            Node(
                package="livox_ros_driver2",
                executable="livox_ros_driver2_node",
                name="mid360_lidar_publisher",
                output="screen",
                additional_env={
                    "LD_LIBRARY_PATH": os.pathsep.join(
                        part for part in (os.environ.get("LD_LIBRARY_PATH", ""), "/usr/local/lib") if part
                    )
                },
                parameters=[
                    {
                        "xfer_format": 0,
                        "multi_topic": 0,
                        "data_src": 0,
                        "publish_freq": 10.0,
                        "output_data_type": 0,
                        "frame_id": "livox_frame",
                        "user_config_path": LaunchConfiguration("config_file"),
                        "cmdline_input_bd_code": "livox0000000001",
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="mid360_preview_rviz",
                output="screen",
                arguments=["-d", rviz_path],
                condition=IfCondition(LaunchConfiguration("launch_rviz")),
            ),
        ]
    )
