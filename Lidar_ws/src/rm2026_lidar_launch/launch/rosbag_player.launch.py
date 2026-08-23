import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def _load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file) or {}


def _resolve_path(path_value, base_dir):
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base_dir, path_value))


def _load_default_rosbag_file(pkg_share):
    params = _load_yaml(os.path.join(pkg_share, "config", "params.yaml"))
    rosbag_player_cfg = params.get("rosbag_player", {})
    return str(rosbag_player_cfg.get("rosbag_file", "rosbag"))


def _create_rosbag_container(context, *args, **kwargs):
    pkg_share = get_package_share_directory("rm2026_lidar_launch")
    rosbag_file = LaunchConfiguration("rosbag_file").perform(context).strip()
    rosbag_file = _resolve_path(rosbag_file, pkg_share)

    rosbag_player_node = ComposableNode(
        package="rosbag_player",
        plugin="RosbagPlayer",
        name="rosbag_player_node",
        parameters=[{"rosbag_file": rosbag_file}],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    rosbag_player_container = ComposableNodeContainer(
        name="rosbag_player_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[rosbag_player_node],
        output="both",
        emulate_tty=True,
        on_exit=Shutdown(),
    )

    return [rosbag_player_container]


def generate_launch_description():
    pkg_share = get_package_share_directory("rm2026_lidar_launch")

    rosbag_file_arg = DeclareLaunchArgument(
        "rosbag_file",
        default_value=_load_default_rosbag_file(pkg_share),
        description="需要播放的rosbag路径（支持绝对路径，若为相对路径则基于rm2026_lidar_launch的share目录）",
    )

    return LaunchDescription(
        [rosbag_file_arg, OpaqueFunction(function=_create_rosbag_container)]
    )
