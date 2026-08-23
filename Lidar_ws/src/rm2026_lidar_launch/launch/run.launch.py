import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file) or {}


def generate_launch_description():
    pkg_share = get_package_share_directory("rm2026_lidar_launch")
    params = _load_yaml(os.path.join(pkg_share, "config", "params.yaml"))
    pipeline_cfg = params.get("lidar_pipeline", {})
    localization_cfg = pipeline_cfg.get("localization", {})
    dynamic_cloud_cfg = pipeline_cfg.get("dynamic_cloud", {})

    localization_source_leaf_size_arg = DeclareLaunchArgument(
        "localization_source_leaf_size",
        default_value=str(localization_cfg.get("source_voxel_leaf_size", 0.1)),
        description="localization配准source实时点云体素尺寸",
    )
    localization_target_leaf_size_arg = DeclareLaunchArgument(
        "localization_target_leaf_size",
        default_value=str(localization_cfg.get("target_voxel_leaf_size", 1.0)),
        description="localization配准target场地地图体素尺寸",
    )
    dynamic_cloud_leaf_size_arg = DeclareLaunchArgument(
        "dynamic_cloud_leaf_size",
        default_value=str(dynamic_cloud_cfg.get("voxel_leaf_size", 0.1)),
        description="dynamic_cloud体素下采样尺寸",
    )
    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz",
        default_value=str(pipeline_cfg.get("launch_rviz", True)).lower(),
        description="是否自动启动RViz",
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=str(pipeline_cfg.get("rviz_config", "")),
        description="RViz配置文件路径（支持绝对路径，若为相对路径则基于rm2026_lidar_launch的share目录）",
    )
    livox_launch_file_arg = DeclareLaunchArgument(
        "livox_launch_file",
        default_value=str(pipeline_cfg.get("livox_launch_file", "launch_ROS2/pcd_HAP_launch.py")),
        description="Livox驱动launch文件（支持绝对路径，若为相对路径则基于livox_ros_driver2的share目录）",
    )

    lidar_pipeline_launch = os.path.join(
        pkg_share, "launch", "lidar_pipeline.launch.py"
    )

    return LaunchDescription(
        [
            localization_source_leaf_size_arg,
            localization_target_leaf_size_arg,
            dynamic_cloud_leaf_size_arg,
            launch_rviz_arg,
            rviz_config_arg,
            livox_launch_file_arg,
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(lidar_pipeline_launch),
                launch_arguments={
                    "localization_source_leaf_size": LaunchConfiguration(
                        "localization_source_leaf_size"
                    ),
                    "localization_target_leaf_size": LaunchConfiguration(
                        "localization_target_leaf_size"
                    ),
                    "dynamic_cloud_leaf_size": LaunchConfiguration(
                        "dynamic_cloud_leaf_size"
                    ),
                    "launch_rviz": LaunchConfiguration("launch_rviz"),
                    "rviz_config": LaunchConfiguration("rviz_config"),
                    "livox_launch_file": LaunchConfiguration("livox_launch_file"),
                }.items(),
            ),
        ]
    )
