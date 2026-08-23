import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode


def _load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file) or {}


def _resolve_path(path_value, base_dir):
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base_dir, path_value))


def _is_true(value):
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def _cfg_list(cfg, key, default):
    value = cfg.get(key, default)
    if not isinstance(value, list) or len(value) != len(default):
        return default
    return value


def _cfg_box_list(cfg, key):
    """校验变长轴对齐区域列表，并展平为ROS2支持的double数组。"""
    value = cfg.get(key, [])
    if not isinstance(value, list):
        raise RuntimeError(f"{key}必须是区域列表")

    flattened = []
    for index, box in enumerate(value, start=1):
        if not isinstance(box, list) or len(box) != 6:
            raise RuntimeError(
                f"{key}第{index}个区域必须包含6个值: "
                "[x_min, x_max, y_min, y_max, z_min, z_max]"
            )
        try:
            box = [float(item) for item in box]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{key}第{index}个区域包含非数值参数") from exc
        if not (box[0] < box[1] and box[2] < box[3] and box[4] < box[5]):
            raise RuntimeError(
                f"{key}第{index}个区域边界非法，要求各维min < max且不能包含NaN"
            )
        flattened.extend(box)
    return flattened


def _build_actions(context, *args, **kwargs):
    pkg_share = get_package_share_directory("rm2026_lidar_launch")
    params_file = LaunchConfiguration("params_file").perform(context).strip()
    params_file = _resolve_path(params_file, pkg_share)
    params = _load_yaml(params_file)
    common_cfg = params.get("common", {})
    test_cfg = params.get("rosbag_lidar_fusion_test", {})
    pipeline_cfg = params.get("lidar_pipeline", {})
    # 回放与正式启动共用同一套算法参数，测试配置只保留 rosbag 播放控制项。
    localization_cfg = pipeline_cfg.get("localization", {})
    gicp_cfg = localization_cfg.get("gicp", {})
    ndt_cfg = localization_cfg.get("ndt", {})
    ndt_fine_cfg = ndt_cfg.get("fine", {})
    dynamic_cloud_cfg = pipeline_cfg.get("dynamic_cloud", {})
    cluster_cfg = pipeline_cfg.get("cluster", {})

    extrinsics_file = str(common_cfg.get("extrinsics_file", "config/lidar_extrinsics.yaml"))
    params_root = os.path.dirname(os.path.dirname(params_file))
    extrinsics_path = _resolve_path(extrinsics_file, params_root)
    if not os.path.exists(extrinsics_path):
        extrinsics_path = _resolve_path(extrinsics_file, pkg_share)
    extrinsics_cfg = _load_yaml(extrinsics_path)

    tf_cfg = extrinsics_cfg.get("lidar_tf", {})
    translation = tf_cfg.get("translation", [])
    rpy = tf_cfg.get("rpy", [])
    invert = bool(tf_cfg.get("invert", False))

    if len(translation) != 3 or len(rpy) != 3:
        raise RuntimeError(
            "lidar_extrinsics.yaml must provide translation[3] and rpy[3]"
        )

    map_cfg = common_cfg.get("map", {})
    map_package = str(map_cfg.get("package", "dynamic_cloud"))
    map_relative_path = str(map_cfg.get("relative_path", "config/RM2026.pcd"))
    map_base = get_package_share_directory(map_package)
    map_path = _resolve_path(map_relative_path, map_base)

    rosbag_file = LaunchConfiguration("rosbag_file").perform(context).strip()
    rosbag_file = _resolve_path(rosbag_file, pkg_share)
    playback_rate = float(LaunchConfiguration("playback_rate").perform(context))
    playback_delay_sec = float(
        LaunchConfiguration("playback_delay_sec").perform(context)
    )
    loop_playback = _is_true(
        LaunchConfiguration("loop_playback").perform(context)
    )
    start_rosbag = _is_true(
        LaunchConfiguration("start_rosbag").perform(context)
    )
    start_offset_sec = float(
        LaunchConfiguration("start_offset_sec").perform(context)
    )

    if playback_rate <= 0.0:
        raise RuntimeError("playback_rate must be greater than 0")
    if playback_delay_sec < 0.0:
        raise RuntimeError("playback_delay_sec must not be negative")
    if start_offset_sec < 0.0:
        raise RuntimeError("start_offset_sec must not be negative")

    rosbag_play_cmd = [
        "ros2",
        "bag",
        "play",
        rosbag_file,
        "--rate",
        str(playback_rate),
        "--delay",
        str(playback_delay_sec),
        "--start-offset",
        str(start_offset_sec),
        "--clock",
        "100.0",
        "--disable-keyboard-controls",
    ]
    if loop_playback:
        rosbag_play_cmd.append("--loop")

    rosbag_player = ExecuteProcess(
        cmd=rosbag_play_cmd,
        name="rosbag_player",
        output="screen",
        emulate_tty=True,
    )

    localization = ComposableNode(
        package="localization",
        plugin="tdt_radar::Localization",
        name="localization",
        parameters=[
            {
                "map_path": map_path,
                "use_sim_time": True,
                "accumulate_time": int(localization_cfg.get("accumulate_time", 20)),
                "backend": str(localization_cfg.get("backend", "gicp")),
                "enable_registration": bool(
                    localization_cfg.get("enable_registration", True)
                ),
                "continuous_registration": bool(
                    localization_cfg.get("continuous_registration", False)
                ),
                "accept_unconverged_result": bool(
                    localization_cfg.get("accept_unconverged_result", False)
                ),
                "registration_interval_sec": float(
                    localization_cfg.get("registration_interval_sec", 0.0)
                ),
                "max_registration": int(
                    localization_cfg.get("max_registration", 100)
                ),
                "enable_background_filter": bool(
                    localization_cfg.get("enable_background_filter", False)
                ),
                "background_angular_bin_deg": float(
                    localization_cfg.get("background_angular_bin_deg", 0.1)
                ),
                "background_range_tolerance": float(
                    localization_cfg.get("background_range_tolerance", 0.1)
                ),
                "background_candidate_confirm_count": int(
                    localization_cfg.get(
                        "background_candidate_confirm_count", 2
                    )
                ),
                "background_candidate_tolerance": float(
                    localization_cfg.get(
                        "background_candidate_tolerance", 0.15
                    )
                ),
                "registration_reset_auto": bool(
                    localization_cfg.get("registration_reset_auto", True)
                ),
                "registration_reset_inlier_distance": float(
                    localization_cfg.get(
                        "registration_reset_inlier_distance", 0.25
                    )
                ),
                "registration_reset_inlier_ratio_threshold": float(
                    localization_cfg.get(
                        "registration_reset_inlier_ratio_threshold", 0.3
                    )
                ),
                "registration_reset_min_points": int(
                    localization_cfg.get("registration_reset_min_points", 200)
                ),
                "registration_reset_min_age_frames": int(
                    localization_cfg.get(
                        "registration_reset_min_age_frames", 10
                    )
                ),
                "registration_reset_freeze_frames": int(
                    localization_cfg.get("registration_reset_freeze_frames", 5)
                ),
                "initial_translation": translation,
                "initial_rpy": rpy,
                "initial_extrinsics_are_parent_to_child": not invert,
                "use_initial_guess": bool(localization_cfg.get("use_initial_guess", True)),
                "fitness_threshold": float(localization_cfg.get("fitness_threshold", 0.1)),
                "transformation_epsilon": float(
                    localization_cfg.get("transformation_epsilon", 1e-4)
                ),
                "stable_required_count": int(
                    localization_cfg.get("stable_required_count", 3)
                ),
                "max_translation_jump": float(
                    localization_cfg.get("max_translation_jump", 1.0)
                ),
                "max_rotation_jump_deg": float(
                    localization_cfg.get("max_rotation_jump_deg", 20.0)
                ),
                "transform_smooth_alpha": float(
                    localization_cfg.get("transform_smooth_alpha", 0.35)
                ),
                "gicp.max_correspondence_distance": float(
                    gicp_cfg.get("max_correspondence_distance", 1.5)
                ),
                "gicp.max_iteration": int(
                    gicp_cfg.get("max_iteration", 80)
                ),
                "gicp.euclidean_fitness_epsilon": float(
                    gicp_cfg.get("euclidean_fitness_epsilon", 1e-3)
                ),
                "gicp.rotation_epsilon": float(
                    gicp_cfg.get("rotation_epsilon", 2e-3)
                ),
                "gicp.correspondence_randomness": int(
                    gicp_cfg.get("correspondence_randomness", 20)
                ),
                "gicp.maximum_optimizer_iterations": int(
                    gicp_cfg.get("maximum_optimizer_iterations", 20)
                ),
                "ndt.resolution": float(ndt_cfg.get("resolution", 1.0)),
                "ndt.max_iteration": int(
                    ndt_cfg.get("max_iteration", 60)
                ),
                "ndt.step_size": float(ndt_cfg.get("step_size", 0.1)),
                "ndt.fine.resolution": float(
                    ndt_fine_cfg.get("resolution", 0.2)
                ),
                "ndt.fine.max_iteration": int(
                    ndt_fine_cfg.get("max_iteration", 20)
                ),
                "ndt.fine.step_size": float(
                    ndt_fine_cfg.get("step_size", 0.05)
                ),
                "ndt.outlier_ratio": float(ndt_cfg.get("outlier_ratio", 0.35)),
                "ndt.num_threads": int(ndt_cfg.get("num_threads", 4)),
                "ndt.neighborhood_search": str(
                    ndt_cfg.get("neighborhood_search", "direct7")
                ),
                "source_voxel_leaf_size": float(
                    LaunchConfiguration("localization_source_leaf_size").perform(context)
                ),
                "target_voxel_leaf_size": float(
                    LaunchConfiguration("localization_target_leaf_size").perform(context)
                ),
            }
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    dynamic_cloud = ComposableNode(
        package="dynamic_cloud",
        plugin="tdt_radar::DynamicCloud",
        name="dynamic_cloud",
        parameters=[
            {
                "map_path": map_path,
                "use_sim_time": True,
                "voxel_leaf_size": float(
                    LaunchConfiguration("dynamic_cloud_leaf_size").perform(context)
                ),
                "accumulate_time": int(dynamic_cloud_cfg.get("accumulate_time", 3)),
                "near_accumulate_time": int(
                    dynamic_cloud_cfg.get(
                        "near_accumulate_time",
                        dynamic_cloud_cfg.get("accumulate_time", 3),
                    )
                ),
                "middle_accumulate_time": int(
                    dynamic_cloud_cfg.get(
                        "middle_accumulate_time",
                        dynamic_cloud_cfg.get("accumulate_time", 3),
                    )
                ),
                "far_accumulate_time": int(
                    dynamic_cloud_cfg.get(
                        "far_accumulate_time",
                        dynamic_cloud_cfg.get("accumulate_time", 3),
                    )
                ),
                "region_split_axis": str(
                    dynamic_cloud_cfg.get(
                        "region_split_axis",
                        dynamic_cloud_cfg.get("half_split_axis", "x"),
                    )
                ),
                "near_middle_split_position": float(
                    dynamic_cloud_cfg.get(
                        "near_middle_split_position",
                        dynamic_cloud_cfg.get("half_split_position", 14.0),
                    )
                ),
                "middle_far_split_position": float(
                    dynamic_cloud_cfg.get(
                        "middle_far_split_position",
                        dynamic_cloud_cfg.get("half_split_position", 14.0),
                    )
                ),
                "near_region_positive": bool(
                    dynamic_cloud_cfg.get(
                        "near_region_positive",
                        dynamic_cloud_cfg.get("near_half_positive", False),
                    )
                ),
                "near_map_distance_threshold": float(
                    dynamic_cloud_cfg.get(
                        "near_map_distance_threshold",
                        dynamic_cloud_cfg.get("map_distance_threshold", 0.1),
                    )
                ),
                "middle_map_distance_threshold": float(
                    dynamic_cloud_cfg.get(
                        "middle_map_distance_threshold",
                        dynamic_cloud_cfg.get("map_distance_threshold", 0.1),
                    )
                ),
                "far_map_distance_threshold": float(
                    dynamic_cloud_cfg.get(
                        "far_map_distance_threshold",
                        dynamic_cloud_cfg.get("map_distance_threshold", 0.1),
                    )
                ),
                "dynamic_thread_num": int(
                    dynamic_cloud_cfg.get("dynamic_thread_num", 12)
                ),
                "dart_min_points": int(dynamic_cloud_cfg.get("dart_min_points", 5)),
                "fly_min_points": int(dynamic_cloud_cfg.get("fly_min_points", 40)),
                "main_roi": _cfg_list(
                    dynamic_cloud_cfg,
                    "main_roi",
                    [0.1, 5.0, 0.5, 5.0, -1000.0, 1000.0],
                ),
                "dart_exclude_roi": _cfg_list(
                    dynamic_cloud_cfg,
                    "dart_exclude_roi",
                    [26.5, 28.0, 3.5, 5.5, 0.0, 2.5],
                ),
                "home_exclude_roi": _cfg_list(
                    dynamic_cloud_cfg,
                    "home_exclude_roi",
                    [26.5, 28.0, 13.0, 15.0, 0.0, 3.0],
                ),
                "map_exclude_axis_aligned_rois": _cfg_box_list(
                    dynamic_cloud_cfg, "map_exclude_axis_aligned_rois"
                ),
                "map_exclude_rotated_roi": _cfg_list(
                    dynamic_cloud_cfg,
                    "map_exclude_rotated_roi",
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1000.0, 1000.0],
                ),
                "dart_roi": _cfg_list(
                    dynamic_cloud_cfg,
                    "dart_roi",
                    [27.2226, 27.4111, 3.925, 4.525, 1.7132, 2.4722],
                ),
                "fly_safe_roi": _cfg_list(
                    dynamic_cloud_cfg,
                    "fly_safe_roi",
                    [25.225, 27.5, 0.2, 2.2, 1.7, 3.0],
                ),
                "fly_warn_roi": _cfg_list(
                    dynamic_cloud_cfg,
                    "fly_warn_roi",
                    [19.83, 25.3, 0.2, 4.556, 1.7, 3.0],
                ),
                "fly_alarm_roi": _cfg_list(
                    dynamic_cloud_cfg,
                    "fly_alarm_roi",
                    [13.0, 20.5, 0.2, 4.556, 1.7, 3.0],
                ),
                "fly_detect_roi": _cfg_list(
                    dynamic_cloud_cfg,
                    "fly_detect_roi",
                    [13.0, 25.3, 0.2, 4.556, 1.7, 3.0],
                ),
            }
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    cluster = ComposableNode(
        package="cluster",
        plugin="tdt_radar::Cluster",
        name="cluster",
        parameters=[
            {
                "use_sim_time": True,
                "region_split_axis": str(
                    dynamic_cloud_cfg.get("region_split_axis", "x")
                ),
                "near_middle_split_position": float(
                    dynamic_cloud_cfg.get("near_middle_split_position", 7.0)
                ),
                "middle_far_split_position": float(
                    dynamic_cloud_cfg.get("middle_far_split_position", 14.0)
                ),
                "near_region_positive": bool(
                    dynamic_cloud_cfg.get("near_region_positive", False)
                ),
                "near_cluster_tolerance": float(
                    cluster_cfg.get("near_cluster_tolerance", 0.25)
                ),
                "near_min_cluster_size": int(
                    cluster_cfg.get("near_min_cluster_size", 5)
                ),
                "near_max_cluster_size": int(
                    cluster_cfg.get("near_max_cluster_size", 1000)
                ),
                "middle_cluster_tolerance": float(
                    cluster_cfg.get("middle_cluster_tolerance", 0.25)
                ),
                "middle_min_cluster_size": int(
                    cluster_cfg.get("middle_min_cluster_size", 5)
                ),
                "middle_max_cluster_size": int(
                    cluster_cfg.get("middle_max_cluster_size", 1000)
                ),
                "far_cluster_tolerance": float(
                    cluster_cfg.get("far_cluster_tolerance", 0.25)
                ),
                "far_min_cluster_size": int(
                    cluster_cfg.get("far_min_cluster_size", 5)
                ),
                "far_max_cluster_size": int(
                    cluster_cfg.get("far_max_cluster_size", 1000)
                ),
            }
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    container = ComposableNodeContainer(
        name="rosbag_lidar_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[localization],
        output="both",
        emulate_tty=True,
    )

    load_pipeline = LoadComposableNodes(
        target_container=container,
        composable_node_descriptions=[dynamic_cloud, cluster],
    )

    # 统一回放启动器使用 start_rosbag=false：先把三个计算节点全部拉起，
    # 等 UI/模型就绪后再独立启动 ros2 bag play，避免人工标定期间消耗数据。
    actions = [container]
    if start_rosbag:
        actions.append(rosbag_player)
    actions.append(
        TimerAction(
            period=(
                (playback_delay_sec if start_rosbag else 0.0)
                + float(pipeline_cfg.get("startup_delay_sec", 2.0))
            ),
            actions=[load_pipeline],
        )
    )

    if _is_true(LaunchConfiguration("launch_rviz").perform(context)):
        rviz_cfg = LaunchConfiguration("rviz_config").perform(context).strip()
        if not rviz_cfg:
            rviz_cfg = str(pipeline_cfg.get("rviz_config", "")).strip()
        if rviz_cfg:
            rviz_path = _resolve_path(rviz_cfg, pkg_share)
        else:
            livox_pkg_share = get_package_share_directory("livox_ros_driver2")
            rviz_path = os.path.join(
                livox_pkg_share, "config", "display_point_cloud_ROS2.rviz"
            )
        if not os.path.exists(rviz_path):
            raise RuntimeError(f"rviz config not found: {rviz_path}")
        actions.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_path],
                parameters=[{"use_sim_time": True}],
            )
        )

    return actions


def generate_launch_description():
    pkg_share = get_package_share_directory("rm2026_lidar_launch")
    params = _load_yaml(os.path.join(pkg_share, "config", "params.yaml"))
    test_cfg = params.get("rosbag_lidar_fusion_test", {})
    pipeline_cfg = params.get("lidar_pipeline", {})
    localization_cfg = pipeline_cfg.get("localization", {})
    dynamic_cloud_cfg = pipeline_cfg.get("dynamic_cloud", {})

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(pkg_share, "config", "params.yaml"),
        description="雷达 launch 参数文件；统一回放时显式传入源码目录配置",
    )

    rosbag_file_arg = DeclareLaunchArgument(
        "rosbag_file",
        default_value=str(test_cfg.get("rosbag_file", "rosbag")),
        description="需要播放的rosbag路径（支持绝对路径，若为相对路径则基于rm2026_lidar_launch的share目录）",
    )
    playback_rate_arg = DeclareLaunchArgument(
        "playback_rate",
        default_value=str(test_cfg.get("playback_rate", 1.0)),
        description="rosbag播放倍率，1.0表示按录制速度播放",
    )
    playback_delay_arg = DeclareLaunchArgument(
        "playback_delay_sec",
        default_value=str(test_cfg.get("playback_delay_sec", 1.0)),
        description="开始播放前等待订阅节点启动的时间（秒）",
    )
    start_rosbag_arg = DeclareLaunchArgument(
        "start_rosbag",
        default_value="true",
        description="是否由本 launch 启动 rosbag；统一回放启动器设为 false",
    )
    start_offset_arg = DeclareLaunchArgument(
        "start_offset_sec",
        default_value="0.0",
        description="从 rosbag 起点跳过的秒数",
    )
    loop_playback_arg = DeclareLaunchArgument(
        "loop_playback",
        default_value=str(test_cfg.get("loop_playback", False)).lower(),
        description="rosbag播放结束后是否循环播放",
    )
    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz",
        default_value=str(test_cfg.get("launch_rviz", True)).lower(),
        description="是否自动启动RViz",
    )
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
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=str(test_cfg.get("rviz_config", "")).strip(),
        description="RViz配置文件路径；留空时复用params_file中的lidar_pipeline.rviz_config",
    )

    return LaunchDescription(
        [
            params_file_arg,
            rosbag_file_arg,
            playback_rate_arg,
            playback_delay_arg,
            start_rosbag_arg,
            start_offset_arg,
            loop_playback_arg,
            launch_rviz_arg,
            rviz_config_arg,
            localization_source_leaf_size_arg,
            localization_target_leaf_size_arg,
            dynamic_cloud_leaf_size_arg,
            OpaqueFunction(function=_build_actions),
        ]
    )
