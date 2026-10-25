#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一启动 ROS2 雷达 rosbag 与 Python 相机时间戳回放。"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = WORKSPACE_ROOT / "Main_ws"
ROS_WORKSPACE = WORKSPACE_ROOT / "Lidar_ws"
LIVOX_WORKSPACE = Path(
    os.environ.get("LIVOX_WS_DIR", WORKSPACE_ROOT / "../livox_ws")
).resolve()
ROS_LAUNCH = (
    ROS_WORKSPACE
    / "src/rm2026_lidar_launch/launch/lidar_pipeline.launch.py"
)
ROS_PARAMS = ROS_WORKSPACE / "src/rm2026_lidar_launch/config/params.yaml"


class ReplayConfigurationError(RuntimeError):
    pass


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _resolve_workspace_path(value: str, field_name: str) -> Path:
    if not str(value).strip():
        raise ReplayConfigurationError(f"test.yaml 未填写 {field_name}")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def _resolve_camera_dir(value: str) -> Path:
    path = _resolve_workspace_path(value, "replay.camera_session_dir")
    candidates = (path, path / "camera")
    for candidate in candidates:
        if (
            (candidate / "camera_record.mp4").is_file()
            and (candidate / "camera_timestamps.csv").is_file()
        ):
            return candidate
    raise ReplayConfigurationError(
        "相机目录必须包含 camera_record.mp4 和 camera_timestamps.csv，"
        f"当前路径: {path}"
    )


def _resolve_bag_dir(value: str) -> tuple[Path, Path]:
    path = _resolve_workspace_path(value, "replay.rosbag_path")
    if path.is_file() and path.suffix == ".db3":
        path = path.parent
    candidates = (path, path / "rosbag_livox_lidar")
    for candidate in candidates:
        metadata = candidate / "metadata.yaml"
        if metadata.is_file():
            return candidate, metadata
    raise ReplayConfigurationError(
        f"未在 {path} 或其 rosbag_livox_lidar 子目录找到 metadata.yaml"
    )


def _read_bag_range(metadata_path: Path) -> tuple[float, float]:
    metadata = _load_yaml(metadata_path).get("rosbag2_bagfile_information", {})
    try:
        start_ns = int(metadata["starting_time"]["nanoseconds_since_epoch"])
        duration_ns = int(metadata["duration"]["nanoseconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayConfigurationError(
            f"rosbag metadata 缺少有效起止时间: {metadata_path}"
        ) from error
    if start_ns <= 0 or start_ns >= 2**63 - 1 or duration_ns <= 0:
        raise ReplayConfigurationError(
            f"rosbag 没有有效数据或尚未正常关闭: {metadata_path}"
        )
    start = start_ns / 1e9
    return start, start + duration_ns / 1e9


def _read_camera_range(csv_path: Path, time_basis: str) -> tuple[float, float, int]:
    column = "camera_stamp" if time_basis == "sensor_stamp" else "wall_time_s"
    first = None
    last = None
    count = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        if "frame_index" not in fields or column not in fields:
            raise ReplayConfigurationError(
                f"相机 CSV 缺少 frame_index/{column}: {csv_path}"
            )
        for row in reader:
            raw = str(row.get(column, "")).strip()
            if not raw and column == "camera_stamp":
                raw = str(row.get("wall_time_s", "")).strip()
            if not raw:
                continue
            stamp = float(raw)
            if last is not None and stamp < last:
                raise ReplayConfigurationError(
                    f"相机时间戳不是单调递增: {last:.9f} -> {stamp:.9f}"
                )
            first = stamp if first is None else first
            last = stamp
            count += 1
    if first is None or last is None or count == 0:
        raise ReplayConfigurationError(f"相机 CSV 中没有有效时间戳: {csv_path}")
    return first, last, count


def _merge_recorded_calibration(
    params: dict, camera_dir: Path, run_dir: Path
) -> tuple[str, dict]:
    """校验录像标定快照，并将录像时的内外参写入回放参数。"""
    snapshot_path = camera_dir / "calibration_snapshot.yaml"
    if not snapshot_path.is_file():
        raise ReplayConfigurationError(
            f"replay 相机目录缺少 calibration_snapshot.yaml: {snapshot_path}"
        )

    snapshot = _load_yaml(snapshot_path)
    intrinsics = snapshot.get("intrinsics") or {}
    camera_matrix = intrinsics.get("K")
    dist_coeffs = intrinsics.get("dist_coeffs")
    if (
        not isinstance(camera_matrix, list)
        or len(camera_matrix) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in camera_matrix)
    ):
        raise ReplayConfigurationError(
            "calibration_snapshot.yaml::intrinsics.K 必须为 3x3 矩阵: "
            f"{snapshot_path}"
        )
    if not isinstance(dist_coeffs, list) or not dist_coeffs:
        raise ReplayConfigurationError(
            "calibration_snapshot.yaml::intrinsics.dist_coeffs 不能为空: "
            f"{snapshot_path}"
        )

    extrinsics = snapshot.get("extrinsics") or {}
    rotation = extrinsics.get("rotation_matrix")
    translation = extrinsics.get("translation_vector")
    if rotation is None or translation is None:
        raise ReplayConfigurationError(
            "calibration_snapshot.yaml 缺少完整 rotation_matrix/"
            f"translation_vector: {snapshot_path}"
        )

    transform = params.get("transform")
    if not isinstance(transform, dict):
        transform = {}
        params["transform"] = transform
    transform["K"] = copy.deepcopy(camera_matrix)
    transform["dist_coeffs"] = copy.deepcopy(dist_coeffs)
    transform["R"] = copy.deepcopy(rotation)
    transform["t"] = copy.deepcopy(translation)
    recorded_extrinsics_path = camera_dir / "extrinsics.txt"
    if recorded_extrinsics_path.is_file():
        import_path = recorded_extrinsics_path
    else:
        import_path = run_dir / "recorded_extrinsics.txt"
        rows = [
            "timestamp: replay_snapshot",
            "source: recording_snapshot",
            "",
            "rotation_matrix:",
        ]
        rows.extend(" ".join(str(float(value)) for value in row) for row in rotation)
        rows.extend(("", "translation_vector:"))
        translation_flat = [
            float(value)
            for row in translation
            for value in (row if isinstance(row, list) else [row])
        ]
        rows.extend(str(value) for value in translation_flat)
        rows.extend(("", "extrinsic_3x4:"))
        for row, value in zip(rotation, translation_flat):
            rows.append(" ".join(str(float(item)) for item in row) + f" {value}")
        rows.append("")
        import_path.write_text("\n".join(rows), encoding="utf-8")

    snapshot_info = {
        "path": str(snapshot_path),
        "image_width": intrinsics.get("image_width"),
        "image_height": intrinsics.get("image_height"),
    }
    return str(import_path), snapshot_info


def _prepare_runtime_files(config_path: Path) -> dict:
    original = _load_yaml(config_path)
    replay = original.get("replay", {})
    if not bool(replay.get("enabled", True)):
        raise ReplayConfigurationError("replay.enabled=false，拒绝启动测试回放")

    time_basis = str(replay.get("time_basis", "sensor_stamp")).lower()
    if time_basis not in ("sensor_stamp", "arrival_stamp"):
        raise ReplayConfigurationError(
            "replay.time_basis 只能是 sensor_stamp 或 arrival_stamp"
        )
    playback_rate = float(replay.get("playback_rate", 1.0))
    if playback_rate <= 0.0:
        raise ReplayConfigurationError("replay.playback_rate 必须大于 0")
    camera_dir = _resolve_camera_dir(replay.get("camera_session_dir", ""))
    bag_dir, metadata_path = _resolve_bag_dir(replay.get("rosbag_path", ""))
    camera_csv = camera_dir / "camera_timestamps.csv"
    camera_video = camera_dir / "camera_record.mp4"
    camera_start, camera_end, camera_count = _read_camera_range(
        camera_csv, time_basis
    )
    bag_start, bag_end = _read_bag_range(metadata_path)
    valid_start = max(camera_start, bag_start)
    valid_end = min(camera_end, bag_end)
    if valid_start >= valid_end:
        raise ReplayConfigurationError(
            "相机与 rosbag 时间范围没有交集:\n"
            f"  camera={camera_start:.6f}~{camera_end:.6f}\n"
            f"  rosbag={bag_start:.6f}~{bag_end:.6f}"
        )

    # 相机与雷达从首个共同有效时间戳同时开始，不再让雷达预播放。
    replay_start = valid_start
    start_offset = max(0.0, replay_start - bag_start)
    actual_preroll = 0.0

    output_root_text = str(replay.get("output_root", "")).strip()
    output_root = (
        _resolve_workspace_path(output_root_text, "replay.output_root")
        if output_root_text
        else ALGORITHM_ROOT / "logs/replay"
    )
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    runtime_params_source = _resolve_workspace_path(
        original.get("runtime", {}).get(
            "params_config", "Main_ws/config/params.yaml"
        ),
        "runtime.params_config",
    )
    runtime_params = _load_yaml(runtime_params_source)
    debug = runtime_params.setdefault("debug", {})
    debug["camera_mode"] = "replay"
    debug["camera_color_order"] = "rgb"
    debug["inference_video"] = False
    debug["video_source"] = str(camera_video)
    metrics = debug.setdefault("performance_metrics", {})
    if metrics.get("enabled", False):
        metrics["output_path"] = str(run_dir / "runtime_metrics.jsonl")

    extrinsics_path, calibration_snapshot = _merge_recorded_calibration(
        runtime_params, camera_dir, run_dir
    )
    runtime_params_path = run_dir / "runtime_params.yaml"
    with runtime_params_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(runtime_params, file, allow_unicode=True, sort_keys=False)

    ready_file = run_dir / "tracking.ready"
    resolved = copy.deepcopy(original)
    resolved["_resolved"] = {
        "workspace_root": str(WORKSPACE_ROOT),
        "run_dir": str(run_dir),
        "camera_dir": str(camera_dir),
        "camera_video": str(camera_video),
        "camera_timestamps": str(camera_csv),
        "rosbag_dir": str(bag_dir),
        "bag_start_timestamp": bag_start,
        "bag_end_timestamp": bag_end,
        "camera_start_timestamp": camera_start,
        "camera_end_timestamp": camera_end,
        "valid_start_timestamp": valid_start,
        "valid_end_timestamp": valid_end,
        "replay_start_timestamp": replay_start,
        "rosbag_start_offset_sec": start_offset,
        "actual_lidar_preroll_sec": actual_preroll,
        "tracking_ready_file": str(ready_file),
        "runtime_params": str(runtime_params_path),
        "extrinsics_import_path": extrinsics_path,
        "calibration_snapshot": calibration_snapshot["path"],
    }
    resolved_test_path = run_dir / "resolved_test.yaml"
    with resolved_test_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(resolved, file, allow_unicode=True, sort_keys=False)

    return {
        "config": resolved,
        "run_dir": run_dir,
        "runtime_params": runtime_params_path,
        "resolved_test": resolved_test_path,
        "ready_file": ready_file,
        "bag_dir": bag_dir,
        "start_offset": start_offset,
        "playback_rate": playback_rate,
        "playback_delay": max(0.0, float(replay.get("discovery_delay_sec", 0.5))),
        "play_duration_wall": (valid_end - replay_start) / playback_rate,
        "camera_count": camera_count,
        "intrinsics_source": calibration_snapshot["path"],
        "calibration_snapshot": calibration_snapshot,
        "extrinsics_path": extrinsics_path,
        "actual_preroll": actual_preroll,
        "valid_start": valid_start,
        "valid_end": valid_end,
    }


def _open_log(path: Path):
    return path.open("w", encoding="utf-8", buffering=1)


def _start_process(command: list[str], log_file) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _terminate_process(process: subprocess.Popen | None, name: str, timeout: float) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"[同步回放] 正在停止{name}...")
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=timeout)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3.0)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _ros_environment_script(command: str) -> list[str]:
    script = f"""
set -e
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
# 海康 MVS 安装目录中的旧版 libusb 会被系统动态链接器优先选中，
# 而 PCL 需要含 libusb_set_option 的系统版本。ROS 回放进程不使用 MVS，显式预加载系统库。
if [ -f /lib/x86_64-linux-gnu/libusb-1.0.so.0 ]; then
    export LD_PRELOAD=/lib/x86_64-linux-gnu/libusb-1.0.so.0
fi
source /opt/ros/jazzy/setup.bash
if [ -f {LIVOX_WORKSPACE}/install/setup.bash ]; then
    source {LIVOX_WORKSPACE}/install/setup.bash
fi
source {ROS_WORKSPACE}/install/setup.bash
{command}
"""
    return ["bash", "--noprofile", "--norc", "-c", script]


def _wait_ros_nodes(process: subprocess.Popen, timeout: float) -> None:
    required = {
        "localization": {"localization", "localization_node"},
        "dynamic_cloud": {"dynamic_cloud", "dynamic_cloud_node"},
        "cluster": {"cluster", "cluster_node"},
    }
    deadline = time.monotonic() + timeout
    query = _ros_environment_script("ros2 node list")
    last_output = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ROS 雷达管线提前退出，返回码 {process.returncode}")
        try:
            result = subprocess.run(
                query,
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            last_output = result.stdout
            names = {line.rsplit("/", 1)[-1] for line in result.stdout.splitlines()}
            if all(aliases & names for aliases in required.values()):
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(0.5)
    raise RuntimeError(
        f"等待 ROS 节点 {tuple(required)} 超时。最后一次 ros2 node list:\n{last_output}"
    )


def _wait_file(path: Path, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process.poll() is not None:
            raise RuntimeError(f"Python 主程序提前退出，返回码 {process.returncode}")
        time.sleep(0.2)
    raise RuntimeError(f"等待 UI 点击“启动雷达站”超时: {path}")


def _print_preflight(runtime: dict) -> None:
    resolved = runtime["config"]["_resolved"]
    print("[同步回放] 预检通过")
    print(f"  rosbag: {resolved['rosbag_dir']}")
    print(f"  camera: {resolved['camera_dir']}")
    print(
        "  有效交集: "
        f"{runtime['valid_start']:.6f} ~ {runtime['valid_end']:.6f} "
        f"({runtime['valid_end'] - runtime['valid_start']:.3f}s)"
    )
    print(
        f"  同步 rosbag offset: {runtime['start_offset']:.3f}s，"
        "相机/雷达从共同起点同时播放"
    )
    print(f"  相机 CSV 帧数: {runtime['camera_count']}")
    print(
        f"  运行内参来源: {runtime['intrinsics_source']}"
        "::intrinsics.K/dist_coeffs"
    )
    snapshot = runtime["calibration_snapshot"]
    print(f"  录像标定快照: {snapshot['path']}")
    print(
        "  快照图像尺寸（仅供人工核验）: "
        f"{snapshot.get('image_width') or '未记录'} × "
        f"{snapshot.get('image_height') or '未记录'}"
    )
    print(f"  外参导入初值: {runtime['extrinsics_path']}")
    print(f"  本次运行目录: {runtime['run_dir']}")


def run(config_path: Path, preflight_only: bool) -> int:
    runtime = _prepare_runtime_files(config_path)
    _print_preflight(runtime)
    if preflight_only:
        return 0

    ros_setup = ROS_WORKSPACE / "install/setup.bash"
    if not ros_setup.is_file():
        raise ReplayConfigurationError(
            "Lidar_ws 尚未构建，缺少 install/setup.bash。请先执行：\n"
            "  source /opt/ros/jazzy/setup.bash\n"
            "  cd Lidar_ws && ./build.sh"
        )

    config = runtime["config"]
    runtime_cfg = config.get("runtime", {})
    ros_timeout = max(1.0, float(runtime_cfg.get("ros_ready_timeout_sec", 30.0)))
    tracking_timeout = max(
        1.0, float(runtime_cfg.get("tracking_ready_timeout_sec", 600.0))
    )
    shutdown_timeout = max(
        1.0, float(runtime_cfg.get("shutdown_timeout_sec", 10.0))
    )
    keep_ui = bool(runtime_cfg.get("keep_ui_after_replay", True))
    launch_rviz = str(bool(runtime_cfg.get("launch_rviz", False))).lower()

    ros_log = _open_log(runtime["run_dir"] / "ros_pipeline.log")
    python_log = _open_log(runtime["run_dir"] / "python_app.log")
    bag_log = _open_log(runtime["run_dir"] / "rosbag_play.log")
    ros_process = None
    python_process = None
    bag_process = None
    try:
        ros_params = _load_yaml(ROS_PARAMS)
        ros_pipeline_cfg = ros_params.get("lidar_pipeline", {})
        localization_cfg = ros_pipeline_cfg.get("localization", {})
        localization_source_leaf_size = float(
            localization_cfg.get("source_voxel_leaf_size", 0.1)
        )
        localization_target_leaf_size = float(
            localization_cfg.get("target_voxel_leaf_size", 0.1)
        )
        dynamic_cloud_leaf_size = float(
            ros_pipeline_cfg.get("dynamic_cloud", {}).get("voxel_leaf_size", 0.1)
        )
        # 同步回放直接复用正式 lidar_pipeline 的 RViz 配置，避免测试界面
        # 与比赛时订阅的话题、Fixed Frame 和显示样式不一致。
        rviz_config = str(ros_pipeline_cfg.get("rviz_config", "")).strip()
        rviz_config_arg = (
            f" rviz_config:={shlex.quote(rviz_config)}" if rviz_config else ""
        )
        ros_command = _ros_environment_script(
            "exec ros2 launch "
            f"{shlex.quote(str(ROS_LAUNCH))} "
            f"params_file:={shlex.quote(str(ROS_PARAMS))} "
            "start_livox:=false use_sim_time:=true "
            f"localization_source_leaf_size:={localization_source_leaf_size} "
            f"localization_target_leaf_size:={localization_target_leaf_size} "
            f"dynamic_cloud_leaf_size:={dynamic_cloud_leaf_size} "
            f"launch_rviz:={launch_rviz}"
            f"{rviz_config_arg}"
        )
        ros_process = _start_process(ros_command, ros_log)
        print("[同步回放] 正在等待 localization/dynamic_cloud/cluster 就绪...")
        _wait_ros_nodes(ros_process, ros_timeout)
        print("[同步回放] ROS 雷达管线已就绪。")

        device_config = _resolve_workspace_path(
            runtime_cfg.get(
                "device_config", "Main_ws/config/device.yaml"
            ),
            "runtime.device_config",
        )
        python_script = f"""
set -e
source /opt/ros/jazzy/setup.bash
source {ROS_WORKSPACE}/install/setup.bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {shlex.quote(os.environ.get('CONDA_ENV', 'cud128'))}
cd {shlex.quote(str(ALGORITHM_ROOT))}
export RM_EXTRINSICS_IMPORT_PATH={shlex.quote(str(runtime['extrinsics_path']))}
exec bash run_scripts/run_test.sh {shlex.quote(str(runtime['runtime_params']))} {shlex.quote(str(device_config))} {shlex.quote(str(runtime['resolved_test']))}
"""
        python_process = _start_process(
            ["bash", "--noprofile", "--norc", "-c", python_script],
            python_log,
        )
        print(
            "[同步回放] Python/UI 已启动。请在 UI 中导入录像外参（或手动标定），"
            "然后点击“启动雷达站”。"
        )
        _wait_file(runtime["ready_file"], python_process, tracking_timeout)
        print("[同步回放] 主循环已就绪，现在开始 rosbag 和相机同步回放。")

        bag_command = _ros_environment_script(
            "exec ros2 bag play "
            f"{shlex.quote(str(runtime['bag_dir']))} "
            f"--rate {runtime['playback_rate']} "
            f"--delay {runtime['playback_delay']} "
            f"--start-offset {runtime['start_offset']:.9f} "
            "--clock 100.0 --disable-keyboard-controls"
        )
        bag_process = _start_process(bag_command, bag_log)
        deadline = (
            time.monotonic()
            + runtime["playback_delay"]
            + runtime["play_duration_wall"]
            + 1.0
        )
        while bag_process.poll() is None:
            if python_process.poll() is not None:
                raise RuntimeError(
                    f"Python 主程序在回放中退出，返回码 {python_process.returncode}"
                )
            if ros_process.poll() is not None:
                raise RuntimeError(
                    f"ROS 雷达管线在回放中退出，返回码 {ros_process.returncode}"
                )
            if time.monotonic() >= deadline:
                _terminate_process(bag_process, "rosbag", shutdown_timeout)
                break
            time.sleep(0.25)

        print("[同步回放] 有效时间交集已经播放完毕。")
        if keep_ui:
            print("[同步回放] UI 将保留供检查；关闭 UI 或按 Ctrl+C 后统一退出。")
            while python_process.poll() is None:
                if ros_process.poll() is not None:
                    raise RuntimeError(
                        f"ROS 雷达管线提前退出，返回码 {ros_process.returncode}"
                    )
                time.sleep(0.5)
        return 0
    finally:
        _terminate_process(bag_process, "rosbag", shutdown_timeout)
        _terminate_process(python_process, "Python/UI", shutdown_timeout)
        _terminate_process(ros_process, "ROS 雷达管线", shutdown_timeout)
        ros_log.close()
        python_log.close()
        bag_log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="相机/雷达时间戳无偏差同步回放")
    parser.add_argument("config", type=Path, help="test.yaml 路径")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查路径和时间交集，不启动 ROS/UI",
    )
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        print(f"错误: test.yaml 不存在: {config_path}", file=sys.stderr)
        return 2
    try:
        return run(config_path, args.preflight_only)
    except KeyboardInterrupt:
        print("\n[同步回放] 收到 Ctrl+C，正在统一退出。")
        return 130
    except (ReplayConfigurationError, RuntimeError, OSError, ValueError) as error:
        print(f"[同步回放] 启动失败: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
