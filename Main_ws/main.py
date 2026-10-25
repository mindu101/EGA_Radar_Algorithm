# -*- coding: utf-8 -*-
"""
Main entry point for RM2026 Radar Algorithm System
Copyright (c) 2025 香港科技大学ENTERPRIZE战队（HKUST ENTERPRIZE Team）

Licensed under the MIT License. See LICENSE file in the project root for license information.
"""

import os

from driver.referee.referee_comm import RefereeCommManager
from utils.config import (
    load_cfg_from_cfg_file,
    merge_cfg_from_list,
)
from interface.core import launch
import argparse
import rclpy

from main_event_loop import MainEventLoop
from utils.runtime_config import PARAMS_CONFIG_ENV, TEST_CONFIG_ENV


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pytorch Referring Expression Segmentation"
    )
    parser.add_argument("--gpu", default="0,1")
    parser.add_argument(
        "--config", default="path to xxx.yaml", type=str, help="config file"
    )
    parser.add_argument(
        "--device_config",
        default="path to device.yaml",
        type=str,
        help="camera and lidar setting",
    )
    parser.add_argument(
        "--test_config",
        default=None,
        type=str,
        help="同步回放 test.yaml；不提供时保持正式比赛逻辑",
    )
    parser.add_argument(
        "--disable-referee-io",
        action="store_true",
        help="不启动裁判串口与通信线程（无裁判系统的现场测试）",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="override some settings in the config.",
    )

    cli_args = parser.parse_args()
    assert cli_args.config is not None
    cfg = load_cfg_from_cfg_file(cli_args.device_config)
    if cli_args.opts is not None:
        cfg = merge_cfg_from_list(cfg, cli_args.opts)
    return cli_args, cfg

def load_yaml_config(path: str):
    import yaml
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def shutdown_runtime(camera, referee, event_loop):
    """统一释放运行时资源，确保窗口关闭或 Ctrl+C 后进程可退出。"""
    # 先停主事件循环，避免其继续访问已关闭的相机/串口
    try:
        if event_loop is not None:
            event_loop.stop()
    except Exception as e:
        print(f"[shutdown] stop event_loop failed: {e}")

    try:
        if referee is not None:
            referee.close()
    except Exception as e:
        print(f"[shutdown] close referee failed: {e}")

    try:
        if camera is not None and hasattr(camera, "stop_streaming"):
            camera.stop_streaming()
    except Exception as e:
        print(f"[shutdown] stop camera streaming failed: {e}")

    try:
        if camera is not None and hasattr(camera, "stop_saving_images"):
            camera.stop_saving_images()
    except Exception as e:
        print(f"[shutdown] stop camera saving failed: {e}")

    try:
        if camera is not None and hasattr(camera, "close"):
            camera.close()
    except Exception as e:
        print(f"[shutdown] close camera failed: {e}")

    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception as e:
        print(f"[shutdown] rclpy shutdown failed: {e}")


if __name__ == "__main__":
    camera = None
    referee = None
    event_loop = None
    cleanup_state = {"done": False}

    def safe_shutdown():
        if cleanup_state["done"]:
            return
        cleanup_state["done"] = True
        shutdown_runtime(camera, referee, event_loop)

    cli_args, args = get_parser()
    params_config_path = os.path.abspath(cli_args.config)
    os.environ[PARAMS_CONFIG_ENV] = params_config_path
    if cli_args.test_config:
        os.environ[TEST_CONFIG_ENV] = os.path.abspath(cli_args.test_config)
    else:
        os.environ.pop(TEST_CONFIG_ENV, None)

    yaml_config = load_yaml_config(params_config_path)
    test_config = (
        load_yaml_config(os.environ[TEST_CONFIG_ENV])
        if cli_args.test_config
        else {}
    )
    rclpy.init()

    # 实例化相机
    debug_cfg = yaml_config.get("debug", {})
    camera_mode = debug_cfg.get("camera_mode")
    if not camera_mode:
        camera_mode = (
            "image" if debug_cfg.get("inference_video", False) else "camera"
        )
    camera_mode = camera_mode.lower()
    if camera_mode == "replay":
        if not cli_args.test_config:
            raise ValueError("camera_mode=replay 时必须提供 --test_config")
        from driver.replay_camera import TimestampReplayCamera

        camera = TimestampReplayCamera(os.environ[TEST_CONFIG_ENV])
    elif camera_mode == "ros":
        from driver.ros_camera.ros_image_camera import RosImageCamera

        camera = RosImageCamera(
            topic=debug_cfg.get("ros_image_topic", "/camera_image"),
            queue_size=int(debug_cfg.get("ros_image_queue_size", 5)),
        )
    elif camera_mode == "image":
        from driver.hik_camera.mock_hik import SimpleHikCamera

        camera = SimpleHikCamera(
            video_source=debug_cfg.get("video_source", "")
        )
    elif camera_mode == "camera":
        from driver.hik_camera.hik import SimpleHikCamera

        # with open("test.txt", "w") as f:
        #     f.write(f"args: {args}\n")

        camera = SimpleHikCamera(args)
    else:
        raise ValueError(f"Unknown camera_mode: {camera_mode}")

    referee_cfg = yaml_config["referee"]
    # 正式启动允许用环境变量明确指定裁判串口；未指定时仍使用配置文件，
    # 如果该端口不存在，RefereeCommManager 会自动扫描非 Pluto 的 USB 串口。
    referee_port = os.environ.get("REFEREE_PORT", referee_cfg["port"]).strip()
    referee_baudrate = int(referee_cfg["baudrate"])
    referee = RefereeCommManager(port=referee_port, baudrate=referee_baudrate)
    runtime_test_cfg = test_config.get("runtime", {})
    forced_faction = str(runtime_test_cfg.get("faction", "manual")).lower()
    if forced_faction in ("red", "blue"):
        from driver.referee.referee_comm import FACTION

        referee.faction = FACTION.RED if forced_faction == "red" else FACTION.BLUE
    if not (cli_args.disable_referee_io or bool(runtime_test_cfg.get("disable_referee_io", False))):
        referee.start() #启动裁判系统通讯
    else:
        print("已禁用裁判串口和通信线程。")
    # print("RefereeCommManager started successfully.")

    event_loop = MainEventLoop( #实例化主事件循环
        camera = camera, referee=referee
    )
    camera.start_streaming()  #启动相机图像采集
    
    
    # if camera_mode == "camera" and debug_cfg.get("streaming_video", False):
    # camera.start_saving_threads()  # 启动图像保存线程
    
    
    try:
        # 启动 Qt 主界面事件循环（支持 Ctrl+C 退出）
        launch()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping...")
    finally:
        safe_shutdown()
