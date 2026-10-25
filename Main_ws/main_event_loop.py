# -*- coding: utf-8 -*-
"""
Main Event Loop for RM2026 Radar Algorithm System
Copyright (c) 2025 香港科技大学ENTERPRIZE战队（HKUST ENTERPRIZE Team）

Licensed under the MIT License. See LICENSE file in the project root for license information.
"""
from driver.referee.serial_protocol import *
from driver.referee.referee_comm import RefereeCommManager, FACTION
import yaml
from transform.ray_renderer import PixelToWorld
from transform.solidwork2uwb import solidwork2uwb
from transform.xy_map import XYMapManager
from transform.tf_debug_publisher import CameraTfDebugPublisher
from threading import Thread, Event, Lock, RLock
import logging
import math
import re
import time
from dataclasses import dataclass
from types import SimpleNamespace
from utils.runtime_metrics import RuntimeMetricsExporter
from utils.async_visualization import AsyncTrackingVisualizer
from utils.runtime_config import load_params_config, load_test_config
from tracker.CascadeMatchTracker.pipeline import (
    CompletedFrame,
    DivisionPositionSnapshot,
    LatestPacketMailbox,
    PipelineDiagnostics,
    StageAPacket,
    TrackOutputSnapshot,
    freeze_mapping,
)
########################################################################
import cv2
import numpy as np
########################################################################


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model", "pfa_yolov5"))


@dataclass
class RefereeDivision2dPosition:
    x: int
    y: int
    is_valid: bool
    source: str = "invalid"
    is_guessed: bool = False


class MainEventLoop:
    
    _instance = None

    def __init__(
        self, camera, referee: RefereeCommManager, lidar_client=None, lidar_fly_client=None
    ):
        # print(f"In initializer: {calibrated_R}, {calibrated_T}")
        self.config = load_params_config()
        self.test_config = load_test_config()
        self.camera = camera
        self.camera.register_group("tracker")
        self.referee = referee
        self.lidar_config = self.config.get("lidar_fusion", {})
        self.tf_debug_cfg = self.config.get("debug", {}).get("tf_debug", {})
        self.tf_debug_publisher = CameraTfDebugPublisher(self.tf_debug_cfg)
        self.lidar_client = lidar_client
        self.lidar_fly_client = lidar_fly_client
        self.lidar_points_cm = []
        self.lidar_point_owners = {}
        self.lidar_fly_points_cm = []
        self.projection_points_cm = []
        if self.lidar_client is None and self.lidar_config.get("enabled", False):
            import rclpy
            from driver.lidar.ros2_lidar_client import LidarClient

            if not rclpy.ok():
                rclpy.init()
            self.lidar_client = LidarClient(
                topic=self.lidar_config.get("topic", "/livox/lidar_cluster"),
                queue_size=int(self.lidar_config.get("queue_size", 10)),
            )
            self.lidar_client.start()

        fly_enabled = self.lidar_config.get(
            "fly_enabled", self.lidar_config.get("enabled", False)
        )
        if self.lidar_fly_client is None and fly_enabled:
            import rclpy
            from driver.lidar.ros2_lidar_client import LidarClient

            if not rclpy.ok():
                rclpy.init()
                
            self.lidar_fly_client = LidarClient(
                topic=self.lidar_config.get("fly_topic", "/livox/lidar_fly_cluster"),
                queue_size=int(self.lidar_config.get("queue_size", 10)),
                name="rm2026_lidar_fly_client"
            )
            self.lidar_fly_client.start()

        ## Display status
        self.inference_fps = 0
        self.localization_fps = 0.0
        self.last_localization_time = None
        self.debug_cfg = self.config.get("debug", {})
        self.timing_debug_enabled = bool(self.debug_cfg.get("timing_debug_enabled", True))
        self.timing_alpha = float(self.debug_cfg.get("timing_ema_alpha", 0.2))
        self.log_inference_fps = bool(self.debug_cfg.get("log_inference_fps", False))
        ########################################################################
        # 前哨站颜色二值化和 UI 显示使用同一份相机通道顺序配置，避免红蓝通道互换。
        self.camera_color_order = str(self.debug_cfg.get("camera_color_order", "rgb")).lower()
        if self.camera_color_order not in ("rgb", "bgr"):
            logger.warning(
                "Unknown camera_color_order: %s, defaulting to rgb",
                self.camera_color_order,
            )
            self.camera_color_order = "rgb"
        ########################################################################
        self.loop_timing_stats = {}
        self.loop_frame_timing_ms = {}
        self._timing_stats_lock = RLock()
        self.pipeline_config = self._load_detection_pipeline_config()
        self.pipeline_enabled = bool(self.pipeline_config["enabled"])
        self._pipeline_state_lock = RLock()
        self._pipeline_output_lock = Lock()
        self._latest_pipeline_output = None
        self._pipeline_epoch = 1
        self._pipeline_last_frame_size = None
        self._pipeline_last_frame_stamp = None
        self.pipeline_diagnostics = PipelineDiagnostics()
        self._pipeline_mailbox = (
            LatestPacketMailbox(self.pipeline_diagnostics)
            if self.pipeline_enabled
            else None
        )
        self._pipeline_worker_thread = None
        self._commit_interval_ema_ms = None
        self.imgsize = self.config["car_detector"]["img_size"]
        self.latest_detection_results = []
        self.latest_detection_source_frame = None
        self.latest_detection_source_stamp = None
        self.detect_vis_img = None
        self.track_vis_img = None
        self.track_vis_source_stamp = None
        self.track_vis_sequence = 0
        self.visualization_worker = None
        self.processed_frame_count = 0
        self.metrics_exporter = RuntimeMetricsExporter(
            self.debug_cfg.get("performance_metrics", {})
        )
        self.xy_map_manager = XYMapManager(
            self.config,
            metrics_callback=self._export_xy_map_metrics,
        )
        # 仅把相机接口报告的真实尺寸作为预热值；检测开始后还会逐帧使用
        # img.shape 核对。XY Map 不再从 params.yaml 猜测图像宽高。
        initial_width = int(getattr(self.camera, "width", 0) or 0)
        initial_height = int(getattr(self.camera, "height", 0) or 0)
        if initial_width > 0 and initial_height > 0:
            self.xy_map_manager.set_frame_size(initial_width, initial_height)

        # ROI 格式为原图坐标 (x1, y1, x2, y2)。
        self.outpost_roi = None
        self.outpost_alive = 1
        ########################################################################
        self.outpost_alive_ratio = None
        ########################################################################
        ########################################################################
        # 前哨站二值化调试图与阈值配置。
        self.outpost_binary_image = None
        self.outpost_light_config = self._load_outpost_light_config(
            self.config.get("outpost_light", {})
        )
        ########################################################################

        os.environ["OPENBLAS_FPE"] = "0"

        # 0 - 4 for blue, 5 - 9 for red
        self.divisions_pos = [
            RefereeDivision2dPosition(-1, -1, False) for _ in range(10)
        ]
        
        # Load weight to the cuda during the initialization
        self.dummy_pixel_world_transform = PixelToWorld.build_from_config(
            self.config
        )
        if self.config["tracker"]["type"] == "CascadeMatchTracker":
            from tracker.CascadeMatchTracker.tracker import CascadeMatchTracker
            self.tracker = CascadeMatchTracker(
                self.config,
                pixel_world_transform=self.dummy_pixel_world_transform,
                visualize=True,
                xy_map_manager=self.xy_map_manager,
            )
        # elif self.config["tracker"]["type"] == "BotIdMatchTracker":
        #     from tracker.CascadeMatchTracker.botidtracker import BotIdMatchTracker
        #     self.tracker = BotIdMatchTracker(
        #         self.config, pixel_world_transform=self.dummy_pixel_world_transform, visualize=True
        #     )
        else:
            raise ValueError(f"Unknown tracker type: {self.config['tracker']['type']}")
        
        # warm up
        self.tracker.warmup(warmup_num=10)

        # 叠图只服务 UI，后台线程按最新帧覆盖运行，不能阻塞定位主流程。
        self.visualization_worker = AsyncTrackingVisualizer(
            self.config,
            on_image_ready=self._publish_track_visualization,
        )
        
        self.__class__._instance = self

    def _load_detection_pipeline_config(self) -> dict:
        """读取并严格校验流水线配置；旧配置缺少该节点时保持串行。"""
        defaults = {
            "enabled": False,
            "camera_timeout_sec": 0.10,
            "max_commit_age_sec": 0.20,
            "worker_join_timeout_sec": 5.0,
        }
        raw = self.config.get("detection_pipeline")
        if raw is None:
            logger.warning("配置缺少 detection_pipeline，按兼容策略使用串行模式。")
            return defaults
        if not isinstance(raw, dict):
            raise ValueError("detection_pipeline 必须是 YAML 映射")
        unknown = set(raw) - set(defaults)
        if unknown:
            raise ValueError(
                f"detection_pipeline 存在未知参数: {sorted(unknown)}"
            )
        cfg = {**defaults, **raw}
        if not isinstance(cfg["enabled"], bool):
            raise ValueError("detection_pipeline.enabled 必须是 bool")
        for key in (
            "camera_timeout_sec",
            "max_commit_age_sec",
            "worker_join_timeout_sec",
        ):
            try:
                cfg[key] = float(cfg[key])
            except (TypeError, ValueError) as error:
                raise ValueError(f"detection_pipeline.{key} 必须是数值") from error
            if cfg[key] <= 0.0:
                raise ValueError(f"detection_pipeline.{key} 必须大于 0")
        return cfg

    def get_latest_pipeline_output(self):
        """原子读取最近一次完整提交，供 UI 一次性取得同帧数据。"""
        with self._pipeline_output_lock:
            return self._latest_pipeline_output

    def _advance_pipeline_epoch(self, reason: str) -> int:
        """使旧几何/旧数据源的在途包失效。调用方必须持有状态锁。"""
        self._pipeline_epoch += 1
        if self._pipeline_mailbox is not None:
            self._pipeline_mailbox.clear(reason)
        logger.info("检测流水线 epoch 更新为 %d：%s", self._pipeline_epoch, reason)
        return self._pipeline_epoch

    ## Update the reference of the visualization image

    def _publish_track_visualization(self, image, source_stamp, sequence):
        """后台线程只发布不可再修改的成品引用，不在这里调用 Qt。"""
        self.track_vis_source_stamp = source_stamp
        self.track_vis_sequence = int(sequence)
        self.track_vis_img = image

    def reset_projection_lidar_statistics(self, reason: str) -> None:
        """清空相机投影—雷达残差统计，不影响追踪状态。"""
        if hasattr(self, "tracker") and hasattr(
            self.tracker, "reset_projection_lidar_statistics"
        ):
            self.tracker.reset_projection_lidar_statistics(reason)
    
    def set_calibrated_param(self, calibrated_R, calibrated_T):
        """Set the calibrated parameters for the pixel to world transformation."""
        if calibrated_R is None or calibrated_T is None:
            logger.warning("标定参数为空，跳过外参更新。")
            return
        if (
            hasattr(self, "calibrated_R")
            and hasattr(self, "calibrated_T")
            and np.array_equal(np.asarray(self.calibrated_R), np.asarray(calibrated_R))
            and np.array_equal(np.asarray(self.calibrated_T), np.asarray(calibrated_T))
        ):
            return
        with self._pipeline_state_lock:
            self.pixel_world_transform = PixelToWorld.build_from_config_and_extrinsics(
                self.config, calibrated_R, calibrated_T
            )
            self.calibrated_R = calibrated_R
            self.calibrated_T = calibrated_T
            self.tracker.pixel_world_transform = self.pixel_world_transform
            self.reset_projection_lidar_statistics("外参更新")
            self.xy_map_manager.set_calibration(self.pixel_world_transform)
            self._advance_pipeline_epoch("相机外参整体替换")
        if self.tf_debug_publisher.enabled:
            ok = self.tf_debug_publisher.update_extrinsics(calibrated_R, calibrated_T)
            if not ok:
                logger.warning("tf debug 发布失败，请检查 ROS2/tf2 依赖或配置。")

    def set_xy_maps_enabled(self, enabled: bool) -> None:
        """运行时切换 XY Map；未标定时管理器保持等待状态。"""
        enabled = bool(enabled)
        changed = enabled != self.xy_map_manager.enabled
        self.xy_map_manager.set_enabled(enabled)
        if changed:
            self.reset_projection_lidar_statistics(
                "XY Maps 开启" if enabled else "XY Maps 关闭"
            )

    def get_xy_maps_status(self) -> str:
        return self.xy_map_manager.status_text

    def get_xy_maps_enabled(self) -> bool:
        return self.xy_map_manager.enabled

    def consume_xy_map_consistency_events(self) -> list[dict]:
        """供 UI 线程取走 XY Maps 一致性检查结果。"""
        return self.xy_map_manager.consume_consistency_events()

    def _export_xy_map_metrics(self, metrics: dict) -> None:
        if metrics.get("xy_map_generation_error"):
            self.reset_projection_lidar_statistics("XY Maps 生成失败并回退直接投影")
        else:
            self.reset_projection_lidar_statistics("XY Maps 重新生成完成")
        if getattr(self, "metrics_exporter", None) is None:
            return
        self.metrics_exporter.submit({"record_type": "xy_map", **metrics})

    ########################################################################
    def set_outpost_roi(self, outpost_roi):
        """设置前哨站图像 ROI，格式为原图坐标 (x1, y1, x2, y2)。"""
        if outpost_roi is None:
            self.outpost_roi = None
            return

        x1, y1, x2, y2 = [int(round(value)) for value in outpost_roi]
        if x2 <= x1 or y2 <= y1:
            logger.warning("前哨站 ROI 非法，已忽略: %s", outpost_roi)
            return

        self.outpost_roi = (x1, y1, x2, y2)

    def _crop_outpost_image(self, frame):
        """从当前帧中裁出前哨站 ROI。"""
        if self.outpost_roi is None or frame is None:
            return None

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.outpost_roi
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def _bi_image(self, frame):
        # """根据当前配置对前哨站 ROI 做红/蓝 HSV 二值化。"""
        """根据当前配置对前哨站 ROI 做颜色 HSV 二值化。"""
        if frame is None or frame.size == 0:
            return None

        hsv_code = (
            cv2.COLOR_BGR2HSV
            if self.camera_color_order == "bgr"
            else cv2.COLOR_RGB2HSV
        )
        hsv = cv2.cvtColor(frame, hsv_code)

        cfg = self.outpost_light_config
        reverse_mask = bool(cfg.get("reverse", False))

        h_min = int(cfg["h_min"])
        h_max = int(cfg["h_max"])
        lower = (
            int(cfg["h_min"]),
            int(cfg["s_min"]),
            int(cfg["v_min"]),
        )
        upper = (
            int(cfg["h_max"]),
            int(cfg["s_max"]),
            int(cfg["v_max"]),
        )
        if h_min <= h_max:
            mask = cv2.inRange(hsv, lower, upper)
            return cv2.bitwise_not(mask) if reverse_mask else mask

        lower_a = (0, int(cfg["s_min"]), int(cfg["v_min"]))
        upper_a = (h_max, int(cfg["s_max"]), int(cfg["v_max"]))
        lower_b = (h_min, int(cfg["s_min"]), int(cfg["v_min"]))
        upper_b = (179, int(cfg["s_max"]), int(cfg["v_max"]))

        mask = cv2.bitwise_or(cv2.inRange(hsv, lower_a, upper_a), cv2.inRange(hsv, lower_b, upper_b))
        return cv2.bitwise_not(mask) if reverse_mask else mask

    def judge_outpost_alive(self, outpost_image):
        """按红/蓝二值化面积比例判断前哨站是否存活。"""
        if outpost_image is None:
            self.outpost_binary_image = None
            self.outpost_alive_ratio = None
            return self.outpost_alive

        binary_image = self._bi_image(outpost_image)
        self.outpost_binary_image = binary_image
        if binary_image is None or binary_image.size == 0:
            self.outpost_alive_ratio = None
            return self.outpost_alive

        ratio = cv2.countNonZero(binary_image) / float(binary_image.size)
        self.outpost_alive_ratio = ratio
        return int(ratio >= float(self.outpost_light_config["alive_ratio_threshold"]))

    @staticmethod
    def _default_outpost_light_config():
        """默认前哨站灯光二值化配置。"""
        return {
            "color": "blue",
            "reverse": False,
            "h_min": 95,
            "h_max": 130,
            "s_min": 80,
            "s_max": 255,
            "v_min": 80,
            "v_max": 255,
            "alive_ratio_threshold": 0.03,
        }

    @staticmethod
    def _default_outpost_light_thresholds(color):
        """给不同颜色提供 params.yaml 缺省时的 HSV 阈值。"""
        defaults = {
            "blue": {"h_min": 95, "h_max": 130, "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
            "red": {"h_min": 170, "h_max": 10, "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
            "green": {"h_min": 35, "h_max": 85, "s_min": 60, "s_max": 255, "v_min": 60, "v_max": 255},
            "white": {"h_min": 0, "h_max": 179, "s_min": 0, "s_max": 70, "v_min": 160, "v_max": 255},
        }
        return defaults.get(color, defaults["blue"]).copy()

    def _load_outpost_light_config(self, cfg):
        """从 params.yaml 中读取前哨站灯光配置。"""
        result = self._default_outpost_light_config()
        cfg = cfg if isinstance(cfg, dict) else {}
        color = str(cfg.get("color", result["color"])).lower()
        result.update(self._default_outpost_light_thresholds(color))
        thresholds = cfg.get("hsv_thresholds", {})
        color_thresholds = thresholds.get(color, {}) if isinstance(thresholds, dict) else {}
        result.update({k: cfg[k] for k in result.keys() & cfg.keys() if k != "color"})
        result.update({k: color_thresholds[k] for k in result.keys() & color_thresholds.keys()})
        result["color"] = color if color in ("blue", "red", "green", "white") else "blue"
        return result

    def set_outpost_light_config(self, config):
        """更新前哨站灯光判断参数。"""
        if not isinstance(config, dict):
            return

        current = self._default_outpost_light_config()
        current.update(self.outpost_light_config)
        for key in current:
            if key in config:
                current[key] = config[key]
        current["color"] = str(current["color"]).lower()
        if current["color"] not in ("blue", "red", "green", "white"):
            current["color"] = "blue"
        self.outpost_light_config = current
    ########################################################################


    def run(self):
        """Run the main event loop for the tracker."""
        self.reset_projection_lidar_statistics("启动雷达站")
        self.set_event = Event()
        if self.pipeline_enabled:
            self._pipeline_worker_thread = Thread(
                target=self._pipeline_stage_b_worker,
                name="armor-tracking-pipeline-worker",
                daemon=True,
            )
            self._pipeline_worker_thread.start()
        self.thread = Thread(target=self.main_loop_thread, daemon=False)
        self.thread.start()

    def _signal_replay_tracking_ready(self) -> None:
        """通知统一启动器：模型、UI 标定和主循环均已就绪，可以启动 rosbag。"""
        resolved_cfg = self.test_config.get("_resolved", {})
        ready_file = str(resolved_cfg.get("tracking_ready_file", "")).strip()
        if not ready_file:
            return
        try:
            ready_file = os.path.abspath(os.path.expanduser(ready_file))
            os.makedirs(os.path.dirname(ready_file), exist_ok=True)
            with open(ready_file, "w", encoding="utf-8") as file:
                file.write(f"ready_wall_time={time.time():.9f}\n")
            logger.info("同步回放主循环已就绪: %s", ready_file)
        except OSError as error:
            logger.error("无法写入同步回放就绪标记 %s: %s", ready_file, error)

    def _format_lidar_points(self, lidar_points, faction):
        if not lidar_points:
            return []

        mirror_for_blue = self.lidar_config.get("mirror_for_blue", True)
        field_length = float(self.lidar_config.get("field_length", 28.0))
        field_width = float(self.lidar_config.get("field_width", 15.0))
        formatted = []
        for x, y in lidar_points:
            if mirror_for_blue and faction == FACTION.BLUE:
                x = field_length - x
                y = field_width - y
            formatted.append((x * 100.0, y * 100.0))
        return formatted

    def _format_projection_points(self, detections, faction):
        """提取本帧有效 car 底边中心投影，转换为右侧场地图使用的厘米坐标。"""
        if not detections:
            return []

        field_length = float(self.lidar_config.get("field_length", 28.0))
        field_width = float(self.lidar_config.get("field_width", 15.0))
        formatted = []
        for detection in detections:
            point = getattr(detection, "pos_3d", None)
            if point is None or len(point) < 3:
                continue
            try:
                x, y = solidwork2uwb(
                    [float(point[0]), float(point[1]), float(point[2])], faction
                )
            except (TypeError, ValueError, IndexError):
                continue
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if not (0.0 <= x <= field_length and 0.0 <= y <= field_width):
                continue
            formatted.append((x * 100.0, y * 100.0))
        return formatted

    def _format_fly_points(self, lidar_points, faction) -> list[tuple[int, int, int]]:
        if not lidar_points:
            return []

        mirror_for_blue = self.lidar_config.get("mirror_for_blue", True)
        field_length = float(self.lidar_config.get("field_length", 28.0))
        field_width = float(self.lidar_config.get("field_width", 15.0))
        formatted_points = []
        for x, y, z in lidar_points:
            if mirror_for_blue and faction == FACTION.BLUE:
                x = field_length - x
                y = field_width - y
            formatted_x = int(x * 100.0)
            formatted_y = int(y * 100.0)
            formatted_z = int(z * 100.0)
            formatted_points.append((formatted_x, formatted_y, formatted_z))
        return formatted_points

    @staticmethod
    def _to_uint16_coord_cm(value) -> int:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0
        if not math.isfinite(value) or value <= 0:
            return 0
        return max(0, min(0xFFFF, int(round(value))))

    def _update_localization_fps(self, now: float):
        """更新当前定位输出帧率，使用指数平滑减少抖动。"""
        if self.last_localization_time is not None:
            dt = now - self.last_localization_time
            if dt > 0:
                instant_fps = 1.0 / dt
                if self.localization_fps <= 0:
                    self.localization_fps = instant_fps
                else:
                    self.localization_fps = 0.8 * self.localization_fps + 0.2 * instant_fps
        self.last_localization_time = now

    def _record_timing_stat(self, stats, name: str, duration_s: float):
        """记录主流程耗时统计。"""
        duration_ms = float(duration_s) * 1000.0
        with self._timing_stats_lock:
            if stats is self.loop_timing_stats:
                self.loop_frame_timing_ms[name] = duration_ms
            if not getattr(self, "timing_debug_enabled", False):
                return

            stat = stats.get(name)
            if stat is None:
                stats[name] = {
                    "last_ms": duration_ms,
                    "avg_ms": duration_ms,
                    "max_ms": duration_ms,
                    "count": 1,
                }
                return

            stat["last_ms"] = duration_ms
            stat["avg_ms"] = (1.0 - self.timing_alpha) * stat["avg_ms"] + self.timing_alpha * duration_ms
            stat["max_ms"] = max(stat["max_ms"], duration_ms)
            stat["count"] += 1

    def _get_bottleneck_item(self, stats, exclude_total: bool = True):
        """获取某一组耗时里平均耗时最高的阶段。"""
        items = list(stats.items())
        if exclude_total:
            filtered_items = [
                item
                for item in items
                if "总计" not in item[0] and "总时间" not in item[0]
            ]
            if filtered_items:
                items = filtered_items
        if not items:
            return None
        return max(items, key=lambda item: item[1]["avg_ms"])

    def _format_bottleneck_text(self, item):
        """格式化瓶颈阶段文本。"""
        if item is None:
            return "暂无数据"
        name, stat = item
        return f"{self._friendly_timing_name(name)} avg {stat['avg_ms']:.1f} ms"

    @staticmethod
    def _friendly_timing_name(name: str) -> str:
        """把历史内部名转换成用户能直接理解的模块覆盖名。"""
        aliases = {
            "取图像": "01 获取最新相机帧（模块1总时间）",
            "前哨站判断": "02 前哨站灯光判断（模块2细分）",
            "雷达点读取": "02 雷达/己方/飞坡点快照（模块2细分）",
            "追踪器处理": "03-11 串行检测与跟踪（模块3至11总时间）",
            "车辆与装甲检测": "03-09 车辆与装甲神经网络检测（模块3至9总时间）",
            "观测预处理与投影": "09 观测构建与投影（模块9总时间）",
            "匹配状态机总计": "10 跟踪融合状态机（模块10总时间）",
            "追踪总计": "03-10 串行追踪调用（模块3至10总时间）",
            "输入快照": "10/输入快照整理（模块10细分）",
            "二维KF预测": "10/二维卡尔曼预测（模块10细分）",
            "己方哨兵雷达剔除": "10/己方哨兵雷达点剔除（模块10细分）",
            "彩色三元组匹配": "10/彩色相机-雷达三元组匹配（模块10细分）",
            "灰色身份维持匹配": "10/灰色装甲身份维持匹配（模块10细分）",
            "历史雷达匈牙利匹配": "10/历史轨迹与雷达匈牙利匹配（模块10细分）",
            "灰色模糊猜测匹配": "10/灰色装甲模糊匹配（模块10细分）",
            "纯雷达猜测": "10/纯雷达目标猜测（模块10细分）",
            "丢失与固定点处理": "10/目标丢失与固定点处理（模块10细分）",
            "输出提交": "11 跟踪输出整理（模块11细分）",
            "异步可视化提交": "12/异步可视化提交（模块12细分）",
            "TF调试发布": "12/TF调试发布（模块12细分）",
            "定位结果整理": "11 定位结果整理（模块11总时间）",
            "裁判消息打包": "12/裁判消息打包（模块12细分）",
            "后台合并可视化": "12/后台合并可视化（模块12细分）",
            "主循环总计": "01-12 串行主循环总时间（模块1至12总时间）",
        }
        return aliases.get(name, name)

    def _format_timing_section(self, title: str, stats, max_items=None):
        """按模块编号稳定格式化耗时统计。"""
        lines = [f"{title}:"]
        if not stats:
            lines.append("  暂无数据")
            return lines

        sorted_items = sorted(stats.items(), key=lambda item: self._timing_sort_key(item[0]))
        visible_items = (
            sorted_items if max_items is None else sorted_items[: int(max_items)]
        )
        for name, stat in visible_items:
            lines.append(
                f"  {self._friendly_timing_name(name)}: "
                f"{stat['last_ms']:.1f}/{stat['avg_ms']:.1f}/{stat['max_ms']:.1f}"
            )
        return lines

    def _timing_sort_key(self, name: str):
        """把 03.2.4.1 这类点分编号转为可稳定比较的数字键。"""
        friendly_name = self._friendly_timing_name(str(name))
        prefix = friendly_name.split(maxsplit=1)[0].replace("/", ".")
        endpoint = prefix.split("-")[-1]
        if re.fullmatch(r"\d+(?:\.\d+)*", endpoint):
            parts = tuple(int(part) for part in endpoint.split("."))
            is_range = 1 if "-" in prefix else 0
            # 非合计项中，Python tuple 天然保证 03 < 03.1 < 03.1.1。
            # 范围合计项按其结束模块排在该大模块的细分之后。
            return (0, parts[0], is_range, parts[1:], friendly_name)
        if friendly_name.startswith("A阶段"):
            return (0, 4, 2, (), friendly_name)
        if friendly_name.startswith("B阶段"):
            return (0, 11, 2, (), friendly_name)
        return (1, friendly_name)

    def _merge_timing_stats(self, *stats_groups):
        """按对用户显示的名称合并统计，消除跨分区重复词条。"""
        merged = {}
        for stats in stats_groups:
            for name, stat in stats.items():
                merged[self._friendly_timing_name(name)] = stat
        return merged

    def set_timing_debug_enabled(self, enabled: bool, clear_stats: bool = True):
        """运行时切换耗时调试统计开关。"""
        self.timing_debug_enabled = bool(enabled)
        if hasattr(self, "tracker") and self.tracker is not None:
            self.tracker.timing_debug_enabled = self.timing_debug_enabled
        if clear_stats:
            with self._timing_stats_lock:
                self.loop_timing_stats.clear()
            if hasattr(self, "tracker") and self.tracker is not None:
                timing_lock = getattr(self.tracker, "_timing_lock", None)
                if timing_lock is None:
                    self.tracker.track_timing_stats.clear()
                    self.tracker.detector_timing_stats.clear()
                else:
                    with timing_lock:
                        self.tracker.track_timing_stats.clear()
                        self.tracker.detector_timing_stats.clear()
            if self.visualization_worker is not None:
                self.visualization_worker.clear_timing_stats()

    def set_car_detector_conf_thres(self, conf_thres: float) -> float:
        """运行时设置整车检测置信度阈值，下一帧检测立即生效。"""
        conf_thres = max(0.0, min(1.0, float(conf_thres)))
        self.config.setdefault("car_detector", {})["conf_thres"] = conf_thres

        if hasattr(self, "tracker") and self.tracker is not None:
            if hasattr(self.tracker, "set_car_detector_conf_thres"):
                conf_thres = self.tracker.set_car_detector_conf_thres(conf_thres)
            elif hasattr(self.tracker, "car_detector") and hasattr(
                self.tracker.car_detector, "conf_thres"
            ):
                self.tracker.car_detector.conf_thres = conf_thres

        logger.info("整车检测置信度阈值已设置为 %.3f", conf_thres)
        return conf_thres

    def get_car_detector_conf_thres(self) -> float:
        """读取当前整车检测置信度阈值。"""
        if hasattr(self, "tracker") and self.tracker is not None:
            if hasattr(self.tracker, "get_car_detector_conf_thres"):
                return self.tracker.get_car_detector_conf_thres()
            if hasattr(self.tracker, "car_detector") and hasattr(
                self.tracker.car_detector, "conf_thres"
            ):
                return float(self.tracker.car_detector.conf_thres)
        return float(self.config.get("car_detector", {}).get("conf_thres", 0.3))

    def get_timing_debug_text(self) -> str:
        """返回当前跟踪定位链路的耗时调试文本。"""
        if not getattr(self, "timing_debug_enabled", False):
            return "跟踪定位耗时调试已关闭。"

        with self._timing_stats_lock:
            loop_stats = {
                name: dict(stat) for name, stat in self.loop_timing_stats.items()
            }
        timing_lock = getattr(self.tracker, "_timing_lock", None)
        if timing_lock is None:
            tracker_stats = {
                name: dict(stat)
                for name, stat in getattr(self.tracker, "track_timing_stats", {}).items()
            }
            detector_stats = {
                name: dict(stat)
                for name, stat in getattr(self.tracker, "detector_timing_stats", {}).items()
            }
        else:
            with timing_lock:
                tracker_stats = {
                    name: dict(stat)
                    for name, stat in getattr(self.tracker, "track_timing_stats", {}).items()
                }
                detector_stats = {
                    name: dict(stat)
                    for name, stat in getattr(self.tracker, "detector_timing_stats", {}).items()
                }
        visualization_stats = (
            self.visualization_worker.get_timing_stats()
            if self.visualization_worker is not None
            else {}
        )
        visualization_diagnostics = (
            self.visualization_worker.get_diagnostics()
            if self.visualization_worker is not None
            else {}
        )
        ordered_module_stats = self._merge_timing_stats(
            loop_stats,
            detector_stats,
            tracker_stats,
            visualization_stats,
        )

        completed = self.get_latest_pipeline_output()
        diagnostics = self.pipeline_diagnostics.snapshot()
        if completed is None:
            headline = "单帧延迟: 等待首次提交 | 吞吐: 等待首次提交"
        else:
            interval_ema_ms = float(
                completed.pipeline_timings_ms.get(
                    "稳态提交间隔EMA（界面吞吐口径）", 0.0
                )
            )
            interval_text = (
                "首帧"
                if interval_ema_ms <= 0.0
                else f"{interval_ema_ms:.1f} ms (EMA)"
            )
            headline = (
                f"单帧端到端延迟: {completed.processing_age_ms:.1f} ms | "
                f"稳定提交间隔: {interval_text} | "
                "稳定定位吞吐: "
                f"{float(completed.pipeline_timings_ms.get('稳定定位吞吐FPS（EMA界面口径）', 0.0)):.2f} FPS"
            )
        lines = [
            "跟踪定位耗时调试（单位 ms，顺序为 last/EMA平均/历史最大）",
            headline,
            "说明：流水线 FPS = 1/提交间隔；不能用单帧总延迟的倒数代替。",
            (
                f"帧序号: 源 {diagnostics.source_sequence} / "
                f"提交 {diagnostics.commit_sequence} | "
                f"待处理覆盖 {diagnostics.pending_replaced_total} | "
                f"过期丢弃 "
                f"{diagnostics.stale_before_stage_b_total + diagnostics.stale_before_state_total}"
            ),
            "",
            f"主循环最慢: {self._format_bottleneck_text(self._get_bottleneck_item(loop_stats))}",
            f"跟踪器最慢: {self._format_bottleneck_text(self._get_bottleneck_item(tracker_stats))}",
            f"检测器最慢: {self._format_bottleneck_text(self._get_bottleneck_item(detector_stats))}",
            f"异步可视化最慢: {self._format_bottleneck_text(self._get_bottleneck_item(visualization_stats))}",
            "",
        ]
        lines.extend(
            self._format_timing_section(
                "按模块编号排序（已合并跨分区重复词条）",
                ordered_module_stats,
            )
        )
        if visualization_diagnostics:
            lines.append(
                "  任务统计: "
                f"提交 {visualization_diagnostics['submitted']} / "
                f"完成 {visualization_diagnostics['rendered']} / "
                f"覆盖 {visualization_diagnostics['overwritten']} / "
                f"异常 {visualization_diagnostics['errors']}"
            )
        if hasattr(self.tracker, "get_projection_lidar_statistics_text"):
            lines.append("")
            lines.append(self.tracker.get_projection_lidar_statistics_text())
        return "\n".join(lines)

    def get_pipeline_module_text(self) -> str:
        """返回详情窗口独立展示的编号管线和最新一帧直观指标。"""
        completed = self.get_latest_pipeline_output()
        diagnostics = self.pipeline_diagnostics.snapshot()
        mode = "两级流水线" if self.pipeline_enabled else "兼容串行"
        lines = [
            f"运行模式：{mode}；epoch={self._pipeline_epoch}",
            "标记：[计时/原子: CPU] 表示计时层级和主要执行位置；CPU+GPU 表示混合阶段；CPU→GPU/GPU→CPU 表示跨设备传输。",
            "补充：[父级计时/原子: XXX] 表示该原子步骤计入最近父模块，没有伪造独立耗时。",
            "注意：CUDA Event 的GPU执行时间与CPU调用墙钟时间可能重叠，带“含等待”的项目不可与GPU子项直接求和。",
            "",
            "A 阶段（主循环线程，可与上一帧 B 阶段重叠）",
            "01 [模块: CPU] 获取并接纳最新相机帧",
            "  01.1 [计时/原子: CPU] 等待并取得 tracker 组最新帧",
            "  01.2 [计时/原子: CPU] 分配源序号、记录时间戳/尺寸、检查 epoch",
            "02 [模块: CPU] 同源状态与传感器快照",
            "  02.1 [计时/原子: CPU] 阵营快照",
            "  02.2 [计时/原子: CPU] 前哨站 ROI 裁剪",
            "  02.3 [计时/原子: CPU] HSV 二值化、亮区比例和存活判断",
            "  02.4 [计时/原子: CPU] 地面雷达快照",
            "  02.5 [计时/原子: CPU] 己方哨兵位置快照",
            "  02.6 [计时/原子: CPU] 飞坡雷达点快照",
            "03 [模块: CPU+GPU] 车辆检测与 BoT-SORT",
            "  03.1 [模块: CPU] 项目侧车辆模型输入准备",
            "    03.1.1 [计时/原子: CPU] 当前原始相机帧线性缩放到车辆模型输入尺寸（默认1280×1280）",
            "    03.1.2 [计时/原子: CPU] RGB→BGR 负步长共享视图",
            "  03.2 [模块: CPU+GPU] Ultralytics model.track",
            "    03.2.1 [计时: CPU+GPU] Ultralytics 内部预处理总时间",
            "      03.2.1.1 [父级计时/原子: CPU] LetterBox 尺寸检查与缩放填充",
            "      03.2.1.2 [父级计时/原子: CPU] batch 堆叠、BGR→RGB、BHWC→BCHW",
            "      03.2.1.3 [父级计时/原子: CPU] 连续内存化与 NumPy→Torch Tensor",
            "      03.2.1.4 [父级计时/原子: CPU→GPU] 输入张量上传",
            "      03.2.1.5 [父级计时/原子: GPU] FP16/FP32 转换与除以255归一化",
            "    03.2.2 [计时/原子: GPU] 车辆神经网络 TensorRT 推理",
            "    03.2.3 [计时: CPU+GPU] YOLO 后处理总时间",
            "      03.2.3.1 [父级计时/原子: GPU] 置信度/类别过滤与 NMS",
            "      03.2.3.2 [父级计时/原子: GPU] 检测框缩放回 model.track 输入尺寸",
            "      03.2.3.3 [父级计时/原子: CPU+GPU] Results/Boxes CPU对象封装GPU张量",
            "    03.2.4 [计时: GPU→CPU+CPU] BoT-SORT 回调总时间",
            "      03.2.4.1 [计时/原子: CPU] BOTSORT.update 跟踪更新（当前ReID关闭）",
            "        03.2.4.1.1 [父级计时/原子: CPU] 高低置信度检测分组与轨迹初始化",
            "        03.2.4.1.2 [父级计时/原子: CPU] 卡尔曼批量预测；GMC当前关闭",
            "        03.2.4.1.3 [父级计时/原子: CPU] 高置信度IoU第一轮关联；ReID当前关闭",
            "        03.2.4.1.4 [父级计时/原子: CPU] 低置信度 IoU 第二轮关联",
            "        03.2.4.1.5 [父级计时/原子: CPU] 未确认轨迹关联",
            "        03.2.4.1.6 [父级计时/原子: CPU] 新轨迹激活与丢失/删除状态维护",
            "        03.2.4.1.7 [父级计时/原子: CPU] 去重并生成激活轨迹数组",
            "      03.2.4.2 [计时: GPU→CPU+CPU] BoT-SORT回调中update之外的总时间",
            "        03.2.4.2.1 [父级计时/原子: GPU→CPU] Boxes转NumPy并组装跟踪器输入",
            "        03.2.4.2.2 [父级计时/原子: CPU] 轨迹结果按检测索引写回Results",
            "    03.2.5 [计时/原子: CPU+GPU] 无法归属到03.2.1–03.2.4的Ultralytics框架墙钟余量",
            "  03.3 [模块: CPU或GPU→CPU] 车辆跟踪结果解析",
            "    03.3.1 [计时/原子: CPU或GPU→CPU] 跟踪结果张量合并与 CPU 转换",
            "    03.3.2 [计时/原子: CPU] 坐标还原与 Python tuple 组装",
            "04 [模块: CPU] 车辆 ROI 生成",
            "  04.1 [计时/原子: CPU] 车辆框裁边与无效框/无ID过滤",
            "  04.2 [计时/原子: CPU] ROI切片与只读视图封装",
            "05 [模块: CPU] latest-wins 邮箱交接与提交准入",
            "  05.1 [计时/原子: CPU] A包在容量1邮箱中的驻留等待",
            "  05.2 [计时/原子: CPU] 关闭、epoch、顺序和首次年龄检查",
            "  05.3 [计时/原子: CPU] 状态事务前最终年龄检查",
            "B 阶段（单一装甲/跟踪线程，按提交帧序串行）",
            "06 [模块: CPU+GPU] 装甲板位置检测",
            "  06.1 [计时/原子: CPU] 各车辆ROI转BGR共享视图",
            "  06.2 [计时: CPU+GPU] Ultralytics装甲内部预处理",
            "    06.2.1 [父级计时/原子: CPU] LetterBox尺寸检查与缩放填充",
            "    06.2.2 [父级计时/原子: CPU] batch堆叠、BGR→RGB、BHWC→BCHW",
            "    06.2.3 [父级计时/原子: CPU] 连续内存化与NumPy→Torch Tensor",
            "    06.2.4 [父级计时/原子: CPU→GPU] 输入张量上传",
            "    06.2.5 [父级计时/原子: GPU] FP16/FP32转换与除以255归一化",
            "  06.3 [计时/原子: GPU] 装甲板位置神经网络推理",
            "  06.4 [计时: CPU+GPU] 装甲YOLO后处理",
            "    06.4.1 [父级计时/原子: GPU] 置信度/类别过滤与NMS",
            "    06.4.2 [父级计时/原子: GPU] 检测框缩放回各车辆ROI尺寸",
            "    06.4.3 [父级计时/原子: CPU+GPU] Results/Boxes CPU对象封装GPU张量",
            "  06.5 [计时/原子: CPU+GPU] 装甲Ultralytics框架调度余量",
            "  06.6 [计时/原子: GPU→CPU+CPU] 装甲结果批量回传与坐标组装",
            "07 [模块: CPU] 数字识别输入准备",
            "  07.1 [计时/原子: CPU] 装甲框裁边与装甲ROI切片",
            "  07.2 [计时/原子: CPU] 装甲ROI直传数字分类器（numpy直传）",
            "08 [模块: CPU+GPU] 数字神经网络识别",
            "  08.1 [计时: CPU] 每张数字图像NormalTransform",
            "    08.1.1 [父级计时/原子: CPU] 黑边填充为正方形",
            "    08.1.2 [父级计时/原子: CPU] 缩放到64×64",
            "    08.1.3 [父级计时/原子: CPU] PIL转Tensor",
            "    08.1.4 [父级计时/原子: CPU] ImageNet均值方差归一化",
            "  08.2 [计时/原子: CPU] 数字批量张量堆叠",
            "  08.3 [计时/原子: CPU→GPU] 批量张量上传并转FP16",
            "  08.4 [计时/原子: GPU] 数字神经网络FP16推理",
            "  08.5 [计时/原子: GPU] softmax、argmax与结果张量拼接",
            "  08.6 [计时/原子: GPU→CPU] 数字结果回传调用墙钟时间（含等待）",
            "09 [模块: CPU] 检测语义组装与射线投影",
            "  09.1 [计时/原子: CPU] 装甲颜色+数字合成class_id并分配回车辆",
            "  09.2 [计时/原子: CPU] 装甲候选解析与最佳装甲选择",
            "  09.3 [计时/原子: CPU] 车辆底边中心射线投影",
            "  09.4 [计时/原子: CPU] SingleDetectionResult封装",
            "  09.5 [计时/原子: CPU] 跟踪融合CarObservation构建",
            "10 [模块: CPU] 相机/雷达跟踪融合状态机",
            "  10.1 [计时/原子: CPU] 地面雷达输入快照整理",
            "  10.2 [计时/原子: CPU] 全目标二维卡尔曼预测",
            "  10.3 [计时/原子: CPU] BoT-ID语义历史更新",
            "  10.4 [计时/原子: CPU] 己方哨兵反馈匹配与雷达点剔除",
            "  10.5 [计时/原子: CPU] 彩色相机-雷达三元组匹配",
            "  10.6 [计时/原子: CPU] 灰色装甲身份维持匹配",
            "  10.7 [计时/原子: CPU] 旧猜测抢占标记",
            "  10.8 [计时/原子: CPU] 历史轨迹-雷达匈牙利匹配",
            "  10.9 [计时/原子: CPU] 灰色装甲模糊猜测匹配",
            "  10.10 [计时/原子: CPU] 纯雷达目标猜测",
            "  10.11 [计时/原子: CPU] 未匹配目标丢失与固定点处理",
            "  10.12 [计时/原子: CPU] 跟踪状态输出有效性整理",
            "11 [模块: CPU] 不可变输出快照与原子提交",
            "  11.1 [计时/原子: CPU] 跟踪/雷达/投影/飞坡/定位输出快照",
            "  11.2 [计时/原子: CPU] 提交序号、间隔EMA与诊断元数据",
            "  11.3 [计时/原子: CPU] CompletedFrame及兼容输出构造",
            "  11.4 [计时/原子: CPU] 诊断计数和最新输出原子发布",
            "12 [模块: CPU] 已提交帧的下游发布",
            "  12.1 [计时/原子: CPU] 异步可视化任务提交",
            "  12.2 [计时/原子: CPU] TF调试目标发布",
            "  12.3 [计时/原子: CPU] 裁判定位消息打包更新",
            "  12.4 [计时/原子: CPU] 定位吞吐和耗时统计合并",
            "  12.5 [计时/原子: CPU] 性能指标异步提交",
            "",
            "总时间覆盖关系：",
            "  A阶段总时间 = 模块 01–04",
            "  B阶段总时间 = 模块 06–11（不含邮箱等待和下游发布）",
            "  单帧端到端延迟 = 该帧进入 A 至模块 11 提交",
            "  稳态提交间隔 = 相邻两次模块 11 提交的时间差",
            "  界面稳定提交间隔 = 原始提交间隔的 EMA（alpha=debug.timing_ema_alpha，默认0.2）",
            "  实际吞吐 FPS = 1 / 界面稳定提交间隔",
            "  跨帧重叠节省估算 = max(0, A总时间+B总时间-提交间隔)",
            "",
        ]
        if completed is None:
            lines.append("最新数据：等待首次成功提交。")
        else:
            overlap_ms = float(
                completed.pipeline_timings_ms.get("跨帧重叠节省估算", 0.0)
            )
            interval_ema_ms = float(
                completed.pipeline_timings_ms.get(
                    "稳态提交间隔EMA（界面吞吐口径）", 0.0
                )
            )
            lines.extend(
                [
                    (
                        f"最新提交：源帧 #{completed.source_sequence} → "
                        f"提交 #{completed.commit_sequence}"
                    ),
                    f"  单帧端到端延迟：{completed.processing_age_ms:.2f} ms",
                    (
                        "  稳态提交间隔：首帧暂无"
                        if interval_ema_ms <= 0.0
                        else f"  稳态提交间隔EMA：{interval_ema_ms:.2f} ms"
                    ),
                    "  稳定定位吞吐EMA："
                    f"{float(completed.pipeline_timings_ms.get('稳定定位吞吐FPS（EMA界面口径）', 0.0)):.2f} FPS",
                    (
                        "  当前原始提交吞吐：首帧暂无"
                        if completed.commit_fps is None
                        else f"  当前原始提交吞吐：{completed.commit_fps:.2f} FPS"
                    ),
                    f"  跨帧重叠节省估算：{overlap_ms:.2f} ms",
                    "",
                    "最新提交的分模块耗时：",
                ]
            )
            for name, value in (
                list(completed.stage_a_timings_ms.items())
                + list(completed.stage_b_timings_ms.items())
                + list(completed.pipeline_timings_ms.items())
            ):
                lines.append(f"  {name}: {float(value):.2f} ms")
            with self._timing_stats_lock:
                commit_and_downstream = [
                    (name, dict(stat))
                    for name, stat in self.loop_timing_stats.items()
                    if name.startswith("11.3")
                    or name.startswith("11.4")
                    or name.startswith("11 输出")
                    or name.startswith("12")
                ]
                for name, stat in commit_and_downstream:
                    lines.append(
                        f"  {name}: {stat['last_ms']:.2f} ms"
                    )
        lines.append(
            f"累计：发布 {diagnostics.published_total} / "
            f"取走 {diagnostics.taken_total} / 提交 {diagnostics.committed_total} / "
            f"覆盖 {diagnostics.pending_replaced_total}"
        )
        return "\n".join(lines)


    def stop(self):
        """Stop the main event loop."""
        if getattr(self, "set_event", None) is None or getattr(self, "thread", None) is None:
            logger.warning("Main event loop is not running, skip loop thread stop.")
        else:
            with self._pipeline_state_lock:
                self.set_event.set()
                if self._pipeline_mailbox is not None:
                    self._pipeline_mailbox.close()
            self.thread.join()
        pipeline_worker_stopped = True
        if self._pipeline_worker_thread is not None:
            self._pipeline_worker_thread.join(
                timeout=float(self.pipeline_config["worker_join_timeout_sec"])
            )
            pipeline_worker_stopped = not self._pipeline_worker_thread.is_alive()
            if not pipeline_worker_stopped:
                logger.error(
                    "装甲跟踪流水线线程在 %.1f 秒内未退出；暂不关闭其下游依赖。",
                    self.pipeline_config["worker_join_timeout_sec"],
                )
                return
        if self.visualization_worker is not None:
            self.visualization_worker.close()
        if self.lidar_client is not None:
            self.lidar_client.stop()
        if self.lidar_fly_client is not None and self.lidar_fly_client is not self.lidar_client:
            self.lidar_fly_client.stop()
        if self.tf_debug_publisher is not None:
            self.tf_debug_publisher.stop()
        if getattr(self, "xy_map_manager", None) is not None:
            self.xy_map_manager.close()
        if self.metrics_exporter is not None:
            self.metrics_exporter.close()

    @staticmethod
    def _get_frame_age_ms(frame_stamp, now_wall_time: float):
        """计算相机时间戳到当前时刻的延迟，时钟域不一致时返回空值。"""
        try:
            age_s = float(now_wall_time) - float(frame_stamp)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(age_s) or age_s < -0.1 or age_s > 3600.0:
            return None
        return age_s * 1000.0

    def _export_frame_metrics(
        self,
        frame_id: int,
        frame_stamp,
        input_age_ms,
        processing_ms: float,
        error_text: str | None,
    ):
        """汇总当前帧各层计时；实际磁盘写入由后台线程完成。"""
        output_age_ms = self._get_frame_age_ms(frame_stamp, time.time())
        latest_completed = self.get_latest_pipeline_output()
        detector_stages_ms = {}
        if latest_completed is not None:
            detector_stages_ms.update(latest_completed.stage_a_timings_ms)
            detector_stages_ms.update(latest_completed.stage_b_timings_ms)
        self.metrics_exporter.submit(
            {
                "frame_id": int(frame_id),
                "frame_timestamp": (
                    float(frame_stamp) if frame_stamp is not None else None
                ),
                "input_frame_age_ms": input_age_ms,
                "end_to_end_latency_ms": output_age_ms,
                "processing_ms": float(processing_ms),
                "processing_fps": (
                    1000.0 / processing_ms if processing_ms > 0.0 else 0.0
                ),
                "error": error_text,
                "object_counts": {
                    "detections": len(self.latest_detection_results),
                    "active_tracks": sum(
                        1 for track in getattr(self, "tracks", []) if track.is_active
                    ),
                    "lidar_points": len(self.lidar_points_cm),
                    "fly_points": len(self.lidar_fly_points_cm),
                },
                "stages_ms": {
                    "main_loop": dict(self.loop_frame_timing_ms),
                    "tracker": dict(
                        getattr(self.tracker, "track_frame_timing_ms", {})
                    ),
                    "detector": dict(detector_stages_ms),
                    "visualization": (
                        self.visualization_worker.get_frame_timing_ms()
                        if self.visualization_worker is not None
                        else {}
                    ),
                },
                "visualization": (
                    self.visualization_worker.get_diagnostics()
                    if self.visualization_worker is not None
                    else {}
                ),
            }
        )

    #打包雷达到哨兵数据
    # def pack_radar2sentrymsg(self) -> Radar2SentryMessage:
    #     if self.faction == FACTION.RED or self.faction == FACTION.UNKONWN:  # Sending red target if blue
    #         is_blue = False
    #         hero_x = (
    #             self.divisions_pos[0].x / 100.0
    #             if self.divisions_pos[0].is_valid
    #             else -8888
    #         )
    #         hero_y = (
    #             self.divisions_pos[0].y / 100.0
    #             if self.divisions_pos[0].is_valid
    #             else -8888
    #         )
    #         engineer_x = (
    #             self.divisions_pos[1].x / 100.0
    #             if self.divisions_pos[1].is_valid
    #             else -8888
    #         )
    #         engineer_y = (
    #             self.divisions_pos[1].y / 100.0
    #             if self.divisions_pos[1].is_valid
    #             else -8888
    #         )
    #         standard_3_x = (
    #             self.divisions_pos[2].x / 100.0
    #             if self.divisions_pos[2].is_valid
    #             else -8888
    #         )
    #         standard_3_y = (
    #             self.divisions_pos[2].y / 100.0
    #             if self.divisions_pos[2].is_valid
    #             else -8888
    #         )
    #         standard_4_x = (
    #             self.divisions_pos[3].x / 100.0
    #             if self.divisions_pos[3].is_valid
    #             else -8888
    #         )
    #         standard_4_y = (
    #             self.divisions_pos[3].y / 100.0
    #             if self.divisions_pos[3].is_valid
    #             else -8888
    #         )
    #         sentry_x = (
    #             self.divisions_pos[4].x / 100.0
    #             if self.divisions_pos[4].is_valid
    #             else -8888
    #         )
    #         sentry_y = (
    #             self.divisions_pos[4].y / 100.0
    #             if self.divisions_pos[4].is_valid
    #             else -8888
    #         )
    #     else:
    #         is_blue = True
    #         hero_x = (
    #             self.divisions_pos[5].x / 100.0
    #             if self.divisions_pos[5].is_valid
    #             else -8888
    #         )
    #         hero_y = (
    #             self.divisions_pos[5].y / 100.0
    #             if self.divisions_pos[5].is_valid
    #             else -8888
    #         )
    #         engineer_x = (
    #             self.divisions_pos[6].x / 100.0
    #             if self.divisions_pos[6].is_valid
    #             else -8888
    #         )
    #         engineer_y = (
    #             self.divisions_pos[6].y / 100.0
    #             if self.divisions_pos[6].is_valid
    #             else -8888
    #         )
    #         standard_3_x = (
    #             self.divisions_pos[7].x / 100.0
    #             if self.divisions_pos[7].is_valid
    #             else -8888
    #         )
    #         standard_3_y = (
    #             self.divisions_pos[7].y / 100.0
    #             if self.divisions_pos[7].is_valid
    #             else -8888
    #         )
    #         standard_4_x = (
    #             self.divisions_pos[8].x / 100.0
    #             if self.divisions_pos[8].is_valid
    #             else -8888
    #         )
    #         standard_4_y = (
    #             self.divisions_pos[8].y / 100.0
    #             if self.divisions_pos[8].is_valid
    #             else -8888
    #         )
    #         sentry_x = (
    #             self.divisions_pos[9].x / 100.0
    #             if self.divisions_pos[9].is_valid
    #             else -8888
    #         )
    #         sentry_y = (
    #             self.divisions_pos[9].y / 100.0
    #             if self.divisions_pos[9].is_valid
    #             else -8888
    #         )
    #     return Radar2SentryMessage(
    #         is_blue=is_blue,
    #         hero_x=hero_x,
    #         hero_y=hero_y,
    #         engineer_x=engineer_x,
    #         engineer_y=engineer_y,
    #         standard_3_x=standard_3_x,
    #         standard_3_y=standard_3_y,
    #         standard_4_x=standard_4_x,
    #         standard_4_y=standard_4_y,
    #         sentry_x=sentry_x,
    #         sentry_y=sentry_y,  # 40
    #         suggested_target=0,  # No target suggested
    #         flags=0,  # No flags set
    #     )

    # def pack_radar2clientmsg(self) -> Radar2ClientMessage:

    #     if self.faction == FACTION.RED or self.faction == FACTION.UNKONWN:
    #         hero_x = (
    #             int(self.divisions_pos[0].x) if self.divisions_pos[0].is_valid else 0
    #         )
    #         hero_y = (
    #             int(self.divisions_pos[0].y) if self.divisions_pos[0].is_valid else 0
    #         )
    #         engineer_x = (
    #             int(self.divisions_pos[1].x) if self.divisions_pos[1].is_valid else 0
    #         )
    #         engineer_y = (
    #             int(self.divisions_pos[1].y) if self.divisions_pos[1].is_valid else 0
    #         )
    #         standard_3_x = (
    #             int(self.divisions_pos[2].x) if self.divisions_pos[2].is_valid else 0
    #         )
    #         standard_3_y = (
    #             int(self.divisions_pos[2].y) if self.divisions_pos[2].is_valid else 0
    #         )
    #         standard_4_x = (
    #             int(self.divisions_pos[3].x) if self.divisions_pos[3].is_valid else 0
    #         )
    #         standard_4_y = (
    #             int(self.divisions_pos[3].y) if self.divisions_pos[3].is_valid else 0
    #         )
    #         sentry_x = (
    #             int(self.divisions_pos[4].x) if self.divisions_pos[4].is_valid else 0
    #         )
    #         sentry_y = (
    #             int(self.divisions_pos[4].y) if self.divisions_pos[4].is_valid else 0
    #         )
    #     else:
    #         hero_x = (
    #             int(self.divisions_pos[5].x) if self.divisions_pos[5].is_valid else 0
    #         )
    #         hero_y = (
    #             int(self.divisions_pos[5].y) if self.divisions_pos[5].is_valid else 0
    #         )
    #         engineer_x = (
    #             int(self.divisions_pos[6].x) if self.divisions_pos[6].is_valid else 0
    #         )
    #         engineer_y = (
    #             int(self.divisions_pos[6].y) if self.divisions_pos[6].is_valid else 0
    #         )
    #         standard_3_x = (
    #             int(self.divisions_pos[7].x) if self.divisions_pos[7].is_valid else 0
    #         )
    #         standard_3_y = (
    #             int(self.divisions_pos[7].y) if self.divisions_pos[7].is_valid else 0
    #         )
    #         standard_4_x = (
    #             int(self.divisions_pos[8].x) if self.divisions_pos[8].is_valid else 0
    #         )
    #         standard_4_y = (
    #             int(self.divisions_pos[8].y) if self.divisions_pos[8].is_valid else 0
    #         )
    #         sentry_x = (
    #             int(self.divisions_pos[9].x) if self.divisions_pos[9].is_valid else 0
    #         )
    #         sentry_y = (
    #             int(self.divisions_pos[9].y) if self.divisions_pos[9].is_valid else 0
    #         )
    #     return Radar2ClientMessage(
    #         hero_x=hero_x,
    #         hero_y=hero_y,
    #         engineer_x=engineer_x,
    #         engineer_y=engineer_y,
    #         standard_3_x=standard_3_x,
    #         standard_3_y=standard_3_y,
    #         standard_4_x=standard_4_x,
    #         standard_4_y=standard_4_y,
    #         standard_5_x=0,  # No standard 5 position
    #         standard_5_y=0,  # No standard 5 position
    #         sentry_x=sentry_x,
    #         sentry_y=sentry_y,  # 40
    #     )

    ########################################################################
    def pack_radar2sentrymsg(self) -> Radar2SentryMessage:
        """打包雷达到哨兵消息，写入 0x0A05 派生的可瞄准状态。"""
        return Radar2SentryMessage(
            is_blue=self.faction == FACTION.BLUE,
            targetable_flags=self.referee.build_sentry_targetable_flags(),
        )
    ########################################################################
    
    def pack_radar2clientmsg(self):
        """
        更新敌方位置信息
        self.divisions_pos需要修改
        """
        if self.faction == FACTION.RED or self.faction == FACTION.UNKONWN:
            hero_x = (
                int(self.divisions_pos[0].x) if self.divisions_pos[0].is_valid else 0
            )
            hero_y = (
                int(self.divisions_pos[0].y) if self.divisions_pos[0].is_valid else 0
            )
            engineer_x = (
                int(self.divisions_pos[1].x) if self.divisions_pos[1].is_valid else 0
            )
            engineer_y = (
                int(self.divisions_pos[1].y) if self.divisions_pos[1].is_valid else 0
            )
            standard_3_x = (
                int(self.divisions_pos[2].x) if self.divisions_pos[2].is_valid else 0
            )
            standard_3_y = (
                int(self.divisions_pos[2].y) if self.divisions_pos[2].is_valid else 0
            )
            standard_4_x = (
                int(self.divisions_pos[3].x) if self.divisions_pos[3].is_valid else 0
            )
            standard_4_y = (
                int(self.divisions_pos[3].y) if self.divisions_pos[3].is_valid else 0
            )
            sentry_x = (
                int(self.divisions_pos[4].x) if self.divisions_pos[4].is_valid else 0
            )
            sentry_y = (
                int(self.divisions_pos[4].y) if self.divisions_pos[4].is_valid else 0
            )
        else:
            hero_x = (
                int(self.divisions_pos[5].x) if self.divisions_pos[5].is_valid else 0
            )
            hero_y = (
                int(self.divisions_pos[5].y) if self.divisions_pos[5].is_valid else 0
            )
            engineer_x = (
                int(self.divisions_pos[6].x) if self.divisions_pos[6].is_valid else 0
            )
            engineer_y = (
                int(self.divisions_pos[6].y) if self.divisions_pos[6].is_valid else 0
            )
            standard_3_x = (
                int(self.divisions_pos[7].x) if self.divisions_pos[7].is_valid else 0
            )
            standard_3_y = (
                int(self.divisions_pos[7].y) if self.divisions_pos[7].is_valid else 0
            )
            standard_4_x = (
                int(self.divisions_pos[8].x) if self.divisions_pos[8].is_valid else 0
            )
            standard_4_y = (
                int(self.divisions_pos[8].y) if self.divisions_pos[8].is_valid else 0
            )
            sentry_x = (
                int(self.divisions_pos[9].x) if self.divisions_pos[9].is_valid else 0
            )
            sentry_y = (
                int(self.divisions_pos[9].y) if self.divisions_pos[9].is_valid else 0
            )

        drone_x, drone_y = (0, 0)
        if self.lidar_fly_points_cm:
            drone_x, drone_y, _ = self.lidar_fly_points_cm[0]

        return Radar2ClientMessage(
            enemy_hero_x=self._to_uint16_coord_cm(hero_x),
            enemy_hero_y=self._to_uint16_coord_cm(hero_y),
            enemy_engineer_x=self._to_uint16_coord_cm(engineer_x),
            enemy_engineer_y=self._to_uint16_coord_cm(engineer_y),
            enemy_standard_3_x=self._to_uint16_coord_cm(standard_3_x),
            enemy_standard_3_y=self._to_uint16_coord_cm(standard_3_y),
            enemy_standard_4_x=self._to_uint16_coord_cm(standard_4_x),
            enemy_standard_4_y=self._to_uint16_coord_cm(standard_4_y),
            enemy_drone_x=self._to_uint16_coord_cm(drone_x),
            enemy_drone_y=self._to_uint16_coord_cm(drone_y),
            enemy_sentry_x=self._to_uint16_coord_cm(sentry_x),
            enemy_sentry_y=self._to_uint16_coord_cm(sentry_y),

            self_hero_x=self._to_uint16_coord_cm(self.referee.radar2client_msg.self_hero_x),
            self_hero_y=self._to_uint16_coord_cm(self.referee.radar2client_msg.self_hero_y),
            self_engineer_x=self._to_uint16_coord_cm(self.referee.radar2client_msg.self_engineer_x),
            self_engineer_y=self._to_uint16_coord_cm(self.referee.radar2client_msg.self_engineer_y),
            self_standard_3_x=self._to_uint16_coord_cm(self.referee.radar2client_msg.self_standard_3_x),
            self_standard_3_y=self._to_uint16_coord_cm(self.referee.radar2client_msg.self_standard_3_y),
            self_standard_4_x=self._to_uint16_coord_cm(self.referee.radar2client_msg.self_standard_4_x),
            self_standard_4_y=self._to_uint16_coord_cm(self.referee.radar2client_msg.self_standard_4_y),
            self_drone_x=self._to_uint16_coord_cm(self.referee.radar2client_msg.self_drone_x),
            self_drone_y=self._to_uint16_coord_cm(self.referee.radar2client_msg.self_drone_y),
            self_sentry_x=self._to_uint16_coord_cm(self.referee.radar2client_msg.self_sentry_x),
            self_sentry_y=self._to_uint16_coord_cm(self.referee.radar2client_msg.self_sentry_y),
        )

    def main_loop_thread(self):
        """按启动时读取的配置选择流水线或兼容串行实现。"""
        if self.pipeline_enabled:
            self._pipeline_stage_a_loop()
        else:
            self._serial_main_loop_thread()

    @staticmethod
    def _snapshot_track(track) -> TrackOutputSnapshot:
        position = getattr(track, "output_position_2d_m", None)
        box = getattr(track, "last_car_box", None)
        pos_3d = tuple(float(value) for value in getattr(track, "pos_3d", [0, 0, 0])[:3])
        return TrackOutputSnapshot(
            class_id=int(track.class_id),
            name=str(track.name),
            identity_state=track.identity_state,
            spatial_state=track.spatial_state,
            frame_source=track.frame_source,
            output_valid=bool(track.output_valid),
            output_position_2d_m=(
                None
                if position is None
                else (float(position[0]), float(position[1]))
            ),
            is_guessed_output=bool(track.is_guessed_output),
            pos_3d=pos_3d,
            last_car_box=(
                None if box is None else tuple(float(value) for value in box[:4])
            ),
        )

    @staticmethod
    def _build_division_snapshots(tracks) -> tuple[DivisionPositionSnapshot, ...]:
        positions = [
            DivisionPositionSnapshot(-1, -1, False) for _ in range(10)
        ]
        for track in tracks:
            class_id = int(track.class_id)
            position = getattr(track, "output_position_2d_m", None)
            if bool(track.output_valid) and position is not None:
                positions[class_id] = DivisionPositionSnapshot(
                    x=int(round(float(position[0]) * 100.0)),
                    y=int(round(float(position[1]) * 100.0)),
                    is_valid=True,
                    source=str(track.frame_source.value),
                    is_guessed=bool(track.is_guessed_output),
                )
        return tuple(positions)

    def _pipeline_packet_is_stale(self, packet: StageAPacket) -> bool:
        return (
            time.monotonic() - float(packet.ingress_monotonic)
            > float(self.pipeline_config["max_commit_age_sec"])
        )

    def _pipeline_stage_a_loop(self):
        """A 阶段：取图、同源快照、车辆推理/BoT-SORT 和车辆 ROI 裁剪。"""
        logger.info("检测流水线已启用：A=车辆检测，B=装甲识别与跟踪融合。")
        time.sleep(2)
        self._signal_replay_tracking_ready()
        timeout = float(self.pipeline_config["camera_timeout_sec"])
        while not self.set_event.is_set():
            frame_started = time.perf_counter()
            try:
                stage_started = time.perf_counter()
                frame, frame_stamp = self.camera.get_image_latest(
                    "tracker", timeout=timeout
                )
                get_frame_ms = (time.perf_counter() - stage_started) * 1000.0
                if frame is None:
                    continue

                ingress_monotonic = time.monotonic()
                metadata_started = time.perf_counter()
                source_sequence = self.pipeline_diagnostics.next_source_sequence()
                self.processed_frame_count = source_sequence
                input_age_ms = self._get_frame_age_ms(frame_stamp, time.time())
                height, width = frame.shape[:2]
                frame_size = (int(width), int(height))
                with self._pipeline_state_lock:
                    if (
                        self._pipeline_last_frame_size is not None
                        and frame_size != self._pipeline_last_frame_size
                    ):
                        self._advance_pipeline_epoch(
                            f"输入分辨率变化 {self._pipeline_last_frame_size}->{frame_size}"
                        )
                    self._pipeline_last_frame_size = frame_size
                    try:
                        numeric_stamp = float(frame_stamp)
                    except (TypeError, ValueError):
                        numeric_stamp = None
                    if (
                        numeric_stamp is not None
                        and math.isfinite(numeric_stamp)
                        and self._pipeline_last_frame_stamp is not None
                        and numeric_stamp < self._pipeline_last_frame_stamp - 0.1
                    ):
                        self._advance_pipeline_epoch(
                            "相机/回放时间戳回退，判定数据源已重启"
                        )
                    if numeric_stamp is not None and math.isfinite(numeric_stamp):
                        self._pipeline_last_frame_stamp = numeric_stamp
                    packet_epoch = self._pipeline_epoch
                metadata_ms = (time.perf_counter() - metadata_started) * 1000.0

                stage_started = time.perf_counter()
                atomic_started = time.perf_counter()
                faction = self.referee.get_faction()
                faction_ms = (time.perf_counter() - atomic_started) * 1000.0
                atomic_started = time.perf_counter()
                outpost_image = self._crop_outpost_image(frame)
                outpost_crop_ms = (time.perf_counter() - atomic_started) * 1000.0
                atomic_started = time.perf_counter()
                outpost_alive = int(bool(self.judge_outpost_alive(outpost_image)))
                outpost_judge_ms = (time.perf_counter() - atomic_started) * 1000.0
                atomic_started = time.perf_counter()
                lidar_snapshot = None
                lidar_points = None
                if self.lidar_client is not None:
                    if hasattr(self.lidar_client, "get_snapshot"):
                        lidar_snapshot = self.lidar_client.get_snapshot()
                        lidar_points = list(lidar_snapshot.points_xy_m)
                    else:
                        lidar_points = self.lidar_client.get_points()
                        lidar_snapshot = SimpleNamespace(
                            points_xy_m=tuple(lidar_points),
                            ros_stamp_sec=None,
                            received_monotonic=time.monotonic(),
                        )
                ground_lidar_ms = (time.perf_counter() - atomic_started) * 1000.0
                atomic_started = time.perf_counter()
                ally_snapshot = None
                if hasattr(self.referee, "get_sentry_self_position_snapshot"):
                    ally_snapshot = self.referee.get_sentry_self_position_snapshot()
                ally_snapshot_ms = (time.perf_counter() - atomic_started) * 1000.0
                atomic_started = time.perf_counter()
                lidar_fly_points = ()
                if self.lidar_fly_client is not None:
                    lidar_fly_points = tuple(
                        tuple(float(value) for value in point[:3])
                        for point in self.lidar_fly_client.get_xyz_points()
                    )
                fly_lidar_ms = (time.perf_counter() - atomic_started) * 1000.0
                snapshot_ms = (time.perf_counter() - stage_started) * 1000.0

                # 该时刻代表此包所有相机/雷达观测的状态机时间。
                observation_monotonic = time.monotonic()
                cars, car_timings = self.tracker.detect_cars(frame)         # detect
                stage_a_ms = (time.perf_counter() - frame_started) * 1000.0
                published_monotonic = time.monotonic()
                stage_a_valid_frame_ms = (
                    published_monotonic - ingress_monotonic
                ) * 1000.0
                packet_frame = frame.view()
                packet_frame.flags.writeable = False
                timings = {
                    "01 获取并接纳最新相机帧（模块01总时间）": get_frame_ms + metadata_ms,
                    "01.1 等待并取得最新相机帧（原子模块01.1）": get_frame_ms,
                    "01.2 帧序号时间戳分辨率与epoch检查（原子模块01.2）": metadata_ms,
                    "02 同源状态与传感器快照（模块02总时间）": snapshot_ms,
                    "02.1 阵营快照（原子模块02.1）": faction_ms,
                    "02.2 前哨站ROI裁剪（原子模块02.2）": outpost_crop_ms,
                    "02.3 前哨站颜色二值化与存活判断（原子模块02.3）": outpost_judge_ms,
                    "02.4 地面雷达快照（原子模块02.4）": ground_lidar_ms,
                    "02.5 己方哨兵位置快照（原子模块02.5）": ally_snapshot_ms,
                    "02.6 飞坡雷达点快照（原子模块02.6）": fly_lidar_ms,
                    **car_timings,
                    "01-04 A阶段总时间（模块1至4总时间）": stage_a_ms,
                    "A阶段有效帧处理时间（模块2至4总时间）": (
                        stage_a_valid_frame_ms
                    ),
                }
                packet = StageAPacket(
                    pipeline_epoch=packet_epoch,
                    source_sequence=source_sequence,
                    frame=packet_frame,
                    frame_stamp=frame_stamp,
                    frame_width=width,
                    frame_height=height,
                    input_frame_age_ms=input_age_ms,
                    ingress_monotonic=ingress_monotonic,
                    observation_monotonic=observation_monotonic,
                    published_monotonic=published_monotonic,
                    faction=faction,
                    lidar_snapshot=lidar_snapshot,
                    ally_snapshot=ally_snapshot,
                    lidar_fly_points=lidar_fly_points,
                    outpost_alive=outpost_alive,
                    cars=cars,
                    stage_a_timings_ms=freeze_mapping(timings),
                )
                replaced = self._pipeline_mailbox.publish_latest(packet)
                if self.set_event.is_set():
                    self.pipeline_diagnostics.increment("shutdown_drop_total")
                    continue
                if replaced:
                    self._export_pipeline_event(
                        "pipeline_drop", packet, "pending_replaced_previous"
                    )
                self.inference_fps = 1000.0 / stage_a_ms if stage_a_ms > 0 else 0.0
                for name, value in timings.items():
                    self._record_timing_stat(
                        self.loop_timing_stats, name, float(value) / 1000.0
                    )
            except Exception:
                import traceback

                self.pipeline_diagnostics.increment("stage_a_error_total")
                logger.error("流水线 A 阶段异常：%s", traceback.format_exc())
                self.metrics_exporter.submit(
                    {
                        "record_type": "pipeline_error",
                        "pipeline_enabled": True,
                        "stage": "stage_a",
                        "reason": "stage_a_error",
                        "diagnostics": vars(
                            self.pipeline_diagnostics.snapshot()
                        ),
                    }
                )

    def _pipeline_stage_b_worker(self):
        """B 阶段：装甲/数字推理、跟踪状态事务以及一致性输出提交。"""
        last_committed_source = 0
        while True:
            packet = self._pipeline_mailbox.take_latest()
            if packet is None:
                return
            taken_monotonic = time.monotonic()
            admission_started = time.perf_counter()
            stage_b_started = time.perf_counter()
            if self.set_event.is_set():
                self.pipeline_diagnostics.increment("shutdown_drop_total")
                continue
            if packet.pipeline_epoch != self._pipeline_epoch:
                self.pipeline_diagnostics.increment("epoch_drop_total")
                continue
            if packet.source_sequence <= last_committed_source:
                self.pipeline_diagnostics.increment("out_of_order_drop_total")
                continue
            if self._pipeline_packet_is_stale(packet):
                self.pipeline_diagnostics.increment("stale_before_stage_b_total")
                self._export_pipeline_event("pipeline_drop", packet, "stale_before_stage_b")
                continue

            mailbox_wait_ms = (taken_monotonic - packet.published_monotonic) * 1000.0
            admission_ms = (time.perf_counter() - admission_started) * 1000.0
            armor_started = time.perf_counter()
            armor_error = None
            try:
                detections, _, armor_timings = self.tracker.detect_armors(
                    packet.frame, packet.cars, packet.observation_monotonic
                )
            except Exception as error:
                armor_error = str(error)
                self.pipeline_diagnostics.increment("armor_error_total")
                self.pipeline_diagnostics.increment("stage_b_error_total")
                logger.exception(
                    "源帧 %d 装甲识别异常，保留车辆/BoT-SORT 观测继续融合。",
                    packet.source_sequence,
                )
                detections = self.tracker.build_vehicle_only_detections(
                    packet.cars, packet.observation_monotonic
                )
                armor_timings = {}
            armor_total_ms = (time.perf_counter() - armor_started) * 1000.0
            armor_timings.setdefault(
                "06-08 装甲板与数字神经网络识别（模块6至8总时间）",
                armor_total_ms,
            )

            state_processed = False
            try:
                effective_faction = self.tracker._effective_faction(packet.faction)
                preprocess_started = time.perf_counter()
                cars = self.tracker._build_car_observations(
                    detections, effective_faction
                )
                preprocess_ms = (
                    time.perf_counter() - preprocess_started
                ) * 1000.0

                prestate_check_started = time.perf_counter()
                if self._pipeline_packet_is_stale(packet):
                    self.pipeline_diagnostics.increment("stale_before_state_total")
                    self._export_pipeline_event(
                        "pipeline_drop", packet, "stale_before_state"
                    )
                    continue
                prestate_check_ms = (
                    time.perf_counter() - prestate_check_started
                ) * 1000.0

                with self._pipeline_state_lock:
                    if self.set_event.is_set():
                        self.pipeline_diagnostics.increment("shutdown_drop_total")
                        continue
                    if packet.pipeline_epoch != self._pipeline_epoch:
                        self.pipeline_diagnostics.increment("epoch_drop_total")
                        continue
                    if packet.source_sequence <= last_committed_source:
                        self.pipeline_diagnostics.increment(
                            "out_of_order_drop_total"
                        )
                        continue
                    if self._pipeline_packet_is_stale(packet):
                        self.pipeline_diagnostics.increment(
                            "stale_before_state_total"
                        )
                        continue

                    with self.tracker._timing_lock:
                        self.tracker.track_frame_timing_ms = {}
                    fusion_started = time.perf_counter()
                    self.tracker.process_observations(
                        cars,
                        packet.lidar_snapshot,
                        packet.ally_snapshot,
                        faction=effective_faction,
                        now_monotonic=packet.observation_monotonic,
                    )
                    state_processed = True
                    fusion_ms = (time.perf_counter() - fusion_started) * 1000.0
                    self.tracker.latest_detections = list(detections)

                    snapshot_started = time.perf_counter()
                    tracker_lidar_points = tuple(
                        tuple(float(value) for value in point[:2])
                        for point in self.tracker.latest_lidar_points
                    )
                    owners = dict(self.tracker.latest_lidar_matches)
                    projection_points = tuple(
                        self._format_projection_points(
                            detections, effective_faction
                        )
                    )
                    fly_points_cm = tuple(
                        self._format_fly_points(
                            packet.lidar_fly_points, packet.faction
                        )
                    )
                    track_snapshots = tuple(
                        self._snapshot_track(track) for track in self.tracker.tracks
                    )
                    divisions = self._build_division_snapshots(self.tracker.tracks)
                    snapshot_ms = (time.perf_counter() - snapshot_started) * 1000.0
                    commit_started = time.perf_counter()
                    commit_now = time.monotonic()
                    diagnostics_before_commit = (
                        self.pipeline_diagnostics.snapshot()
                    )
                    commit_sequence = (
                        diagnostics_before_commit.commit_sequence + 1
                    )
                    previous_commit_monotonic = (
                        diagnostics_before_commit.last_commit_monotonic
                    )
                    commit_interval = (
                        None
                        if previous_commit_monotonic is None
                        else commit_now - previous_commit_monotonic
                    )
                    commit_interval_ms = (
                        None
                        if commit_interval is None
                        else commit_interval * 1000.0
                    )
                    if commit_interval_ms is not None:
                        if self._commit_interval_ema_ms is None:
                            self._commit_interval_ema_ms = commit_interval_ms
                        else:
                            self._commit_interval_ema_ms = (
                                (1.0 - self.timing_alpha)
                                * self._commit_interval_ema_ms
                                + self.timing_alpha * commit_interval_ms
                            )
                    commit_fps = (
                        None
                        if not commit_interval or commit_interval <= 0.0
                        else 1.0 / commit_interval
                    )
                    stage_b_ms = (
                        time.perf_counter() - stage_b_started
                    ) * 1000.0
                    processing_age_ms = (
                        commit_now - packet.ingress_monotonic
                    ) * 1000.0
                    commit_metadata_ms = (
                        time.perf_counter() - commit_started
                    ) * 1000.0
                    stage_b_timings = {
                        **armor_timings,
                        "09.5 跟踪融合观测构建（原子模块09.5）": preprocess_ms,
                        "10 跟踪融合状态机（模块10总时间）": fusion_ms,
                        "11.1-11.2 输出快照与提交元数据（模块11.1至11.2总时间）": (
                            snapshot_ms + commit_metadata_ms
                        ),
                        "11.1 跟踪雷达投影飞坡与定位输出快照（原子模块11.1）": snapshot_ms,
                        "11.2 提交序号间隔EMA与诊断元数据计算（原子模块11.2）": commit_metadata_ms,
                        "06-11 B阶段总时间（模块6至11总时间）": stage_b_ms,
                    }
                    stage_a_ms = float(
                        packet.stage_a_timings_ms.get(
                            "A阶段有效帧处理时间（模块2至4总时间）",
                            0.0,
                        )
                    )
                    serial_equivalent_ms = stage_a_ms + stage_b_ms
                    overlap_gain_ms = (
                        0.0
                        if commit_interval_ms is None
                        else max(0.0, serial_equivalent_ms - commit_interval_ms)
                    )
                    pipeline_timings = {
                        "05 邮箱交接与提交准入（模块05总时间）": (
                            mailbox_wait_ms + admission_ms + prestate_check_ms
                        ),
                        "05.1 A包在latest-wins邮箱驻留等待（原子模块05.1）": mailbox_wait_ms,
                        "05.2 epoch顺序关闭与首次年龄检查（原子模块05.2）": admission_ms,
                        "05.3 状态事务前最终年龄检查（原子模块05.3）": prestate_check_ms,
                        "01-11 单帧端到端处理延迟": processing_age_ms,
                        "稳态提交间隔（吞吐口径）": (
                            0.0 if commit_interval_ms is None else commit_interval_ms
                        ),
                        "稳态提交间隔EMA（界面吞吐口径）": (
                            0.0
                            if self._commit_interval_ema_ms is None
                            else self._commit_interval_ema_ms
                        ),
                        "稳定定位吞吐FPS（EMA界面口径）": (
                            0.0
                            if not self._commit_interval_ema_ms
                            else 1000.0 / self._commit_interval_ema_ms
                        ),
                        "跨帧重叠节省估算": overlap_gain_ms,
                    }
                    completed_build_started = time.perf_counter()
                    completed = CompletedFrame(
                        pipeline_epoch=packet.pipeline_epoch,
                        source_sequence=packet.source_sequence,
                        commit_sequence=commit_sequence,
                        source_gap_from_previous_commit=(
                            packet.source_sequence - last_committed_source
                        ),
                        frame=packet.frame,
                        frame_stamp=packet.frame_stamp,
                        input_frame_age_ms=packet.input_frame_age_ms,
                        faction=packet.faction,
                        outpost_alive=packet.outpost_alive,
                        detections=tuple(detections),
                        tracks=track_snapshots,
                        tracker_lidar_points=tracker_lidar_points,
                        lidar_point_owners=freeze_mapping(owners),
                        projection_points_cm=projection_points,
                        lidar_fly_points_cm=fly_points_cm,
                        divisions_pos=divisions,
                        stage_a_timings_ms=packet.stage_a_timings_ms,
                        stage_b_timings_ms=freeze_mapping(stage_b_timings),
                        pipeline_timings_ms=freeze_mapping(pipeline_timings),
                        processing_age_ms=processing_age_ms,
                        commit_interval_ms=commit_interval_ms,
                        commit_fps=commit_fps,
                        error=armor_error,
                    )
                    compatible_lidar_points_cm = [
                        (point[0] * 100.0, point[1] * 100.0)
                        for point in tracker_lidar_points
                    ]
                    compatible_divisions = [
                        RefereeDivision2dPosition(
                            item.x,
                            item.y,
                            item.is_valid,
                            item.source,
                            item.is_guessed,
                        )
                        for item in divisions
                    ]
                    completed_build_ms = (
                        time.perf_counter() - completed_build_started
                    ) * 1000.0
                    atomic_publish_started = time.perf_counter()
                    recorded_sequence, _ = self.pipeline_diagnostics.record_commit(
                        commit_now
                    )
                    if recorded_sequence != commit_sequence:
                        raise RuntimeError(
                            "流水线 commit_sequence 在提交事务中发生竞争"
                        )

                    # 兼容字段和新快照在同一临界区内替换，UI 不会拼接不同帧。
                    with self._pipeline_output_lock:
                        self._latest_pipeline_output = completed
                        self.faction = packet.faction
                        self.outpost_alive = packet.outpost_alive
                        self.latest_detection_results = list(detections)
                        self.latest_detection_source_frame = packet.frame
                        self.latest_detection_source_stamp = packet.frame_stamp
                        self.tracks = list(track_snapshots)
                        self.lidar_points_cm = compatible_lidar_points_cm
                        self.lidar_point_owners = owners
                        self.projection_points_cm = list(projection_points)
                        self.lidar_fly_points_cm = list(fly_points_cm)
                        self.divisions_pos = compatible_divisions
                    atomic_publish_ms = (
                        time.perf_counter() - atomic_publish_started
                    ) * 1000.0
                    self._record_timing_stat(
                        self.loop_timing_stats,
                        "11.3 CompletedFrame与兼容输出构造（原子模块11.3）",
                        completed_build_ms / 1000.0,
                    )
                    self._record_timing_stat(
                        self.loop_timing_stats,
                        "11.4 诊断计数与最新输出原子发布（原子模块11.4）",
                        atomic_publish_ms / 1000.0,
                    )
                    self._record_timing_stat(
                        self.loop_timing_stats,
                        "11 输出快照与原子提交（模块11总时间）",
                        (
                            snapshot_ms
                            + commit_metadata_ms
                            + completed_build_ms
                            + atomic_publish_ms
                        )
                        / 1000.0,
                    )
                    last_committed_source = packet.source_sequence
            except Exception:
                import traceback

                if state_processed:
                    self.pipeline_diagnostics.increment("commit_error_total")
                    failure_reason = "commit_error"
                else:
                    self.pipeline_diagnostics.increment("tracker_error_total")
                    failure_reason = "tracker_error"
                self.pipeline_diagnostics.increment("stage_b_error_total")
                logger.error("流水线 B 阶段异常：%s", traceback.format_exc())
                self._export_pipeline_event(
                    "pipeline_error", packet, failure_reason
                )
                continue

            # 以下均为已提交快照的下游消费者，不再持有状态锁。
            try:
                self._publish_completed_frame_downstream(completed)
            except Exception:
                logger.exception(
                    "源帧 %d 已提交，但下游发布异常；流水线继续运行。",
                    completed.source_sequence,
                )

    def _publish_completed_frame_downstream(self, completed: CompletedFrame):
        """发布可视化、TF、裁判消息和指标；慢消费者不得反压状态事务。"""
        downstream_started = time.perf_counter()
        atomic_started = time.perf_counter()
        try:
            if self.visualization_worker is not None:
                self.visualization_worker.submit(
                    completed.frame,
                    completed.detections,
                    completed.tracks,
                    source_stamp=completed.frame_stamp,
                )
        except Exception:
            logger.exception("提交帧 %d 的异步可视化发布失败。", completed.commit_sequence)
        visualization_submit_ms = (time.perf_counter() - atomic_started) * 1000.0
        atomic_started = time.perf_counter()
        try:
            if self.tf_debug_publisher.enabled:
                self.tf_debug_publisher.update_targets(
                    [
                        {
                            "class_id": track.class_id,
                            "pos_3d": list(track.pos_3d),
                        }
                        for track in completed.tracks
                        if track.output_valid
                    ]
                )
        except Exception:
            logger.exception("提交帧 %d 的 TF 发布失败。", completed.commit_sequence)
        tf_publish_ms = (time.perf_counter() - atomic_started) * 1000.0
        atomic_started = time.perf_counter()
        try:
            self.referee.latest_fly_points = list(completed.lidar_fly_points_cm)
            self.referee.update_tracking_radar2client_msg(
                self.pack_radar2clientmsg()
            )
        except Exception:
            logger.exception("提交帧 %d 的裁判消息打包失败。", completed.commit_sequence)
        referee_pack_ms = (time.perf_counter() - atomic_started) * 1000.0
        atomic_started = time.perf_counter()
        self.localization_fps = float(
            completed.pipeline_timings_ms.get(
                "稳定定位吞吐FPS（EMA界面口径）",
                completed.commit_fps or 0.0,
            )
        )
        self.last_localization_time = time.time()
        for name, value in completed.stage_b_timings_ms.items():
            self._record_timing_stat(
                self.loop_timing_stats, name, float(value) / 1000.0
            )
        timing_merge_ms = (time.perf_counter() - atomic_started) * 1000.0
        before_metrics_ms = (time.perf_counter() - downstream_started) * 1000.0
        metrics_started = time.perf_counter()
        try:
            self._export_pipeline_metrics(completed, before_metrics_ms)
        except Exception:
            logger.exception("提交帧 %d 的性能指标导出失败。", completed.commit_sequence)
        metrics_submit_ms = (time.perf_counter() - metrics_started) * 1000.0
        downstream_ms = (time.perf_counter() - downstream_started) * 1000.0
        for name, value in (
            ("12.1 异步可视化任务提交（原子模块12.1）", visualization_submit_ms),
            ("12.2 TF调试目标发布（原子模块12.2）", tf_publish_ms),
            ("12.3 裁判定位消息打包更新（原子模块12.3）", referee_pack_ms),
            ("12.4 定位吞吐与耗时统计合并（原子模块12.4）", timing_merge_ms),
            ("12.5 性能指标异步提交（原子模块12.5）", metrics_submit_ms),
            ("12 下游异步发布与消息打包（模块12总时间）", downstream_ms),
        ):
            self._record_timing_stat(
                self.loop_timing_stats, name, value / 1000.0
            )

    def _export_pipeline_event(
        self, record_type: str, packet: StageAPacket, reason: str
    ) -> None:
        self.metrics_exporter.submit(
            {
                "record_type": record_type,
                "pipeline_enabled": True,
                "pipeline_epoch": packet.pipeline_epoch,
                "source_sequence": packet.source_sequence,
                "reason": reason,
                "diagnostics": vars(self.pipeline_diagnostics.snapshot()),
            }
        )

    def _export_pipeline_metrics(
        self, completed: CompletedFrame, downstream_ms: float
    ) -> None:
        diagnostics = self.pipeline_diagnostics.snapshot()
        pipeline_metrics = {
            **vars(diagnostics),
            "enabled": True,
            "epoch": completed.pipeline_epoch,
            "source_sequence": completed.source_sequence,
            "commit_sequence": completed.commit_sequence,
            "source_gap_from_previous_commit": int(
                completed.source_gap_from_previous_commit
            ),
            "stage_a_ms": float(
                completed.stage_a_timings_ms.get(
                    "01-04 A阶段总时间（模块1至4总时间）", 0.0
                )
            ),
            "stage_b_ms": float(
                completed.stage_b_timings_ms.get(
                    "06-11 B阶段总时间（模块6至11总时间）", 0.0
                )
            ),
            "mailbox_wait_ms": float(
                completed.pipeline_timings_ms.get(
                    "05.1 A包在latest-wins邮箱驻留等待（原子模块05.1）",
                    0.0,
                )
            ),
            "processing_age_ms": completed.processing_age_ms,
            "commit_interval_ms": completed.commit_interval_ms,
            "commit_interval_ema_ms": (
                None
                if completed.commit_interval_ms is None
                else completed.pipeline_timings_ms.get(
                    "稳态提交间隔EMA（界面吞吐口径）"
                )
            ),
            "commit_fps": completed.commit_fps,
            "commit_fps_ema": (
                None
                if completed.commit_interval_ms is None
                else completed.pipeline_timings_ms.get(
                    "稳定定位吞吐FPS（EMA界面口径）"
                )
            ),
        }
        self.metrics_exporter.submit(
            {
                "record_type": "frame",
                "frame_id": completed.source_sequence,
                "frame_timestamp": completed.frame_stamp,
                "pipeline_enabled": True,
                "pipeline_epoch": completed.pipeline_epoch,
                "source_sequence": completed.source_sequence,
                "commit_sequence": completed.commit_sequence,
                "source_commit_gap": completed.source_gap_from_previous_commit,
                "source_gap_from_previous_commit": (
                    completed.source_gap_from_previous_commit
                ),
                "input_frame_age_ms": completed.input_frame_age_ms,
                "end_to_end_latency_ms": self._get_frame_age_ms(
                    completed.frame_stamp, time.time()
                ),
                "processing_ms": completed.processing_age_ms,
                # 兼容旧分析器；流水线真实吞吐必须读取 commit_fps。
                "processing_fps": (
                    1000.0 / completed.processing_age_ms
                    if completed.processing_age_ms > 0
                    else 0.0
                ),
                "commit_interval_ms": completed.commit_interval_ms,
                "commit_interval_ema_ms": (
                    None
                    if completed.commit_interval_ms is None
                    else completed.pipeline_timings_ms.get(
                        "稳态提交间隔EMA（界面吞吐口径）"
                    )
                ),
                "commit_fps": completed.commit_fps,
                "commit_fps_ema": (
                    None
                    if completed.commit_interval_ms is None
                    else completed.pipeline_timings_ms.get(
                        "稳定定位吞吐FPS（EMA界面口径）"
                    )
                ),
                "error": completed.error,
                "object_counts": {
                    "detections": len(completed.detections),
                    "active_tracks": sum(
                        1 for track in completed.tracks if track.output_valid
                    ),
                    "lidar_points": len(completed.tracker_lidar_points),
                    "fly_points": len(completed.lidar_fly_points_cm),
                },
                "stages_ms": {
                    "stage_a": dict(completed.stage_a_timings_ms),
                    "stage_b": dict(completed.stage_b_timings_ms),
                    "pipeline": dict(completed.pipeline_timings_ms),
                    "downstream": {
                        "12 下游发布（模块12总时间）": downstream_ms
                    },
                    "tracker": dict(
                        getattr(self.tracker, "track_frame_timing_ms", {})
                    ),
                    "detector": dict(
                        getattr(self.tracker, "detector_frame_timing_ms", {})
                    ),
                },
                "diagnostics": vars(diagnostics),
                "pipeline": pipeline_metrics,
            }
        )

    def _serial_main_loop_thread(self):

        logger.info(
            "Tracker initialized and the model weights are loaded, starting main loop."
        )
        time.sleep(2)  # Allow some time for the tracker to initialize
        self._signal_replay_tracking_ready()
        while not self.set_event.is_set():
            start = time.perf_counter()
            self.loop_frame_timing_ms = {}
            frame_received = False
            time_stamp = None
            input_age_ms = None
            frame_ingress_monotonic = None
            frame_error = None
            # Update faction
            self.faction = self.referee.get_faction()

            try:
                # Get the latest frame from the camera 相机数据获取
                stage_start = time.perf_counter()
                # frame, time_stamp -> np.ndarray, float
                frame, time_stamp = self.camera.get_image_latest(
                    "tracker",
                    timeout=float(self.pipeline_config["camera_timeout_sec"]),
                )
                self._record_timing_stat(
                    self.loop_timing_stats,
                    "取图像",
                    time.perf_counter() - stage_start,
                )
                if frame is None:
                    # logger.warning("No frame received from camera, skipping iteration.")
                    continue
                frame_received = True
                frame_ingress_monotonic = time.monotonic()
                frame_id = self.pipeline_diagnostics.next_source_sequence()
                self.pipeline_diagnostics.increment("published_total")
                self.pipeline_diagnostics.increment("taken_total")
                self.processed_frame_count = frame_id
                input_age_ms = self._get_frame_age_ms(time_stamp, time.time())

                ########################################################################
                stage_start = time.perf_counter()
                outpost_image = self._crop_outpost_image(frame)
                self.outpost_alive = int(bool(self.judge_outpost_alive(outpost_image)))
                self._record_timing_stat(
                    self.loop_timing_stats,
                    "前哨站判断",
                    time.perf_counter() - stage_start,
                )
                ########################################################################

                stage_start = time.perf_counter()
                lidar_snapshot = None
                lidar_points = None
                if self.lidar_client is not None:
                    if hasattr(self.lidar_client, "get_snapshot"):
                        lidar_snapshot = self.lidar_client.get_snapshot()
                        lidar_points = list(lidar_snapshot.points_xy_m)
                    else:
                        lidar_points = self.lidar_client.get_points()
                        lidar_snapshot = SimpleNamespace(
                            points_xy_m=tuple(lidar_points),
                            ros_stamp_sec=None,
                            received_monotonic=time.monotonic(),
                        )
                ally_snapshot = None
                if hasattr(self.referee, "get_sentry_self_position_snapshot"):
                    ally_snapshot = self.referee.get_sentry_self_position_snapshot()
                self.lidar_points_cm = self._format_lidar_points(
                    lidar_points, self.faction
                )
                lidar_fly_points = []
                if self.lidar_fly_client is not None:
                    lidar_fly_points = self.lidar_fly_client.get_xyz_points()
                self.lidar_fly_points_cm = self._format_fly_points(
                    lidar_fly_points, self.faction
                )
                self.referee.latest_fly_points = self.lidar_fly_points_cm
                self._record_timing_stat(
                    self.loop_timing_stats,
                    "雷达点读取",
                    time.perf_counter() - stage_start,
                )

                # Process the frame with the tracker 使用两个数据进行跟踪定位
                # TODO: 或许可以先image的detect再获得雷达？这样可以减少雷达的误差。或者放在各自的线程里？
                stage_start = time.perf_counter()
                self.tracks, _, _ = self.tracker.track(
                    frame,
                    lidar_snapshot=lidar_snapshot,
                    ally_snapshot=ally_snapshot,
                    faction=self.faction,
                )
                self._record_timing_stat(
                    self.loop_timing_stats,
                    "追踪器处理",
                    time.perf_counter() - stage_start,
                )
                self.latest_detection_results = list(
                    getattr(self.tracker, "latest_detections", [])
                )
                # 使用追踪器实际参与本帧匹配的雷达点，保证 UI 点序号与匹配结果严格对齐。
                tracker_lidar_points = tuple(
                    getattr(self.tracker, "latest_lidar_points", ())
                )
                self.lidar_points_cm = [
                    (float(point[0]) * 100.0, float(point[1]) * 100.0)
                    for point in tracker_lidar_points
                ]
                self.lidar_point_owners = dict(
                    getattr(self.tracker, "latest_lidar_matches", {})
                )
                projection_faction = getattr(self.tracker, "faction", self.faction)
                self.projection_points_cm = self._format_projection_points(
                    self.latest_detection_results, projection_faction
                )
                # 相机缓冲区以不可变 ndarray 发布；保留引用即可，数据集实际导出时再复制。
                self.latest_detection_source_frame = frame
                self.latest_detection_source_stamp = time_stamp

                stage_start = time.perf_counter()
                if self.visualization_worker is not None:
                    self.visualization_worker.submit(
                        frame,
                        self.latest_detection_results,
                        self.tracks,
                        source_stamp=time_stamp,
                    )
                self._record_timing_stat(
                    self.loop_timing_stats,
                    "异步可视化提交",
                    time.perf_counter() - stage_start,
                )

                stage_start = time.perf_counter()
                if self.tf_debug_publisher.enabled:
                    # 发布场地坐标系下的目标定位结果，便于在 RViz 中观察 target 与 camera 的相对关系
                    target_payload = []
                    for track in self.tracks:
                        if not track.output_valid:
                            continue
                        target_payload.append(
                            {
                                "class_id": int(track.class_id),
                                "pos_3d": [
                                    float(track.pos_3d[0]),
                                    float(track.pos_3d[1]),
                                    float(track.pos_3d[2]),
                                ],
                            }
                        )
                    self.tf_debug_publisher.update_targets(target_payload)
                self._record_timing_stat(
                    self.loop_timing_stats,
                    "TF调试发布",
                    time.perf_counter() - stage_start,
                )

                stage_start = time.perf_counter()
                for track in self.tracks:
                    if track.output_valid and track.output_position_2d_m is not None:
                        x_m, y_m = track.output_position_2d_m
                        self.divisions_pos[track.class_id] = RefereeDivision2dPosition(
                            x=int(round(float(x_m) * 100.0)),
                            y=int(round(float(y_m) * 100.0)),
                            is_valid=True,
                            source=str(track.frame_source.value),
                            is_guessed=bool(track.is_guessed_output),
                        )
                    else:
                        self.divisions_pos[track.class_id] = RefereeDivision2dPosition(
                            x=-1,
                            y=-1,
                            is_valid=False,
                            source="invalid",
                            is_guessed=False,
                        )
                self._update_localization_fps(time.time())
                self._record_timing_stat(
                    self.loop_timing_stats,
                    "定位结果整理",
                    time.perf_counter() - stage_start,
                )

                ## 向哨兵和裁判系统发送数据
                ## TODO
                stage_start = time.perf_counter()
                # self.referee.radar2sentry_msg = self.pack_radar2sentrymsg()
                self.referee.update_tracking_radar2client_msg(self.pack_radar2clientmsg())
                self._record_timing_stat(
                    self.loop_timing_stats,
                    "裁判消息打包",
                    time.perf_counter() - stage_start,
                )

                # 串行兼容模式也发布同一种 CompletedFrame，UI 无需分支读取多字段。
                commit_now = time.monotonic()
                commit_sequence, commit_interval = (
                    self.pipeline_diagnostics.record_commit(commit_now)
                )
                commit_interval_ms = (
                    None if commit_interval is None else commit_interval * 1000.0
                )
                if commit_interval_ms is not None:
                    if self._commit_interval_ema_ms is None:
                        self._commit_interval_ema_ms = commit_interval_ms
                    else:
                        self._commit_interval_ema_ms = (
                            (1.0 - self.timing_alpha)
                            * self._commit_interval_ema_ms
                            + self.timing_alpha * commit_interval_ms
                        )
                commit_fps = (
                    None
                    if not commit_interval or commit_interval <= 0.0
                    else 1.0 / commit_interval
                )
                track_snapshots = tuple(
                    self._snapshot_track(track) for track in self.tracker.tracks
                )
                division_snapshots = self._build_division_snapshots(
                    self.tracker.tracks
                )
                processing_age_ms = (
                    commit_now - frame_ingress_monotonic
                ) * 1000.0
                serial_completed = CompletedFrame(
                    pipeline_epoch=self._pipeline_epoch,
                    source_sequence=frame_id,
                    commit_sequence=commit_sequence,
                    source_gap_from_previous_commit=1,
                    frame=frame,
                    frame_stamp=time_stamp,
                    input_frame_age_ms=input_age_ms,
                    faction=self.faction,
                    outpost_alive=self.outpost_alive,
                    detections=tuple(self.latest_detection_results),
                    tracks=track_snapshots,
                    tracker_lidar_points=tracker_lidar_points,
                    lidar_point_owners=freeze_mapping(self.lidar_point_owners),
                    projection_points_cm=tuple(self.projection_points_cm),
                    lidar_fly_points_cm=tuple(self.lidar_fly_points_cm),
                    divisions_pos=division_snapshots,
                    stage_a_timings_ms=freeze_mapping(),
                    stage_b_timings_ms=freeze_mapping(
                        getattr(self.tracker, "track_frame_timing_ms", {})
                    ),
                    pipeline_timings_ms=freeze_mapping(
                        {
                            "01-11 单帧端到端处理延迟": processing_age_ms,
                            "稳态提交间隔（吞吐口径）": (
                                0.0
                                if commit_interval_ms is None
                                else commit_interval_ms
                            ),
                            "稳态提交间隔EMA（界面吞吐口径）": (
                                0.0
                                if self._commit_interval_ema_ms is None
                                else self._commit_interval_ema_ms
                            ),
                            "稳定定位吞吐FPS（EMA界面口径）": (
                                0.0
                                if not self._commit_interval_ema_ms
                                else 1000.0 / self._commit_interval_ema_ms
                            ),
                        }
                    ),
                    processing_age_ms=processing_age_ms,
                    commit_interval_ms=commit_interval_ms,
                    commit_fps=commit_fps,
                )
                with self._pipeline_output_lock:
                    self._latest_pipeline_output = serial_completed

            except Exception as e:
                import traceback

                frame_error = str(e)
                logger.error(f"Error in main loop: {traceback.format_exc()}")
            finally:
                end = time.perf_counter()
                total_duration = end - start
                self._record_timing_stat(
                    self.loop_timing_stats,
                    "主循环总计",
                    total_duration,
                )
                self.inference_fps = 1 / total_duration if total_duration > 0 else 0
                if frame_received:
                    self._export_frame_metrics(
                        frame_id=frame_id,
                        frame_stamp=time_stamp,
                        input_age_ms=input_age_ms,
                        processing_ms=total_duration * 1000.0,
                        error_text=frame_error,
                    )
                if self.log_inference_fps:
                    logger.info(f"Inference FPS: {self.inference_fps:.2f}")
