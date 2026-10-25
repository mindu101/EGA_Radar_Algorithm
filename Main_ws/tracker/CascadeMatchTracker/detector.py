from __future__ import annotations

import time
from threading import RLock
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tracker.CascadeMatchTracker.type import (
    ArmorColor,
    ArmorObservation,
    ProjectionMode,
    SingleDetectionResult,
    TrackingState,
)
from tracker.CascadeMatchTracker.pipeline import CarStageItem


class BaseDetector:
    """整车、装甲板与 digit 的批量检测前端。"""

    def __init__(
        self,
        config: dict,
        pixel_world_transform,
        visualize: bool = False,
        xy_map_manager=None,
    ) -> None:
        self.device = config["device"]
        self.class_names = config["armor_detector"]["class_names"]
        self.config = config
        self.pixel_world_transform = pixel_world_transform
        self.xy_map_manager = xy_map_manager
        self.visualize = bool(visualize)

        from model.yolov12.predictor_with_tracker import PredictorWithTracker

        car_cfg = config["car_detector"]
        self.car_detector = PredictorWithTracker(
            model_path=car_cfg["weights_path"],
            img_size=car_cfg["img_size"],
            max_det=car_cfg["max_det"],
            conf_thres=car_cfg["conf_thres"],
            iou_thres=car_cfg["iou_thres"],
            tracker_config_path=car_cfg["tracker_config_path"],
            verbose=car_cfg.get("verbose", False),
        )

        armor_cfg = config["armor_detector"]
        if not armor_cfg["use_enhanced_detector"]:
            raise ValueError("新状态机要求 armor_detector.use_enhanced_detector=true")
        # three_step 路径（pfa_yolov5 + 传统灯条颜色分类器）已移除：
        # 现役配置只有 two_step（装甲 YOLO + 数字分类），历史实现见 git 记录。
        if armor_cfg["armor_detector_type"] == "three_step":
            raise ValueError(
                "armor_detector_type=three_step 已移除，请改用 two_step"
            )
        from model.armor_detector import TwoStepArmorDetectorClassifier

        self.armor_detector = TwoStepArmorDetectorClassifier.from_config(config)

        digit_cfg = config["digit_detector"]
        self.digit_weight = float(digit_cfg["digit_confidence_weight"])
        self.armor_weight = float(digit_cfg["armor_confidence_weight"])
        self.min_digit_confidence = float(digit_cfg["min_digit_confidence"])
        self.min_combined_confidence = float(digit_cfg["min_combined_confidence"])
        self.ignored_digit_index = int(digit_cfg["ignored_digit_index"])

        debug_cfg = config.get("debug", {})
        self.timing_debug_enabled = bool(debug_cfg.get("timing_debug_enabled", True))
        self.timing_alpha = float(debug_cfg.get("timing_ema_alpha", 0.2))
        self._timing_lock = RLock()
        self.detector_timing_stats = {}
        self.detector_frame_timing_ms = {}
        self.track_frame_timing_ms = {}
        visual_cfg = config.get("visualization", {})
        self.separate_detection_visualization = bool(visual_cfg.get("separate_detection_view", False))
        self.vis_box_thickness = max(1, int(visual_cfg.get("box_thickness", 2)))
        self.vis_label_font_scale = max(0.1, float(visual_cfg.get("label_font_scale", 0.5)))
        self.vis_label_font_thickness = max(1, int(visual_cfg.get("label_font_thickness", 2)))
        self.vis_label_offset_y = max(0, int(visual_cfg.get("label_offset_y", 10)))
        self.vis_panel_title_font_scale = max(0.1, float(visual_cfg.get("panel_title_font_scale", 0.7)))
        self.vis_panel_text_font_scale = max(0.1, float(visual_cfg.get("panel_text_font_scale", 0.5)))
        self.vis_panel_font_thickness = max(1, int(visual_cfg.get("panel_font_thickness", 1)))

        cascade_cfg = config["tracker"]["cascade"]
        self.tracks = [
            TrackingState(
                class_id=class_id,
                name=class_name,
                position_kalman_config=cascade_cfg["position_kalman"],
                max_frames=int(cascade_cfg["bot_history"]["queue_size"]),
            )
            for class_id, class_name in enumerate(self.class_names[:10])
        ]

    def _record_timing_stat(self, stats, name: str, duration_s: float) -> None:
        duration_ms = float(duration_s) * 1000.0
        with self._timing_lock:
            if stats is self.detector_timing_stats:
                self.detector_frame_timing_ms[name] = duration_ms
            elif stats is getattr(self, "track_timing_stats", None):
                self.track_frame_timing_ms[name] = duration_ms
            if not self.timing_debug_enabled:
                return
            stat = stats.get(name)
            if stat is None:
                stats[name] = {"last_ms": duration_ms, "avg_ms": duration_ms, "max_ms": duration_ms, "count": 1}
                return
            stat["last_ms"] = duration_ms
            stat["avg_ms"] = (1.0 - self.timing_alpha) * stat["avg_ms"] + self.timing_alpha * duration_ms
            stat["max_ms"] = max(stat["max_ms"], duration_ms)
            stat["count"] += 1

    def set_car_detector_conf_thres(self, conf_thres: float) -> float:
        value = max(0.0, min(1.0, float(conf_thres)))
        self.config.setdefault("car_detector", {})["conf_thres"] = value
        if hasattr(self.car_detector, "conf_thres"):
            self.car_detector.conf_thres = value
        return value

    def get_car_detector_conf_thres(self) -> float:
        return float(getattr(self.car_detector, "conf_thres", self.config["car_detector"]["conf_thres"]))

    @staticmethod
    def _armor_color(raw_class_id: int) -> Optional[ArmorColor]:
        if 0 <= raw_class_id <= 4:
            return ArmorColor.BLUE
        if 5 <= raw_class_id <= 9:
            return ArmorColor.RED
        if 10 <= raw_class_id <= 14:
            return ArmorColor.GRAY
        return None

    @staticmethod
    def _clip_box(box: Sequence[float], width: int, height: int) -> Optional[List[float]]:
        x1 = max(0.0, min(float(width), float(box[0])))
        y1 = max(0.0, min(float(height), float(box[1])))
        x2 = max(0.0, min(float(width), float(box[2])))
        y2 = max(0.0, min(float(height), float(box[3])))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _project_car_box(
        self, car_box: Sequence[float], frame_size: Tuple[int, int]
    ) -> tuple[Optional[List[float]], ProjectionMode]:
        u = (float(car_box[0]) + float(car_box[2])) * 0.5
        v = float(car_box[3])
        if self.xy_map_manager is None:
            point = self.pixel_world_transform.pixel_to_world((u, v))
            mode = "direct"
        else:
            point, mode = self.xy_map_manager.project_pixel(
                u,
                v,
                frame_size,
                self.pixel_world_transform.pixel_to_world,
            )
        if point is None:
            return None, ProjectionMode.XY_MAP if mode == "xy_map" else ProjectionMode.DIRECT
        values = [float(value) for value in point]
        if len(values) < 3 or not np.isfinite(values[:3]).all():
            return None, ProjectionMode.XY_MAP if mode == "xy_map" else ProjectionMode.DIRECT
        return values[:3], ProjectionMode.XY_MAP if mode == "xy_map" else ProjectionMode.DIRECT

    def _parse_armor_observations(
        self,
        raw_detections,
        roi_width: int,
        roi_height: int,
    ) -> Tuple[ArmorObservation, ...]:
        observations = []
        for raw in raw_detections:
            if len(raw) >= 5:
                raw_class_id, box, armor_conf, digit_conf, digit_index = raw[:5]
            else:
                raw_class_id, box, combined = raw[:3]
                armor_conf = float(combined)
                digit_conf = float(combined)
                digit_index = int(raw_class_id) % 5
            raw_class_id = int(raw_class_id)
            color = self._armor_color(raw_class_id)
            clipped = self._clip_box(box, roi_width, roi_height)
            if color is None or clipped is None:
                continue
            combined_confidence = self.digit_weight * float(digit_conf) + self.armor_weight * float(armor_conf)
            observations.append(
                ArmorObservation(
                    armor_box=tuple(clipped),
                    armor_confidence=float(armor_conf),
                    digit_index=int(digit_index),
                    digit_confidence=float(digit_conf),
                    color=color,
                    raw_class_id=raw_class_id,
                    combined_confidence=float(combined_confidence),
                )
            )
        return tuple(observations)

    def _select_armor(self, observations: Sequence[ArmorObservation]) -> Optional[ArmorObservation]:
        eligible = [
            obs
            for obs in observations
            if obs.digit_index != self.ignored_digit_index
            and obs.digit_confidence >= self.min_digit_confidence
            and obs.combined_confidence >= self.min_combined_confidence
        ]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda obs: (
                obs.combined_confidence,
                obs.digit_confidence,
                obs.armor_confidence,
                -obs.armor_box[0],
                -obs.armor_box[1],
            ),
        )

    def detect_cars(
        self, img: np.ndarray
    ) -> tuple[Tuple[CarStageItem, ...], dict[str, float]]:
        """执行车辆神经网络、BoT-SORT 和车辆 ROI 裁剪（流水线 A 阶段）。"""
        stage_timings_ms: dict[str, float] = {}
        height, width = img.shape[:2]
        if self.xy_map_manager is not None:
            self.xy_map_manager.set_frame_size(width, height)

        stage_start = time.perf_counter()
        car_detections, _ = self.car_detector.predict(img)
        duration = time.perf_counter() - stage_start
        module_name = "03 车辆检测与BoT-SORT（模块03总时间）"
        stage_timings_ms[module_name] = duration * 1000.0
        self._record_timing_stat(
            self.detector_timing_stats,
            module_name,
            duration,
        )
        for sub_name, sub_duration in getattr(self.car_detector, "last_timing_stats", {}).items():
            name = sub_name if str(sub_name).startswith("03") else f"03.{sub_name}"
            # Predictor 也返回了模块03合计；外层已记录完整调用墙钟，
            # 若再写一次会使 count/EMA 每帧更新两次。
            if name == module_name:
                continue
            stage_timings_ms[name] = float(sub_duration) * 1000.0
            self._record_timing_stat(self.detector_timing_stats, name, sub_duration)

        valid_cars: List[CarStageItem] = []
        clip_and_filter_s = 0.0
        roi_view_s = 0.0
        stage_start = time.perf_counter()
        for car_class_id, raw_box, car_conf, track_id in car_detections:
            atomic_started = time.perf_counter()
            car_box = self._clip_box(raw_box, width, height)
            if car_box is None or track_id is None:
                clip_and_filter_s += time.perf_counter() - atomic_started
                continue
            x1, y1, x2, y2 = [int(value) for value in car_box]
            clip_and_filter_s += time.perf_counter() - atomic_started
            atomic_started = time.perf_counter()
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                roi_view_s += time.perf_counter() - atomic_started
                continue
            crop_view = crop.view()
            crop_view.flags.writeable = False
            valid_cars.append(
                CarStageItem(
                    car_class_id=int(car_class_id),
                    car_box=tuple(float(value) for value in car_box),
                    car_conf=float(car_conf),
                    track_id=int(track_id),
                    vehicle_roi=crop_view,
                )
            )
            roi_view_s += time.perf_counter() - atomic_started
        duration = time.perf_counter() - stage_start
        name = "04 车辆ROI生成（模块04总时间）"
        stage_timings_ms[name] = duration * 1000.0
        self._record_timing_stat(self.detector_timing_stats, name, duration)
        for atomic_name, atomic_duration in (
            ("04.1 车辆框裁边与无效框过滤（原子模块04.1）", clip_and_filter_s),
            ("04.2 车辆ROI切片与只读视图封装（原子模块04.2）", roi_view_s),
        ):
            stage_timings_ms[atomic_name] = atomic_duration * 1000.0
            self._record_timing_stat(
                self.detector_timing_stats, atomic_name, atomic_duration
            )
        return tuple(valid_cars), stage_timings_ms

    def build_vehicle_only_detections(
        self,
        cars: Sequence[CarStageItem],
        observation_monotonic: float,
    ) -> List[SingleDetectionResult]:
        """装甲模型异常时仍保留车辆和 BoT-SORT ID。"""
        return [
            SingleDetectionResult(
                class_id=-1,
                class_conf=0.0,
                armor_box=[0.0, 0.0, 0.0, 0.0],
                car_box=list(car.car_box),
                car_conf=float(car.car_conf),
                pos_3d=None,
                time_stamp=float(observation_monotonic),
                bot_id=int(car.track_id),
            )
            for car in cars
        ]

    def detect_armors(
        self,
        img: np.ndarray,
        cars: Sequence[CarStageItem],
        observation_monotonic: float,
    ) -> tuple[List[SingleDetectionResult], Optional[np.ndarray], dict[str, float]]:
        """执行装甲、数字识别和投影（流水线 B 阶段检测部分）。"""
        detections: List[SingleDetectionResult] = []
        stage_timings_ms: dict[str, float] = {}
        frame_size = (int(img.shape[1]), int(img.shape[0]))
        if not cars:
            return detections, None, stage_timings_ms

        crop_imgs = [car.vehicle_roi for car in cars]
        stage_start = time.perf_counter()
        armor_batches, _ = self.armor_detector.predict_batch(crop_imgs)
        duration = time.perf_counter() - stage_start
        name = "06-08 装甲板与数字神经网络识别（模块6至8总时间）"
        stage_timings_ms[name] = duration * 1000.0
        self._record_timing_stat(self.detector_timing_stats, name, duration)
        armor_internal_timings = dict(
            getattr(self.armor_detector, "last_timing_stats", {})
        )
        has_numbered_armor_total = any(
            str(timing_name).startswith("06 ")
            for timing_name in armor_internal_timings
        )
        has_numbered_digit_total = any(
            str(timing_name).startswith("08 ")
            for timing_name in armor_internal_timings
        )
        for sub_name, sub_duration in armor_internal_timings.items():
            if str(sub_name)[:2].isdigit():
                sub_timing_name = str(sub_name)
            elif sub_name.startswith("装甲板位置"):
                if has_numbered_armor_total:
                    continue
                sub_timing_name = "06 装甲位置检测含结果回传（模块06总时间）"
            elif sub_name.startswith("数字神经网络"):
                if has_numbered_digit_total:
                    continue
                sub_timing_name = "08 数字神经网络识别含结果回传（模块08总时间）"
            elif sub_name.startswith("装甲与数字"):
                # 外层 duration 已精确覆盖同一段 predict_batch 调用。
                continue
            else:
                sub_timing_name = f"06-09 {sub_name}（模块06至09细分）"
            stage_timings_ms[sub_timing_name] = float(sub_duration) * 1000.0
            self._record_timing_stat(
                self.detector_timing_stats, sub_timing_name, sub_duration
            )

        stage_start = time.perf_counter()
        parse_select_s = 0.0
        projection_s = 0.0
        detection_pack_s = 0.0
        for raw_armors, car, crop in zip(armor_batches, cars, crop_imgs):
            car_box = car.car_box
            atomic_started = time.perf_counter()
            armor_observations = self._parse_armor_observations(raw_armors, crop.shape[1], crop.shape[0])
            selected = self._select_armor(armor_observations)
            parse_select_s += time.perf_counter() - atomic_started
            pos_3d = None
            projection_mode = ProjectionMode.DIRECT
            atomic_started = time.perf_counter()
            if selected is not None:
                pos_3d, projection_mode = self._project_car_box(car_box, frame_size)
            projection_s += time.perf_counter() - atomic_started
            atomic_started = time.perf_counter()
            detections.append(
                SingleDetectionResult(
                    class_id=-1 if selected is None else selected.raw_class_id,
                    class_conf=0.0 if selected is None else selected.combined_confidence,
                    armor_box=[0.0, 0.0, 0.0, 0.0] if selected is None else list(selected.armor_box),
                    car_box=list(car_box),
                    car_conf=float(car.car_conf),
                    pos_3d=pos_3d,
                    time_stamp=float(observation_monotonic),
                    bot_id=int(car.track_id),
                    armor_observations=armor_observations,
                    selected_armor=selected,
                    projection_mode=projection_mode,
                )
            )
            detection_pack_s += time.perf_counter() - atomic_started
        duration = time.perf_counter() - stage_start
        name = "09 检测结果语义选择与射线投影（模块09总时间）"
        stage_timings_ms[name] = duration * 1000.0
        self._record_timing_stat(self.detector_timing_stats, name, duration)
        for atomic_name, atomic_duration in (
            ("09.2 装甲候选解析与最佳装甲选择（原子模块09.2）", parse_select_s),
            ("09.3 车辆底边中心射线投影（原子模块09.3）", projection_s),
            ("09.4 SingleDetectionResult封装（原子模块09.4）", detection_pack_s),
        ):
            stage_timings_ms[atomic_name] = atomic_duration * 1000.0
            self._record_timing_stat(
                self.detector_timing_stats, atomic_name, atomic_duration
            )

        vis_img = None
        if self.visualize and self.separate_detection_visualization:
            stage_start = time.perf_counter()
            vis_img = self._get_visualized_img(img, detections)
            self._record_timing_stat(
                self.detector_timing_stats,
                "12 检测可视化（模块12下游发布）",
                time.perf_counter() - stage_start,
            )
        return detections, vis_img, stage_timings_ms

    def detect(self, img: np.ndarray) -> tuple[List[SingleDetectionResult], Optional[np.ndarray]]:
        """保留原串行接口，供禁用流水线和回归测试使用。"""
        detect_start = time.perf_counter()
        observation_monotonic = time.monotonic()
        cars, _ = self.detect_cars(img)
        detections, vis_img, _ = self.detect_armors(
            img, cars, observation_monotonic
        )
        self._record_timing_stat(
            self.detector_timing_stats,
            "03-09 串行检测阶段总时间（模块3至9总时间）",
            time.perf_counter() - detect_start,
        )
        return detections, vis_img

    def _get_visualized_img(self, img: np.ndarray, detections: Sequence[SingleDetectionResult]) -> np.ndarray:
        vis_img = img.copy()
        for detection in detections:
            x1, y1, x2, y2 = [int(round(value)) for value in detection.car_box]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), self.vis_box_thickness)
            cv2.putText(
                vis_img,
                f"Car: {detection.car_conf:.2f} ID: {detection.bot_id}",
                (x1, max(0, y1 - self.vis_label_offset_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.vis_label_font_scale,
                (0, 255, 0),
                self.vis_label_font_thickness,
            )
            selected = detection.selected_armor
            if selected is None:
                continue
            ax1 = int(round(x1 + selected.armor_box[0]))
            ay1 = int(round(y1 + selected.armor_box[1]))
            ax2 = int(round(x1 + selected.armor_box[2]))
            ay2 = int(round(y1 + selected.armor_box[3]))
            color = (0, 0, 255) if selected.color == ArmorColor.BLUE else (255, 0, 0) if selected.color == ArmorColor.RED else (255, 255, 255)
            cv2.rectangle(vis_img, (ax1, ay1), (ax2, ay2), color, self.vis_box_thickness)
            confidence_text = (
                f"armor:{selected.armor_confidence:.2f} "
                f"digit:{selected.digit_confidence:.2f} "
                f"combined:{selected.combined_confidence:.2f}"
            )
            cv2.putText(
                vis_img,
                confidence_text,
                (ax1, max(0, ay1 - self.vis_label_offset_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.vis_label_font_scale,
                color,
                self.vis_label_font_thickness,
            )
        return vis_img
