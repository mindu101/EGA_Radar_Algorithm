# -*- coding: utf-8 -*-
"""只供 UI 使用的异步检测/追踪叠图工具。"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import cv2
import numpy as np


logger = logging.getLogger(__name__)


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionVisualSnapshot:
    """检测结果的不可变轻量快照，避免渲染线程读取正在变化的对象。"""

    car_box: Box
    car_conf: float
    bot_id: int
    class_id: int
    armor_box: Optional[Box]
    armor_confidence: Optional[float]
    digit_confidence: Optional[float]
    combined_confidence: Optional[float]


@dataclass(frozen=True)
class TrackVisualSnapshot:
    """追踪结果的不可变轻量快照。"""

    class_id: int
    label: str
    car_box: Box


@dataclass(frozen=True)
class VisualizationRequest:
    """一次异步叠图请求。frame 只保留引用，不复制全分辨率图像。"""

    frame: np.ndarray
    detections: tuple[DetectionVisualSnapshot, ...]
    tracks: tuple[TrackVisualSnapshot, ...]
    vehicle_boxes: tuple[tuple[float, float, float, float, float, int], ...]
    raw_yolo_boxes: tuple[tuple[float, float, float, float, float], ...]
    source_stamp: Optional[float]


def _snapshot_box(box) -> Optional[Box]:
    """将任意框转换为固定四元组；非法框不进入后台线程。"""
    if box is None:
        return None
    try:
        values = tuple(float(value) for value in box[:4])
    except (TypeError, ValueError, IndexError):
        return None
    if len(values) != 4 or not all(np.isfinite(value) for value in values):
        return None
    return values


def snapshot_detections(detections: Sequence) -> tuple[DetectionVisualSnapshot, ...]:
    """在定位线程中复制少量标量，不复制图像或 Tensor。"""
    snapshots = []
    for detection in detections:
        car_box = _snapshot_box(getattr(detection, "car_box", None))
        if car_box is None:
            continue
        class_id = int(getattr(detection, "class_id", -1))
        armor_box = None
        armor_confidence = None
        digit_confidence = None
        combined_confidence = None
        if class_id >= 0:
            armor_box = _snapshot_box(getattr(detection, "armor_box", None))
            selected_armor = getattr(detection, "selected_armor", None)
            if selected_armor is not None:
                armor_confidence = float(
                    getattr(selected_armor, "armor_confidence", 0.0)
                )
                digit_confidence = float(
                    getattr(selected_armor, "digit_confidence", 0.0)
                )
                combined_confidence = float(
                    getattr(selected_armor, "combined_confidence", 0.0)
                )
            elif hasattr(detection, "class_conf"):
                # 兼容只提供旧字段的检测结果；旧结构只能恢复 combined。
                combined_confidence = float(getattr(detection, "class_conf", 0.0))
        snapshots.append(
            DetectionVisualSnapshot(
                car_box=car_box,
                car_conf=float(getattr(detection, "car_conf", 0.0)),
                bot_id=int(getattr(detection, "bot_id", -1)),
                class_id=class_id,
                armor_box=armor_box,
                armor_confidence=armor_confidence,
                digit_confidence=digit_confidence,
                combined_confidence=combined_confidence,
            )
        )
    return tuple(snapshots)


def snapshot_tracks(tracks: Sequence) -> tuple[TrackVisualSnapshot, ...]:
    """只提交当前有效且有框的轨迹。"""
    snapshots = []
    for track in tracks:
        is_active = bool(
            getattr(track, "output_valid", getattr(track, "is_active", False))
        )
        if not is_active:
            continue
        car_box = _snapshot_box(
            getattr(track, "last_car_box", getattr(track, "car_box", None))
        )
        if car_box is None:
            continue
        name = str(getattr(track, "name", ""))
        identity_state = getattr(track, "identity_state", None)
        frame_source = getattr(track, "frame_source", None)
        if identity_state is not None and frame_source is not None:
            identity_text = str(getattr(identity_state, "value", identity_state))
            source_text = str(getattr(frame_source, "value", frame_source))
            label = f"{name} {identity_text}/{source_text}"
        else:
            state = getattr(track, "state", "")
            state_text = str(getattr(state, "value", state))
            bot_id = int(getattr(track, "bot_id", -1))
            label = f"{name} BOT ID:{bot_id} State:{state_text}"
        snapshots.append(
            TrackVisualSnapshot(
                class_id=int(getattr(track, "class_id", -1)),
                label=label,
                car_box=car_box,
            )
        )
    return tuple(snapshots)


def snapshot_vehicle_boxes(boxes: Sequence) -> tuple[tuple, ...]:
    """复制 A 阶段车辆框，供 UI 在装甲识别为空时仍绘制 YOLO 结果。"""
    snapshots = []
    for item in boxes:
        if item is None or len(item) < 6:
            continue
        box = _snapshot_box(item[:4])
        try:
            confidence = float(item[4])
            track_id = int(item[5])
        except (TypeError, ValueError, IndexError):
            continue
        if box is not None and np.isfinite(confidence):
            snapshots.append((*box, confidence, track_id))
    return tuple(snapshots)


def snapshot_raw_yolo_boxes(boxes: Sequence) -> tuple[tuple, ...]:
    """Copy post-NMS detector boxes before tracker filtering for UI fallback."""
    snapshots = []
    for item in boxes:
        if item is None or len(item) < 5:
            continue
        box = _snapshot_box(item[:4])
        try:
            confidence = float(item[4])
        except (TypeError, ValueError, IndexError):
            continue
        if box is not None and np.isfinite(confidence):
            snapshots.append((*box, confidence))
    return tuple(snapshots)


class AsyncTrackingVisualizer:
    """单槽位后台渲染器：始终只处理最新一帧，不让 UI 任务反压定位。"""

    def __init__(
        self,
        config: dict,
        on_image_ready: Optional[
            Callable[[np.ndarray, Optional[float], int], None]
        ] = None,
    ):
        visual_cfg = config.get("visualization", {})
        debug_cfg = config.get("debug", {})

        self.enabled = bool(visual_cfg.get("enabled", True))
        self.update_hz = max(0.1, float(visual_cfg.get("update_hz", 10.0)))
        self.interval_sec = 1.0 / self.update_hz
        self.max_width = max(320, int(visual_cfg.get("output_max_width", 1280)))
        self.max_height = max(240, int(visual_cfg.get("output_max_height", 960)))
        self.box_thickness = max(1, int(visual_cfg.get("box_thickness", 2)))
        self.label_font_scale = max(
            0.1, float(visual_cfg.get("label_font_scale", 0.5))
        )
        self.label_font_thickness = max(
            1, int(visual_cfg.get("label_font_thickness", 2))
        )
        self.label_offset_y = max(0, int(visual_cfg.get("label_offset_y", 10)))
        self.timing_alpha = min(
            1.0, max(0.0, float(debug_cfg.get("timing_ema_alpha", 0.2)))
        )
        self.timing_debug_enabled = bool(
            debug_cfg.get("timing_debug_enabled", True)
        )

        self._on_image_ready = on_image_ready
        self._condition = threading.Condition()
        self._stop_requested = False
        self._closed = False
        self._latest_request: Optional[VisualizationRequest] = None
        self._latest_image: Optional[np.ndarray] = None
        self._latest_source_stamp: Optional[float] = None
        self._latest_sequence = 0
        self._submitted_count = 0
        self._rendered_count = 0
        self._overwritten_count = 0
        self._error_count = 0
        self._timing_stats = {}
        self._frame_timing_ms = {}
        self._thread: Optional[threading.Thread] = None

        if self.enabled:
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="tracking-visualization-worker",
                daemon=True,
            )
            self._thread.start()

    def submit(
        self,
        frame,
        detections: Sequence,
        tracks: Sequence,
        source_stamp=None,
        vehicle_boxes: Sequence = (),
        raw_yolo_boxes: Sequence = (),
    ):
        """非阻塞提交；已有待渲染任务时直接以新任务覆盖。"""
        if not self.enabled or self._closed or frame is None:
            return False
        if not isinstance(frame, np.ndarray) or frame.ndim < 2 or frame.size == 0:
            return False

        # view() 为 O(1)，只把后台线程持有的这个视图设为只读，不修改相机原数组标志。
        frame_view = frame.view()
        frame_view.flags.writeable = False
        request = VisualizationRequest(
            frame=frame_view,
            detections=snapshot_detections(detections),
            tracks=snapshot_tracks(tracks),
            vehicle_boxes=snapshot_vehicle_boxes(vehicle_boxes),
            raw_yolo_boxes=snapshot_raw_yolo_boxes(raw_yolo_boxes),
            source_stamp=self._normalize_stamp(source_stamp),
        )

        with self._condition:
            if self._stop_requested:
                return False
            if self._latest_request is not None:
                self._overwritten_count += 1
            self._latest_request = request
            self._submitted_count += 1
            self._condition.notify()
        return True

    @staticmethod
    def _normalize_stamp(source_stamp) -> Optional[float]:
        try:
            value = float(source_stamp)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def _worker_loop(self):
        next_render_time = 0.0
        while True:
            with self._condition:
                while self._latest_request is None and not self._stop_requested:
                    self._condition.wait()
                if self._stop_requested:
                    return

                now = time.monotonic()
                wait_sec = next_render_time - now
                if wait_sec > 0.0:
                    self._condition.wait(timeout=wait_sec)
                    continue

                request = self._latest_request
                self._latest_request = None

            render_started_monotonic = time.monotonic()
            render_start = time.perf_counter()
            try:
                image = self._render(request)
            except Exception:
                with self._condition:
                    self._error_count += 1
                logger.exception("异步检测/追踪叠图失败")
                next_render_time = render_started_monotonic + self.interval_sec
                continue

            render_ms = (time.perf_counter() - render_start) * 1000.0
            image.flags.writeable = False
            with self._condition:
                self._latest_image = image
                self._latest_source_stamp = request.source_stamp
                self._latest_sequence += 1
                self._rendered_count += 1
                sequence = self._latest_sequence
                self._record_timing_locked("后台合并可视化", render_ms)

            # 回调只能发布成品 ndarray 引用，不能在后台线程中调用任何 Qt API。
            if self._on_image_ready is not None:
                try:
                    self._on_image_ready(image, request.source_stamp, sequence)
                except Exception:
                    logger.exception("异步叠图成品发布失败")

            next_render_time = render_started_monotonic + self.interval_sec

    def _render(self, request: VisualizationRequest) -> np.ndarray:
        source_height, source_width = request.frame.shape[:2]
        scale = min(
            1.0,
            self.max_width / float(source_width),
            self.max_height / float(source_height),
        )
        if scale < 1.0:
            width = max(1, int(round(source_width * scale)))
            height = max(1, int(round(source_height * scale)))
            # UI 叠图重视吞吐量，线性缩放比高质量面积缩放更适合实时显示。
            vis_img = cv2.resize(
                request.frame, (width, height), interpolation=cv2.INTER_LINEAR
            )
        else:
            vis_img = request.frame.copy()

        def scaled_box(box: Box):
            return tuple(int(round(value * scale)) for value in box)

        label_font_scale = max(0.35, self.label_font_scale * scale)
        label_offset_y = max(4, int(round(self.label_offset_y * scale)))
        box_thickness = max(1, int(round(self.box_thickness * scale)))
        label_thickness = max(
            1, int(round(self.label_font_thickness * scale))
        )

        for track in request.tracks:
            color = (0, 0, 255) if track.class_id < 5 else (255, 0, 0)
            x1, y1, x2, y2 = scaled_box(track.car_box)
            cv2.rectangle(
                vis_img,
                (x1, y1),
                (x2, y2),
                color,
                box_thickness,
            )
            cv2.putText(
                vis_img,
                track.label,
                (x1, max(label_offset_y, y1 - label_offset_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                color,
                label_thickness,
            )

        for detection in request.detections:
            x1_car, y1_car, x2_car, y2_car = scaled_box(detection.car_box)
            cv2.rectangle(
                vis_img,
                (x1_car, y1_car),
                (x2_car, y2_car),
                (0, 255, 0),
                box_thickness,
            )
            cv2.putText(
                vis_img,
                f"Car {detection.car_conf:.2f} ID:{detection.bot_id}",
                (x1_car, max(label_offset_y, y1_car - label_offset_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                (0, 255, 0),
                label_thickness,
            )

            if detection.class_id >= 0 and detection.armor_box is not None:
                armor_box = (
                    detection.car_box[0] + detection.armor_box[0],
                    detection.car_box[1] + detection.armor_box[1],
                    detection.car_box[0] + detection.armor_box[2],
                    detection.car_box[1] + detection.armor_box[3],
                )
                x1_armor, y1_armor, x2_armor, y2_armor = scaled_box(armor_box)
                armor_color = (
                    (0, 0, 255)
                    if detection.class_id < 5
                    else (255, 0, 0)
                    if detection.class_id < 10
                    else (255, 255, 255)
                )
                cv2.rectangle(
                    vis_img,
                    (x1_armor, y1_armor),
                    (x2_armor, y2_armor),
                    armor_color,
                    box_thickness,
                )
                confidence_parts = []
                if detection.armor_confidence is not None:
                    confidence_parts.append(
                        f"armor:{detection.armor_confidence:.2f}"
                    )
                if detection.digit_confidence is not None:
                    confidence_parts.append(
                        f"digit:{detection.digit_confidence:.2f}"
                    )
                if detection.combined_confidence is not None:
                    confidence_parts.append(
                        f"combined:{detection.combined_confidence:.2f}"
                    )
                if confidence_parts:
                    cv2.putText(
                        vis_img,
                        " ".join(confidence_parts),
                        (
                            x1_armor,
                            max(label_offset_y, y1_armor - label_offset_y),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        label_font_scale,
                        armor_color,
                        label_thickness,
                    )

        for x1_raw, y1_raw, x2_raw, y2_raw, confidence, track_id in request.vehicle_boxes:
            x1, y1, x2, y2 = scaled_box((x1_raw, y1_raw, x2_raw, y2_raw))
            cv2.rectangle(
                vis_img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                box_thickness,
            )
            cv2.putText(
                vis_img,
                f"YOLO {confidence:.2f} ID:{track_id}",
                (x1, max(label_offset_y, y1 - label_offset_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                (0, 255, 0),
                label_thickness,
            )

        if not request.vehicle_boxes:
            for x1_raw, y1_raw, x2_raw, y2_raw, confidence in request.raw_yolo_boxes:
                x1, y1, x2, y2 = scaled_box((x1_raw, y1_raw, x2_raw, y2_raw))
                cv2.rectangle(
                    vis_img, (x1, y1), (x2, y2), (0, 255, 0), box_thickness
                )
                cv2.putText(
                    vis_img,
                    f"YOLO {confidence:.2f}",
                    (x1, max(label_offset_y, y1 - label_offset_y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    label_font_scale,
                    (0, 255, 0),
                    label_thickness,
                )

        return vis_img

    def _record_timing_locked(self, name: str, duration_ms: float):
        self._frame_timing_ms = {name: float(duration_ms)}
        if not self.timing_debug_enabled:
            return
        stat = self._timing_stats.get(name)
        if stat is None:
            self._timing_stats[name] = {
                "last_ms": float(duration_ms),
                "avg_ms": float(duration_ms),
                "max_ms": float(duration_ms),
                "count": 1,
            }
            return
        stat["last_ms"] = float(duration_ms)
        stat["avg_ms"] = (
            (1.0 - self.timing_alpha) * stat["avg_ms"]
            + self.timing_alpha * float(duration_ms)
        )
        stat["max_ms"] = max(stat["max_ms"], float(duration_ms))
        stat["count"] += 1

    def get_latest(self):
        """返回最新成品引用、源帧时间戳和序号，不复制图像。"""
        with self._condition:
            return (
                self._latest_image,
                self._latest_source_stamp,
                self._latest_sequence,
            )

    def get_timing_stats(self) -> dict:
        with self._condition:
            return {name: dict(stat) for name, stat in self._timing_stats.items()}

    def get_frame_timing_ms(self) -> dict:
        with self._condition:
            return dict(self._frame_timing_ms)

    def clear_timing_stats(self):
        """清空显示用累计耗时，不影响正在等待或渲染的任务。"""
        with self._condition:
            self._timing_stats.clear()
            self._frame_timing_ms.clear()

    def get_diagnostics(self) -> dict:
        with self._condition:
            return {
                "enabled": self.enabled,
                "submitted": self._submitted_count,
                "rendered": self._rendered_count,
                "overwritten": self._overwritten_count,
                "errors": self._error_count,
                "latest_sequence": self._latest_sequence,
                "latest_source_stamp": self._latest_source_stamp,
                "pending": self._latest_request is not None,
            }

    def close(self, timeout_sec: float = 2.0):
        """停止后台线程；可重复调用。"""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._stop_requested = True
            self._latest_request = None
            self._condition.notify_all()

        if self._thread is not None:
            self._thread.join(timeout=max(0.0, float(timeout_sec)))
            if self._thread.is_alive():
                logger.warning("异步检测/追踪叠图线程未在超时前退出")
