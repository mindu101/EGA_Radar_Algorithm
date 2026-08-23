# -*- coding: utf-8 -*-
"""按录制时间戳并以 ROS ``/clock`` 为唯一时钟回放相机录像。"""

from __future__ import annotations

import bisect
import csv
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosgraph_msgs.msg import Clock


@dataclass(frozen=True)
class _FrameRecord:
    frame_index: int
    timestamp: float


class _ReplayBuffer:
    """为 tracker/UI 等消费组保存同一份只读最新帧。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._image = None
        self._stamp = None
        self._sequence = 0
        self._group_last_sequence: dict[str, int] = {}

    def register_group(self, group_id: str) -> None:
        with self._lock:
            self._group_last_sequence.setdefault(group_id, -1)

    def update(self, image: np.ndarray, stamp: float) -> None:
        with self._lock:
            self._image = image
            self._stamp = float(stamp)
            self._sequence += 1
            self._cond.notify_all()

    def wake_all(self) -> None:
        with self._lock:
            self._cond.notify_all()

    def get_latest(
        self, group_id: str, timeout: float
    ) -> Tuple[np.ndarray | None, float | None]:
        timeout = max(0.0, float(timeout))
        with self._lock:
            if group_id not in self._group_last_sequence:
                raise ValueError(f"Group {group_id} not registered")
            deadline = time.monotonic() + timeout
            while True:
                if (
                    self._image is not None
                    and self._group_last_sequence[group_id] != self._sequence
                ):
                    self._group_last_sequence[group_id] = self._sequence
                    return self._image, self._stamp
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None, None
                self._cond.wait(remaining)


class _ReplayClockNode(Node):
    def __init__(self, owner: "TimestampReplayCamera") -> None:
        super().__init__("rm2026_timestamp_camera_replay")
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE
        qos.history = HistoryPolicy.KEEP_LAST
        self.create_subscription(Clock, "/clock", owner._on_clock, qos)


class TimestampReplayCamera:
    """复用真实相机接口，按 ``/clock`` 到期发布 CSV 对应视频帧。"""

    _instance = None

    def __init__(self, test_config_path: str) -> None:
        if self.__class__._instance is not None:
            raise RuntimeError("TimestampReplayCamera instance already exists")
        if not rclpy.ok():
            raise RuntimeError("创建同步回放相机前必须先调用 rclpy.init()")

        config_path = Path(test_config_path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as file:
            root_config = yaml.safe_load(file) or {}
        replay_cfg = root_config.get("replay", {})
        resolved_cfg = root_config.get("_resolved", {})

        camera_dir_text = resolved_cfg.get(
            "camera_dir", replay_cfg.get("camera_session_dir", "")
        )
        self.camera_dir = Path(str(camera_dir_text)).expanduser().resolve()
        self.video_path = Path(
            resolved_cfg.get("camera_video", self.camera_dir / "camera_record.mp4")
        ).expanduser().resolve()
        self.csv_path = Path(
            resolved_cfg.get(
                "camera_timestamps", self.camera_dir / "camera_timestamps.csv"
            )
        ).expanduser().resolve()
        self.time_basis = str(replay_cfg.get("time_basis", "sensor_stamp")).lower()
        if self.time_basis not in ("sensor_stamp", "arrival_stamp"):
            raise ValueError("replay.time_basis 只能为 sensor_stamp 或 arrival_stamp")
        self.overload_policy = str(
            replay_cfg.get("overload_policy", "realtime_drop")
        ).lower()
        if self.overload_policy != "realtime_drop":
            raise ValueError("当前仅支持 overload_policy=realtime_drop")

        self.valid_start = float(resolved_cfg.get("valid_start_timestamp", -float("inf")))
        self.valid_end = float(resolved_cfg.get("valid_end_timestamp", float("inf")))
        self._records = self._load_records()
        if not self._records:
            raise RuntimeError("相机时间戳 CSV 在有效回放区间内没有帧")
        self._timestamps = [record.timestamp for record in self._records]

        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开相机录像: {self.video_path}")
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._capture_next_index = 0
        self._preview_record = self._records[0]
        self._preview_frame = self._read_frame(self._preview_record.frame_index)
        if self._preview_frame is None:
            self.capture.release()
            raise RuntimeError(
                f"无法读取回放首帧: video_frame={self._preview_record.frame_index}"
            )

        self._buffer = _ReplayBuffer()
        self._clock_cond = threading.Condition()
        self._clock_sec = None
        self._last_clock_wall = None
        self._last_publish_wall = None
        self._last_publish_stamp = None
        self._fps = 0.0
        self._cursor = 0
        self._published = 0
        self._dropped = 0
        self._running = False
        self._stop_event = threading.Event()

        self._node = _ReplayClockNode(self)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._clock_thread = threading.Thread(
            target=self._executor.spin,
            name="camera-replay-clock",
            daemon=True,
        )
        self._replay_thread = threading.Thread(
            target=self._replay_loop,
            name="camera-timestamp-replay",
            daemon=True,
        )
        self.__class__._instance = self

        print(
            "[同步回放] 相机已装载: "
            f"{self.video_path}, 帧数={len(self._records)}, "
            f"分辨率={self.width}x{self.height}, "
            f"时间范围={self._timestamps[0]:.6f}~{self._timestamps[-1]:.6f}, "
            f"基准={self.time_basis}"
        )

    def _load_records(self) -> list[_FrameRecord]:
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"相机时间戳文件不存在: {self.csv_path}")
        timestamp_column = (
            "camera_stamp" if self.time_basis == "sensor_stamp" else "wall_time_s"
        )
        records = []
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or [])
            required = {"frame_index", timestamp_column}
            if not required.issubset(fieldnames):
                raise ValueError(
                    f"相机 CSV 缺少字段 {sorted(required - fieldnames)}: {self.csv_path}"
                )
            previous_stamp = -float("inf")
            for row_number, row in enumerate(reader, start=2):
                raw_stamp = str(row.get(timestamp_column, "")).strip()
                if not raw_stamp:
                    if timestamp_column == "camera_stamp":
                        raw_stamp = str(row.get("wall_time_s", "")).strip()
                    if not raw_stamp:
                        continue
                frame_index = int(row["frame_index"])
                stamp = float(raw_stamp)
                if stamp < previous_stamp:
                    raise ValueError(
                        f"相机时间戳非单调递增，CSV 第 {row_number} 行: {stamp}"
                    )
                previous_stamp = stamp
                if self.valid_start <= stamp <= self.valid_end:
                    records.append(_FrameRecord(frame_index=frame_index, timestamp=stamp))
        return records

    def _on_clock(self, msg: Clock) -> None:
        stamp = float(msg.clock.sec) + float(msg.clock.nanosec) * 1e-9
        with self._clock_cond:
            self._clock_sec = stamp
            self._last_clock_wall = time.monotonic()
            self._clock_cond.notify_all()

    def _read_frame(self, frame_index: int) -> np.ndarray | None:
        if frame_index < self._capture_next_index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            self._capture_next_index = frame_index
        while self._capture_next_index < frame_index:
            if not self.capture.grab():
                return None
            self._capture_next_index += 1
        ok, frame = self.capture.read()
        if not ok:
            return None
        self._capture_next_index += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.ascontiguousarray(frame)
        frame.flags.writeable = False
        return frame

    def _replay_loop(self) -> None:
        while not self._stop_event.is_set() and self._cursor < len(self._records):
            with self._clock_cond:
                self._clock_cond.wait_for(
                    lambda: self._stop_event.is_set()
                    or (
                        self._clock_sec is not None
                        and self._timestamps[self._cursor] <= self._clock_sec
                    ),
                    timeout=0.2,
                )
                if self._stop_event.is_set():
                    break
                clock_sec = self._clock_sec
            if clock_sec is None:
                continue

            due_end = bisect.bisect_right(self._timestamps, clock_sec) - 1
            if due_end < self._cursor:
                continue
            if due_end > self._cursor:
                self._dropped += due_end - self._cursor
            record = self._records[due_end]
            self._cursor = due_end + 1
            frame = self._read_frame(record.frame_index)
            if frame is None:
                print(
                    f"[同步回放] 无法读取视频第 {record.frame_index} 帧，停止相机回放。"
                )
                break

            now = time.monotonic()
            if self._last_publish_wall is not None:
                dt = now - self._last_publish_wall
                if dt > 1e-6:
                    instant_fps = 1.0 / dt
                    self._fps = instant_fps if self._fps <= 0 else 0.8 * self._fps + 0.2 * instant_fps
            self._last_publish_wall = now
            self._last_publish_stamp = record.timestamp
            self._published += 1
            self._buffer.update(frame, record.timestamp)

        if self._cursor >= len(self._records):
            print(
                "[同步回放] 相机回放结束: "
                f"发布={self._published}, 实时覆盖丢帧={self._dropped}"
            )
        self._buffer.wake_all()

    def register_group(self, group_id: str) -> None:
        self._buffer.register_group(group_id)

    def get_image_latest(
        self, group_id: str = "", timeout: float = 1.0
    ) -> Tuple[np.ndarray | None, float | None]:
        # rosbag 尚未启动时，UI 持续取得首个有效视频帧用于手动标点。
        # tracker 不能消费该预览帧，避免点击“启动雷达站”后在 /clock 到来前
        # 重复处理同一画面并污染轨迹状态。
        if (
            group_id == "radar_station"
            and self._published == 0
            and self._preview_frame is not None
        ):
            return self._preview_frame, self._preview_record.timestamp
        return self._buffer.get_latest(group_id, timeout)

    def start_streaming(self) -> None:
        if self._running:
            return
        self._clock_thread.start()
        self._replay_thread.start()
        self._running = True

    def stop_streaming(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        with self._clock_cond:
            self._clock_cond.notify_all()
        self._buffer.wake_all()
        self._executor.shutdown()
        self._node.destroy_node()
        self._clock_thread.join(timeout=1.0)
        self._replay_thread.join(timeout=2.0)
        self.capture.release()
        self._running = False

    def start_saving_threads(self) -> None:
        return

    def stop_saving_images(self) -> None:
        return

    def close(self) -> None:
        self.stop_streaming()
        self.__class__._instance = None

    def get_fps(self) -> float:
        return self._fps

    def get_time(self) -> float:
        return float(self._clock_sec or 0.0)

    def is_connected(self) -> bool:
        if self._last_clock_wall is None:
            return False
        return (time.monotonic() - self._last_clock_wall) < 1.0

    def set_exposure(self, _value: int) -> bool:
        return False

    def set_gain(self, _value: int) -> bool:
        return False

    def get_diagnostics(self) -> dict:
        return {
            "published": self._published,
            "dropped": self._dropped,
            "cursor": self._cursor,
            "total": len(self._records),
            "clock": self._clock_sec,
            "last_frame_stamp": self._last_publish_stamp,
        }
