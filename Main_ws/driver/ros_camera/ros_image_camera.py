# -*- coding: utf-8 -*-
"""
ROS2 Image subscriber camera for RM2026 Radar Algorithm System.
"""

from __future__ import annotations

import threading
import time
from typing import Tuple

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)
from sensor_msgs.msg import Image


class _ImageBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._image = None
        self._stamp = -1.0
        self._group_last_stamp = {}

    def register_group(self, group_id: str) -> None:
        with self._lock:
            if group_id not in self._group_last_stamp:
                self._group_last_stamp[group_id] = -1.0

    def update(self, image: np.ndarray, stamp: float) -> None:
        with self._lock:
            self._image = image
            self._stamp = stamp
            self._cond.notify_all()

    def get_latest(
        self, group_id: str, timeout: float
    ) -> Tuple[np.ndarray | None, float | None]:
        with self._lock:
            if group_id not in self._group_last_stamp:
                raise ValueError(f"Group {group_id} not registered")
            start_time = time.time()
            while True:
                if (
                    self._image is not None
                    and self._group_last_stamp[group_id] != self._stamp
                ):
                    self._group_last_stamp[group_id] = self._stamp
                    return self._image, self._stamp

                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return None, None

                self._cond.wait(timeout - elapsed)


class _RosImageSubscriber(Node):
    def __init__(self, topic: str, queue_size: int, buffer: _ImageBuffer) -> None:
        super().__init__("rm2026_ros_image_sub")
        self._buffer = buffer
        self._fps = 0.0
        self._last_frame_time = None
        self._last_msg_time = None

        qos_profile = QoSProfile(depth=queue_size)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_profile.durability = DurabilityPolicy.VOLATILE
        qos_profile.history = HistoryPolicy.KEEP_LAST
        self.create_subscription(Image, topic, self._callback, qos_profile)

    def _callback(self, msg: Image) -> None:
        image = self._decode_image(msg)
        if image is None:
            return

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = time.time()
        self._buffer.update(image, stamp)

        now = time.time()
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 1e-6:
                self._fps = 0.8 * self._fps + 0.2 / dt
        self._last_frame_time = now
        self._last_msg_time = now

    def _decode_image(self, msg: Image) -> np.ndarray | None:
        encoding = msg.encoding.lower()
        if encoding in ("rgb8", "bgr8"):
            channels = 3
        elif encoding in ("rgba8", "bgra8"):
            channels = 4
        elif encoding in ("mono8", "8uc1"):
            channels = 1
        else:
            self.get_logger().warning(
                "Unsupported image encoding: %s", msg.encoding
            )
            return None

        expected = msg.height * msg.step
        data = np.frombuffer(msg.data, dtype=np.uint8)
        if data.size < expected:
            self.get_logger().warning(
                "Image data size mismatch: %d < %d", data.size, expected
            )
            return None

        data = data[:expected].reshape((msg.height, msg.step))
        if channels == 1:
            image = data[:, : msg.width].reshape((msg.height, msg.width))
            image = np.repeat(image[:, :, None], 3, axis=2)
        else:
            row_stride = msg.width * channels
            image = data[:, :row_stride].reshape(
                (msg.height, msg.width, channels)
            )
            if encoding == "bgr8":
                image = image[:, :, ::-1]
            elif encoding == "bgra8":
                image = image[:, :, :3][:, :, ::-1]
            elif encoding == "rgba8":
                image = image[:, :, :3]

        image = np.ascontiguousarray(image)
        image.flags.writeable = False
        return image

    def get_fps(self) -> float:
        return self._fps

    def is_connected(self) -> bool:
        if self._last_msg_time is None:
            return False
        return (time.time() - self._last_msg_time) < 1.0


class RosImageCamera:
    _instance = None

    def __init__(self, topic: str = "/camera_image", queue_size: int = 5) -> None:
        if RosImageCamera._instance is not None:
            raise RuntimeError("RosImageCamera instance already exists")
        if not rclpy.ok():
            rclpy.init()

        self._buffer = _ImageBuffer()
        self._node = _RosImageSubscriber(topic, queue_size, self._buffer)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._running = False
        RosImageCamera._instance = self

    def register_group(self, group_id: str) -> None:
        self._buffer.register_group(group_id)

    def get_image_latest(
        self, group_id: str = "", timeout: float = 1.0
    ) -> Tuple[np.ndarray | None, float | None]:
        return self._buffer.get_latest(group_id, timeout)

    def start_streaming(self) -> None:
        if not self._running:
            self._thread.start()
            self._running = True

    def stop_streaming(self) -> None:
        if not self._running:
            return
        self._executor.shutdown()
        self._node.destroy_node()
        self._thread.join(timeout=1.0)
        self._running = False

    def start_saving_threads(self) -> None:
        return

    def stop_saving_images(self) -> None:
        return

    def close(self) -> None:
        self.stop_streaming()

    def get_fps(self) -> float:
        return self._node.get_fps()

    def is_connected(self) -> bool:
        return self._node.is_connected()

    def set_exposure(self, _value: int) -> bool:
        return False

    def set_gain(self, _value: int) -> bool:
        return False
