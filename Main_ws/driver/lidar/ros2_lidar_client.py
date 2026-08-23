# -*- coding: utf-8 -*-
"""
ROS2 lidar cluster subscriber for RM2026 Radar Algorithm System.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


@dataclass(frozen=True)
class LidarSnapshot:
    points_xy_m: Tuple[Tuple[float, float], ...]
    ros_stamp_sec: Optional[float]
    received_monotonic: Optional[float]


class LidarClusterSubscriber(Node):
    def __init__(self, topic: str, queue_size: int = 10, name: str = "rm2026_lidar_cluster_sub") -> None:
        super().__init__(name)
        self._lock = threading.Lock()
        self._points: List[Tuple[float, float]] = []
        self._xyz_points: List[Tuple[float, float, float]] = []
        self._stamp = None
        self._received_monotonic = None
        self._record_lock = threading.Lock()
        self._record_enabled = False
        self._record_dir = ""
        self._record_frame_index = 0
        self.create_subscription(PointCloud2, topic, self._callback, queue_size)

    def _callback(self, msg: PointCloud2) -> None:
        points = []
        xyz_points = []
        for x, y, _z in point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        ):
            points.append((float(x), float(y)))
            xyz_points.append((float(x), float(y), float(_z)))
        with self._lock:
            self._points = points
            self._xyz_points = xyz_points
            self._stamp = msg.header.stamp
            self._received_monotonic = time.monotonic()
        self._record_frame(msg.header.stamp, xyz_points)

    def get_points(self) -> List[Tuple[float, float]]:
        with self._lock:
            return list(self._points)
    
    def get_xyz_points(self) -> List[Tuple[float, float, float]]:
        with self._lock:
            return list(self._xyz_points)

    def get_snapshot(self) -> LidarSnapshot:
        """在同一把锁内冻结点列表与两个时间戳。"""
        with self._lock:
            stamp = self._stamp
            ros_stamp_sec = None
            if stamp is not None:
                ros_stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            return LidarSnapshot(
                points_xy_m=tuple(self._points),
                ros_stamp_sec=ros_stamp_sec,
                received_monotonic=self._received_monotonic,
            )

    def start_recording(self, save_root_dir: str) -> str:
        """开始录制雷达点云。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir = os.path.join(save_root_dir, f"lidar_{timestamp}")
        os.makedirs(record_dir, exist_ok=True)
        with self._record_lock:
            self._record_dir = record_dir
            self._record_frame_index = 0
            self._record_enabled = True
        self.get_logger().info(f"开始录制雷达点云: {record_dir}")
        return record_dir

    def stop_recording(self) -> None:
        """停止录制雷达点云。"""
        with self._record_lock:
            if not self._record_enabled:
                return
            self._record_enabled = False
            record_dir = self._record_dir
            frame_count = self._record_frame_index
            self._record_dir = ""
        self.get_logger().info(f"停止录制雷达点云: {record_dir}, 共 {frame_count} 帧")

    def _record_frame(self, stamp, xyz_points: List[Tuple[float, float, float]]) -> None:
        """将当前点云帧保存为 npz。"""
        with self._record_lock:
            if not self._record_enabled or not self._record_dir:
                return
            frame_index = self._record_frame_index
            self._record_frame_index += 1
            record_dir = self._record_dir

        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        stamp_text = f"{stamp.sec}_{stamp.nanosec:09d}"
        save_path = os.path.join(
            record_dir, f"lidar_{frame_index:06d}_{stamp_text}.npz"
        )
        np.savez_compressed(
            save_path,
            stamp=stamp_sec,
            points=np.asarray(xyz_points, dtype=np.float32),
        )


class LidarClient:
    def __init__(self, topic: str, queue_size: int = 10, name: str = "rm2026_lidar_client") -> None:
        self._node = LidarClusterSubscriber(topic=topic, queue_size=queue_size, name=name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._executor.shutdown()
        self._node.destroy_node()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_points(self) -> List[Tuple[float, float]]:
        return self._node.get_points()
    
    def get_xyz_points(self) -> List[Tuple[float, float, float]]:
        return self._node.get_xyz_points()

    def get_snapshot(self) -> LidarSnapshot:
        return self._node.get_snapshot()

    def start_recording(self, save_root_dir: str) -> str:
        return self._node.start_recording(save_root_dir)

    def stop_recording(self) -> None:
        self._node.stop_recording()
