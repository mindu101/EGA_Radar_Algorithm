from __future__ import annotations

import math
import threading
from collections import Counter
from typing import Sequence

import numpy as np


class _OnlineResidualStatistics:
    """二维残差的 Welford 在线统计，不保存历史样本。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = np.zeros(2, dtype=np.float64)
        self.m2 = np.zeros((2, 2), dtype=np.float64)
        self.last_delta = None
        self.last_match_distance = None
        self.last_lidar_ratio = None
        self.last_camera_ratio = None

    def update(
        self,
        delta: np.ndarray,
        match_distance: float,
        lidar_ratio: float,
        camera_ratio: float,
    ) -> None:
        self.count += 1
        difference = delta - self.mean
        self.mean += difference / float(self.count)
        difference_after = delta - self.mean
        self.m2 += np.outer(difference, difference_after)
        self.last_delta = delta.copy()
        self.last_match_distance = float(match_distance)
        self.last_lidar_ratio = float(lidar_ratio)
        self.last_camera_ratio = float(camera_ratio)

    def snapshot(self) -> dict:
        covariance = None
        standard_deviation = None
        if self.count >= 2:
            covariance_array = self.m2 / float(self.count - 1)
            covariance = covariance_array.tolist()
            standard_deviation = np.sqrt(
                np.maximum(np.diag(covariance_array), 0.0)
            ).tolist()
        return {
            "count": int(self.count),
            "mean": self.mean.tolist(),
            "covariance": covariance,
            "standard_deviation": standard_deviation,
            "last_delta": (
                None if self.last_delta is None else self.last_delta.tolist()
            ),
            "last_match_distance": self.last_match_distance,
            "last_lidar_ratio": self.last_lidar_ratio,
            "last_camera_ratio": self.last_camera_ratio,
        }


class ProjectionLidarStatistics:
    """筛选可靠相机投影—雷达配对并维护全局/分 class 在线统计。"""

    REJECTION_LABELS = {
        "gray_disabled": "灰色来源未启用",
        "guessed_identity": "猜测身份",
        "missing_armor": "缺少装甲观测",
        "armor_confidence": "armor置信度不足",
        "digit_confidence": "digit置信度不足",
        "combined_confidence": "combined置信度不足",
        "invalid_point": "坐标非法",
        "match_distance": "匹配距离过大",
        "not_mutual_nearest": "非互为最近邻",
        "lidar_margin": "雷达侧margin不足",
        "camera_margin": "相机侧margin不足",
        "lidar_ratio": "雷达侧ratio过大",
        "camera_ratio": "相机侧ratio过大",
        "sample_interval": "采样间隔未到",
    }

    def __init__(self, config: dict, class_names: Sequence[str]) -> None:
        self.enabled = bool(config["enabled"])
        self.include_gray = bool(config["include_gray_camera_lidar"])
        self.min_armor_confidence = float(config["min_armor_confidence"])
        self.min_digit_confidence = float(config["min_digit_confidence"])
        self.min_combined_confidence = float(config["min_combined_confidence"])
        self.max_match_distance_m = float(config["max_match_distance_m"])
        self.min_lidar_margin_m = float(config["min_lidar_margin_m"])
        self.min_camera_margin_m = float(config["min_camera_margin_m"])
        self.max_lidar_ratio = float(config["max_lidar_ambiguity_ratio"])
        self.max_camera_ratio = float(config["max_camera_ambiguity_ratio"])
        self.min_sample_interval_sec = float(config["min_sample_interval_sec"])
        self.class_names = tuple(str(name) for name in class_names)
        self._lock = threading.Lock()
        self._global = _OnlineResidualStatistics()
        self._by_faction = {
            "blue": _OnlineResidualStatistics(),
            "red": _OnlineResidualStatistics(),
        }
        self._by_class = [
            _OnlineResidualStatistics() for _ in range(len(self.class_names))
        ]
        self._last_sample_time = [None] * len(self.class_names)
        self._evaluated = 0
        self._accepted = 0
        self._rejections = Counter()
        self._reset_count = 0
        self._last_reset_reason = "初始化"

    @staticmethod
    def _distance(first: Sequence[float], second: Sequence[float]) -> float:
        return float(
            np.linalg.norm(
                np.asarray(first[:2], dtype=np.float64)
                - np.asarray(second[:2], dtype=np.float64)
            )
        )

    @staticmethod
    def _ambiguity_ratio(distance: float, alternative_distance: float) -> float:
        if not math.isfinite(alternative_distance):
            return 0.0
        if distance <= 0.0:
            return 0.0
        if alternative_distance <= 0.0:
            return 1.0
        # 互为最近邻检查已经保证分母不小于 d1；仅抑制浮点误差越界。
        return min(1.0, max(0.0, distance / alternative_distance))

    def reset(self, reason: str) -> None:
        with self._lock:
            self._global.reset()
            for statistics in self._by_faction.values():
                statistics.reset()
            for statistics in self._by_class:
                statistics.reset()
            self._last_sample_time = [None] * len(self.class_names)
            self._evaluated = 0
            self._accepted = 0
            self._rejections.clear()
            self._reset_count += 1
            self._last_reset_reason = str(reason)

    def _reject_locked(self, reason: str) -> bool:
        self._rejections[reason] += 1
        return False

    def observe(
        self,
        *,
        class_id: int,
        source: str,
        identity_confirmed: bool,
        car_index: int,
        camera_point: Sequence[float],
        lidar_index: int,
        lidar_points: Sequence[Sequence[float]],
        available_lidar_indices: Sequence[int],
        camera_points: Sequence[tuple[int, Sequence[float]]],
        selected_armor,
        now_monotonic: float,
    ) -> bool:
        """评估一对已提交匹配；接受时同时更新全局与对应 class。"""
        if not self.enabled:
            return False
        with self._lock:
            self._evaluated += 1
            if source == "gray_camera_lidar" and not self.include_gray:
                return self._reject_locked("gray_disabled")
            if not identity_confirmed:
                return self._reject_locked("guessed_identity")
            if selected_armor is None:
                return self._reject_locked("missing_armor")
            if float(selected_armor.armor_confidence) < self.min_armor_confidence:
                return self._reject_locked("armor_confidence")
            if float(selected_armor.digit_confidence) < self.min_digit_confidence:
                return self._reject_locked("digit_confidence")
            if (
                float(selected_armor.combined_confidence)
                < self.min_combined_confidence
            ):
                return self._reject_locked("combined_confidence")
            if not 0 <= int(class_id) < len(self._by_class):
                return self._reject_locked("invalid_point")
            try:
                camera_xy = np.asarray(camera_point[:2], dtype=np.float64)
                lidar_xy = np.asarray(lidar_points[lidar_index][:2], dtype=np.float64)
            except (TypeError, ValueError, IndexError):
                return self._reject_locked("invalid_point")
            if (
                camera_xy.shape != (2,)
                or lidar_xy.shape != (2,)
                or not np.isfinite(camera_xy).all()
                or not np.isfinite(lidar_xy).all()
            ):
                return self._reject_locked("invalid_point")

            distance = float(np.linalg.norm(lidar_xy - camera_xy))
            if distance > self.max_match_distance_m:
                return self._reject_locked("match_distance")

            alternative_lidar_distances = [
                self._distance(camera_xy, lidar_points[index])
                for index in available_lidar_indices
                if int(index) != int(lidar_index)
            ]
            alternative_camera_distances = [
                self._distance(point, lidar_xy)
                for index, point in camera_points
                if int(index) != int(car_index)
            ]
            lidar_second = min(alternative_lidar_distances, default=float("inf"))
            camera_second = min(
                alternative_camera_distances, default=float("inf")
            )
            nearest_epsilon = 1e-9
            if (
                lidar_second < distance - nearest_epsilon
                or camera_second < distance - nearest_epsilon
            ):
                return self._reject_locked("not_mutual_nearest")
            if (
                math.isfinite(lidar_second)
                and lidar_second - distance < self.min_lidar_margin_m
            ):
                return self._reject_locked("lidar_margin")
            if (
                math.isfinite(camera_second)
                and camera_second - distance < self.min_camera_margin_m
            ):
                return self._reject_locked("camera_margin")

            lidar_ratio = self._ambiguity_ratio(distance, lidar_second)
            camera_ratio = self._ambiguity_ratio(distance, camera_second)
            if lidar_ratio > self.max_lidar_ratio:
                return self._reject_locked("lidar_ratio")
            if camera_ratio > self.max_camera_ratio:
                return self._reject_locked("camera_ratio")

            now = float(now_monotonic)
            last_sample_time = self._last_sample_time[class_id]
            if (
                last_sample_time is not None
                and now - last_sample_time < self.min_sample_interval_sec
            ):
                return self._reject_locked("sample_interval")

            delta = lidar_xy - camera_xy
            self._global.update(delta, distance, lidar_ratio, camera_ratio)
            faction = "blue" if class_id < 5 else "red"
            self._by_faction[faction].update(
                delta, distance, lidar_ratio, camera_ratio
            )
            self._by_class[class_id].update(
                delta, distance, lidar_ratio, camera_ratio
            )
            self._last_sample_time[class_id] = now
            self._accepted += 1
            return True

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "evaluated": int(self._evaluated),
                "accepted": int(self._accepted),
                "rejected": int(self._evaluated - self._accepted),
                "rejections": dict(self._rejections),
                "reset_count": int(self._reset_count),
                "last_reset_reason": self._last_reset_reason,
                "global": self._global.snapshot(),
                "by_faction": {
                    name: statistics.snapshot()
                    for name, statistics in self._by_faction.items()
                },
                "by_class": {
                    name: statistics.snapshot()
                    for name, statistics in zip(self.class_names, self._by_class)
                },
            }

    @staticmethod
    def _format_vector(values) -> str:
        if values is None:
            return "N/A"
        return f"({values[0]:+.4f}, {values[1]:+.4f})"

    @staticmethod
    def _format_covariance_lines(values, indent: str) -> list[str]:
        if values is None:
            return [f"{indent}cov=N/A"]
        return [
            f"{indent}cov=[[{values[0][0]:.6f}, {values[0][1]:.6f}],",
            f"{indent}     [{values[1][0]:.6f}, {values[1][1]:.6f}]]",
        ]

    @classmethod
    def _append_statistics_block(
        cls,
        lines: list[str],
        title: str,
        statistics: dict,
        indent: str = "",
    ) -> None:
        mean = statistics["mean"] if statistics["count"] > 0 else None
        lines.append(f"{indent}{title}: n={statistics['count']}")
        lines.append(f"{indent}  mean={cls._format_vector(mean)}")
        lines.extend(
            cls._format_covariance_lines(
                statistics["covariance"], f"{indent}  "
            )
        )
        if statistics["last_delta"] is not None:
            lines.append(
                f"{indent}  std="
                f"{cls._format_vector(statistics['standard_deviation'])}"
            )
            lines.append(
                f"{indent}  last={cls._format_vector(statistics['last_delta'])} "
                f"d={statistics['last_match_distance']:.4f} "
                f"ratio(lidar/camera)="
                f"{statistics['last_lidar_ratio']:.3f}/"
                f"{statistics['last_camera_ratio']:.3f}"
            )

    def format_report(self) -> str:
        snapshot = self.snapshot()
        if not snapshot["enabled"]:
            return "投影-雷达可靠匹配残差统计已关闭。"

        lines = [
            "投影-雷达可靠匹配残差统计（单位 m，delta=lidar-camera）",
            (
                f"评估/接受/拒绝: {snapshot['evaluated']}/"
                f"{snapshot['accepted']}/{snapshot['rejected']}，"
                f"最近清零: {snapshot['last_reset_reason']}"
            ),
            "",
        ]

        self._append_statistics_block(lines, "所有机器人总计", snapshot["global"])
        lines.append("")
        self._append_statistics_block(
            lines, "蓝方总计", snapshot["by_faction"]["blue"]
        )
        lines.append("")
        lines.append("蓝方机器人:")
        for name in self.class_names[:5]:
            self._append_statistics_block(
                lines, name, snapshot["by_class"][name], indent="  "
            )
            lines.append("")

        lines.append("================================")
        lines.append("")
        self._append_statistics_block(
            lines, "红方总计", snapshot["by_faction"]["red"]
        )
        lines.append("")
        lines.append("红方机器人:")
        for name in self.class_names[5:10]:
            self._append_statistics_block(
                lines, name, snapshot["by_class"][name], indent="  "
            )
            lines.append("")

        lines.append("拒绝原因:")
        if not snapshot["rejections"]:
            lines.append("  暂无")
        else:
            for key, count in sorted(
                snapshot["rejections"].items(), key=lambda item: (-item[1], item[0])
            ):
                label = self.REJECTION_LABELS.get(key, key)
                lines.append(f"  {label}: {count}")
        return "\n".join(lines)
