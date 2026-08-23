"""车辆—装甲流水线使用的数据包、邮箱和诊断计数器。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

import numpy as np


def freeze_mapping(values: Optional[Mapping] = None) -> Mapping:
    """复制并冻结一层映射，防止发布后被其他线程原地修改。"""
    return MappingProxyType(dict(values or {}))


@dataclass(frozen=True)
class CarStageItem:
    """A 阶段输出的单个车辆及其 ROI。"""

    car_class_id: int
    car_box: Tuple[float, float, float, float]
    car_conf: float
    track_id: int
    vehicle_roi: np.ndarray = field(compare=False, repr=False)


@dataclass(frozen=True)
class StageAPacket:
    """A 阶段交给 B 阶段的完整同源快照。"""

    pipeline_epoch: int
    source_sequence: int
    frame: np.ndarray = field(compare=False, repr=False)
    frame_stamp: Any
    frame_width: int
    frame_height: int
    input_frame_age_ms: Optional[float]
    ingress_monotonic: float
    observation_monotonic: float
    published_monotonic: float
    faction: Any
    lidar_snapshot: Any = field(compare=False, repr=False)
    ally_snapshot: Any = field(compare=False, repr=False)
    lidar_fly_points: Tuple[Tuple[float, float, float], ...]
    outpost_alive: int
    cars: Tuple[CarStageItem, ...]
    stage_a_timings_ms: Mapping[str, float]


@dataclass(frozen=True)
class TrackOutputSnapshot:
    """TrackingState 的只读输出副本，不向 UI 暴露可变状态机对象。"""

    class_id: int
    name: str
    identity_state: Any
    spatial_state: Any
    frame_source: Any
    output_valid: bool
    output_position_2d_m: Optional[Tuple[float, float]]
    is_guessed_output: bool
    pos_3d: Tuple[float, float, float]
    last_car_box: Optional[Tuple[float, float, float, float]]


@dataclass(frozen=True)
class DivisionPositionSnapshot:
    x: int
    y: int
    is_valid: bool
    source: str = "invalid"
    is_guessed: bool = False


@dataclass(frozen=True)
class CompletedFrame:
    """一次成功提交后的跨线程一致性输出。"""

    pipeline_epoch: int
    source_sequence: int
    commit_sequence: int
    source_gap_from_previous_commit: int
    frame: np.ndarray = field(compare=False, repr=False)
    frame_stamp: Any
    input_frame_age_ms: Optional[float]
    faction: Any
    outpost_alive: int
    detections: Tuple[Any, ...]
    tracks: Tuple[TrackOutputSnapshot, ...]
    tracker_lidar_points: Tuple[Tuple[float, float], ...]
    lidar_point_owners: Mapping[int, int]
    projection_points_cm: Tuple[Tuple[float, float], ...]
    lidar_fly_points_cm: Tuple[Tuple[int, int, int], ...]
    divisions_pos: Tuple[DivisionPositionSnapshot, ...]
    stage_a_timings_ms: Mapping[str, float]
    stage_b_timings_ms: Mapping[str, float]
    pipeline_timings_ms: Mapping[str, float]
    processing_age_ms: float
    commit_interval_ms: Optional[float]
    commit_fps: Optional[float]
    error: Optional[str] = None


@dataclass(frozen=True)
class PipelineDiagnosticsSnapshot:
    source_sequence: int
    commit_sequence: int
    published_total: int
    taken_total: int
    committed_total: int
    pending_replaced_total: int
    stale_before_stage_b_total: int
    stale_before_state_total: int
    epoch_drop_total: int
    out_of_order_drop_total: int
    shutdown_drop_total: int
    stage_a_error_total: int
    stage_b_error_total: int
    armor_error_total: int
    tracker_error_total: int
    commit_error_total: int
    last_commit_monotonic: Optional[float]


class PipelineDiagnostics:
    """流水线计数器；所有复合读写均由内部锁保护。"""

    _COUNTERS = (
        "source_sequence",
        "commit_sequence",
        "published_total",
        "taken_total",
        "committed_total",
        "pending_replaced_total",
        "stale_before_stage_b_total",
        "stale_before_state_total",
        "epoch_drop_total",
        "out_of_order_drop_total",
        "shutdown_drop_total",
        "stage_a_error_total",
        "stage_b_error_total",
        "armor_error_total",
        "tracker_error_total",
        "commit_error_total",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        for name in self._COUNTERS:
            setattr(self, name, 0)
        self.last_commit_monotonic: Optional[float] = None

    def increment(self, name: str, amount: int = 1) -> int:
        if name not in self._COUNTERS:
            raise KeyError(name)
        with self._lock:
            value = int(getattr(self, name)) + int(amount)
            setattr(self, name, value)
            return value

    def next_source_sequence(self) -> int:
        return self.increment("source_sequence")

    def record_commit(self, now_monotonic: float) -> tuple[int, Optional[float]]:
        with self._lock:
            previous = self.last_commit_monotonic
            self.commit_sequence += 1
            self.committed_total = self.commit_sequence
            self.last_commit_monotonic = float(now_monotonic)
            interval = None if previous is None else float(now_monotonic) - previous
            return self.commit_sequence, interval

    def snapshot(self) -> PipelineDiagnosticsSnapshot:
        with self._lock:
            values = {name: int(getattr(self, name)) for name in self._COUNTERS}
            values["last_commit_monotonic"] = self.last_commit_monotonic
            return PipelineDiagnosticsSnapshot(**values)


class LatestPacketMailbox:
    """容量为 1 的 latest-wins 邮箱；发布永不等待消费者。"""

    def __init__(
        self, diagnostics: Optional[PipelineDiagnostics] = None
    ) -> None:
        self._condition = threading.Condition()
        self._pending: Optional[StageAPacket] = None
        self._closed = False
        self._diagnostics = diagnostics

    def publish_latest(self, packet: StageAPacket) -> bool:
        """发布最新包；返回是否覆盖了尚未取走的旧包。"""
        with self._condition:
            if self._closed:
                return False
            replaced = self._pending is not None
            self._pending = packet
            if self._diagnostics is not None:
                self._diagnostics.increment("published_total")
                if replaced:
                    self._diagnostics.increment("pending_replaced_total")
            self._condition.notify()
            return replaced

    def take_latest(self) -> Optional[StageAPacket]:
        """等待并取走当前最新包；关闭且无待处理包时返回 None。"""
        with self._condition:
            while self._pending is None and not self._closed:
                self._condition.wait()
            if self._pending is None:
                return None
            packet = self._pending
            self._pending = None
            if self._diagnostics is not None:
                self._diagnostics.increment("taken_total")
            return packet

    def clear(self, reason: str = "") -> Optional[StageAPacket]:
        with self._condition:
            packet = self._pending
            self._pending = None
            if packet is not None and self._diagnostics is not None:
                if reason:
                    self._diagnostics.increment("epoch_drop_total")
            return packet

    def close(self) -> Optional[StageAPacket]:
        """关闭并丢弃待处理包，同时唤醒消费者。"""
        with self._condition:
            self._closed = True
            packet = self._pending
            self._pending = None
            if packet is not None and self._diagnostics is not None:
                self._diagnostics.increment("shutdown_drop_total")
            self._condition.notify_all()
            return packet
