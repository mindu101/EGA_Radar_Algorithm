from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from tracker.CascadeMatchTracker.kalman import KalmanFilter2d


class ArmorColor(Enum):
    BLUE = "blue"
    RED = "red"
    GRAY = "gray"


class ProjectionMode(Enum):
    DIRECT = "direct"
    XY_MAP = "xy_map"


class IdentityState(Enum):
    UNSEEN = "unseen"
    GUESSED = "guessed"
    CONFIRMED = "confirmed"


class SpatialState(Enum):
    UNINITIALIZED = "uninitialized"
    TRACKING = "tracking"
    COASTING = "coasting"
    LOST = "lost"


class FrameSource(Enum):
    INVALID = "invalid"
    CAMERA_LIDAR = "camera_lidar"
    GRAY_CAMERA_LIDAR = "gray_camera_lidar"
    ALLY_LIDAR = "ally_lidar"
    ALLY_FEEDBACK = "ally_feedback"
    LIDAR_HISTORY = "lidar_history"
    GRAY_RADAR_GUESSED = "gray_radar_guessed"
    RADAR_GUESSED = "radar_guessed"
    KF_PREDICTED = "kf_predicted"
    FIXED_GUESSED = "fixed_guessed"


class State(Enum):
    """旧 UI 的兼容状态；不再反向驱动新状态机。"""

    INACTIVE = "inactive"
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


@dataclass(frozen=True)
class ArmorObservation:
    armor_box: Tuple[float, float, float, float]
    armor_confidence: float
    digit_index: int
    digit_confidence: float
    color: ArmorColor
    raw_class_id: int
    combined_confidence: float


@dataclass
class SingleDetectionResult:
    """检测器输出，保留旧字段供可视化和数据集导出使用。"""

    class_id: int
    class_conf: float
    armor_box: List[float]
    car_box: List[float]
    car_conf: float
    pos_3d: Optional[List[float]]
    time_stamp: float
    bot_id: int
    armor_observations: Tuple[ArmorObservation, ...] = ()
    selected_armor: Optional[ArmorObservation] = None
    projection_mode: ProjectionMode = ProjectionMode.DIRECT


@dataclass(frozen=True)
class CarObservation:
    car_index: int
    car_box: Tuple[float, float, float, float]
    car_confidence: float
    bot_id: int
    armor_observations: Tuple[ArmorObservation, ...]
    selected_armor: Optional[ArmorObservation]
    semantic_candidates: Tuple[int, ...]
    projection_mode: ProjectionMode
    ray_world_3d: Optional[Tuple[float, float, float]]
    ray_uwb_2d: Optional[Tuple[float, float]]


@dataclass(frozen=True, order=True)
class TripletCandidate:
    class_id: int
    car_index: int
    lidar_index: int
    cost: float = field(compare=False)


@dataclass(frozen=True)
class TripletMatch:
    class_id: int
    car_index: int
    lidar_index: int
    cost: float


@dataclass(frozen=True)
class HistoryMatch:
    class_id: int
    lidar_index: Optional[int]
    cost: float


@dataclass(frozen=True)
class SemanticHistoryEntry:
    class_id: int
    confidence: float


class BotIdTrack:
    """BoT-SORT ID 的短期语义历史，不保存世界坐标。"""

    def __init__(self, bot_id: int, queue_size: int = 30):
        self.bot_id = int(bot_id)
        self.entries: Deque[Optional[SemanticHistoryEntry]] = deque(maxlen=queue_size)
        self.last_seen_frame = -1

    def update(self, class_id: int, confidence: float, frame_index: Optional[int] = None) -> None:
        self.entries.append(
            None
            if class_id < 0
            else SemanticHistoryEntry(int(class_id), float(confidence))
        )
        if frame_index is not None:
            self.last_seen_frame = int(frame_index)

    def mark_missing(self) -> None:
        self.entries.append(None)

    def get_class_id_exponent_confidence(
        self, history_length: int = 10, tau: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        window = list(self.entries)[-max(1, int(history_length)) :]
        distribution = np.zeros(10, dtype=np.float64)
        for age, entry in enumerate(reversed(window)):
            if entry is None:
                continue
            weight = float(np.exp(-float(tau) * age)) * entry.confidence
            if 0 <= entry.class_id <= 9:
                distribution[entry.class_id] += weight
            elif 10 <= entry.class_id <= 14:
                distribution[entry.class_id - 10] += 0.5 * weight
                distribution[entry.class_id - 5] += 0.5 * weight
        confidence = distribution.astype(np.float32, copy=True)
        total = float(distribution.sum())
        if total > 0.0:
            distribution /= total
        return confidence, distribution.astype(np.float32)


class TrackingState:
    """一个确定 class 的身份、空间和本帧来源状态。"""

    def __init__(
        self,
        class_id: int,
        name: str,
        position_kalman_config: Optional[dict] = None,
        max_frames: int = 30,
    ) -> None:
        position_cfg = position_kalman_config or {}
        self.class_id = int(class_id)
        self.name = str(name)
        self.identity_state = IdentityState.UNSEEN
        self.spatial_state = SpatialState.UNINITIALIZED
        self.frame_source = FrameSource.INVALID
        self.output_valid = False
        self.output_position_2d_m: Optional[List[float]] = None
        self.kalman_filter_2d = KalmanFilter2d(
            initial_pos=[0.0, 0.0],
            q_std=float(position_cfg.get("process_q_std", 2.0)),
            r_std=float(position_cfg.get("lidar_r_std_m", 0.2)),
            dt=float(position_cfg.get("default_dt_sec", 0.033333)),
        )
        self.kalman_initialized = False
        self.associated_bot_id: Optional[int] = None
        self.last_car_box: Optional[List[float]] = None
        self.last_measurement_position: Optional[List[float]] = None
        self.last_camera_frame = -1
        self.last_gray_camera_frame = -1
        self.last_lidar_frame = -1
        self.last_update_monotonic: Optional[float] = None
        self.lidar_miss_count = 0
        self.continuous_loss_count = 0
        self.unseen_frame_count = 0
        self.fixed_guess_point: Optional[List[float]] = None
        self.preempted_this_frame = False
        self.class_id_trajectory: Deque[int] = deque(maxlen=max_frames)
        self.bot_id_trajectory: Deque[int] = deque(maxlen=max_frames)
        self.pos_3d = [0.0, 0.0, 0.0]

    def begin_frame(self) -> None:
        self.frame_source = FrameSource.INVALID
        self.output_valid = False
        self.output_position_2d_m = None
        self.preempted_this_frame = False

    def reset_spatial(self) -> None:
        self.spatial_state = SpatialState.UNINITIALIZED
        self.frame_source = FrameSource.INVALID
        self.output_valid = False
        self.output_position_2d_m = None
        self.last_measurement_position = None
        self.kalman_initialized = False
        self.kalman_filter_2d.reset([0.0, 0.0])
        self.fixed_guess_point = None
        self.continuous_loss_count = 0
        self.unseen_frame_count = 0
        self.lidar_miss_count = 0

    @property
    def is_active(self) -> bool:
        return self.output_valid

    @property
    def is_start_guess(self) -> bool:
        return self.frame_source == FrameSource.FIXED_GUESSED and self.output_valid

    @property
    def is_guessed_output(self) -> bool:
        """当前坐标是否仍属于猜测身份或直接猜点结果。"""
        return self.identity_state == IdentityState.GUESSED or self.frame_source in {
            FrameSource.GRAY_RADAR_GUESSED,
            FrameSource.RADAR_GUESSED,
            FrameSource.FIXED_GUESSED,
        }

    @property
    def state(self) -> State:
        if self.spatial_state == SpatialState.LOST:
            return State.LOST
        if self.identity_state == IdentityState.CONFIRMED:
            return State.CONFIRMED
        if self.identity_state == IdentityState.GUESSED:
            return State.TENTATIVE
        return State.INACTIVE

    @property
    def bot_id(self) -> int:
        return -1 if self.associated_bot_id is None else self.associated_bot_id

    @bot_id.setter
    def bot_id(self, value: Optional[int]) -> None:
        self.associated_bot_id = None if value is None or int(value) < 0 else int(value)

    @property
    def car_box(self) -> Optional[List[float]]:
        return self.last_car_box

    @car_box.setter
    def car_box(self, value: Optional[Sequence[float]]) -> None:
        self.last_car_box = None if value is None else list(value)

    @property
    def pos_2d_uwb(self) -> List[float]:
        if self.output_position_2d_m is not None:
            return list(self.output_position_2d_m)
        if self.kalman_initialized:
            return self.kalman_filter_2d.get_position()
        return [0.0, 0.0]

    @pos_2d_uwb.setter
    def pos_2d_uwb(self, value: Sequence[float]) -> None:
        self.output_position_2d_m = [float(value[0]), float(value[1])]

    @property
    def guess_point(self) -> List[float]:
        return list(self.fixed_guess_point or [0.0, 0.0])

    @guess_point.setter
    def guess_point(self, value: Sequence[float]) -> None:
        self.fixed_guess_point = [float(value[0]), float(value[1])]
