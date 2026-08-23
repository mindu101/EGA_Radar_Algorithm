from __future__ import annotations

import logging
import math
import time
from threading import RLock
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from driver.referee.referee_comm import FACTION
from tracker.CascadeMatchTracker.detector import BaseDetector
from tracker.CascadeMatchTracker.guess_pts import PointGuesser
from tracker.CascadeMatchTracker.projection_lidar_statistics import (
    ProjectionLidarStatistics,
)
from tracker.CascadeMatchTracker.type import (
    ArmorColor,
    BotIdTrack,
    CarObservation,
    FrameSource,
    HistoryMatch,
    IdentityState,
    ProjectionMode,
    SpatialState,
    TrackingState,
    TripletCandidate,
    TripletMatch,
)
from transform.solidwork2uwb import solidwork2uwb


logger = logging.getLogger(__name__)


class CascadeMatchTracker(BaseDetector):
    """按确定优先级关联当前相机证据、雷达点和二维轨迹。"""

    SOURCE_PRIORITY = {
        FrameSource.ALLY_LIDAR: 0,
        FrameSource.ALLY_FEEDBACK: 1,
        FrameSource.CAMERA_LIDAR: 2,
        FrameSource.GRAY_CAMERA_LIDAR: 3,
        FrameSource.LIDAR_HISTORY: 4,
        FrameSource.GRAY_RADAR_GUESSED: 5,
        FrameSource.RADAR_GUESSED: 6,
        FrameSource.KF_PREDICTED: 7,
        FrameSource.FIXED_GUESSED: 8,
        FrameSource.INVALID: 9,
    }
    ROLE_BY_OFFSET = {
        0: "hero",
        1: "engineer",
        2: "standard_3",
        3: "standard_4",
        4: "sentry",
    }

    def __init__(
        self,
        config: Dict,
        pixel_world_transform,
        visualize: bool = False,
        xy_map_manager=None,
    ) -> None:
        self._validate_config(config)
        super().__init__(
            config,
            pixel_world_transform,
            visualize=visualize,
            xy_map_manager=xy_map_manager,
        )
        self._initialize_runtime(config)

    @classmethod
    def create_for_test(cls, config: Dict) -> "CascadeMatchTracker":
        """不加载 GPU 模型的 CPU 状态机测试入口。"""
        cls._validate_config(config)
        self = cls.__new__(cls)
        self.config = config
        self.class_names = list(config["armor_detector"]["class_names"])
        self.xy_map_manager = None
        self.pixel_world_transform = None
        self.visualize = False
        self.detector_timing_stats = {}
        self.detector_frame_timing_ms = {}
        self.track_frame_timing_ms = {}
        self.timing_debug_enabled = True
        self.timing_alpha = 0.2
        self._timing_lock = RLock()
        cascade = config["tracker"]["cascade"]
        self.tracks = [
            TrackingState(
                class_id=index,
                name=name,
                position_kalman_config=cascade["position_kalman"],
                max_frames=int(cascade["bot_history"]["queue_size"]),
            )
            for index, name in enumerate(self.class_names[:10])
        ]
        visual_cfg = config.get("visualization", {})
        self.visualization_enabled = False
        self.visualization_update_hz = float(visual_cfg.get("update_hz", 30.0))
        self.visualization_interval_sec = 1.0 / max(0.1, self.visualization_update_hz)
        self.visualization_max_width = int(visual_cfg.get("output_max_width", 1280))
        self.visualization_max_height = int(visual_cfg.get("output_max_height", 960))
        self.vis_box_thickness = int(visual_cfg.get("box_thickness", 2))
        self.vis_label_font_scale = float(visual_cfg.get("label_font_scale", 0.5))
        self.vis_label_font_thickness = int(visual_cfg.get("label_font_thickness", 1))
        self.vis_label_offset_y = int(visual_cfg.get("label_offset_y", 10))
        self._initialize_runtime(config)
        return self

    def _initialize_runtime(self, config: Dict) -> None:
        self.cascade = config["tracker"]["cascade"]
        self.position_cfg = self.cascade["position_kalman"]
        self.colored_cfg = self.cascade["colored_triplet_match"]
        self.gray_cfg = self.cascade["gray_match"]
        self.history_cfg = self.cascade["history_lidar_match"]
        self.ally_cfg = self.cascade["ally_fusion"]
        self.loss_cfg = self.cascade["loss"]
        self.guessing_cfg = self.cascade["guessing"]
        self.bot_cfg = self.cascade["bot_history"]
        self.lidar_config = config["lidar_fusion"]
        self.point_guesser = PointGuesser(config["tracker"]["point_guesser_config_path"])
        self.projection_lidar_statistics = ProjectionLidarStatistics(
            config["projection_lidar_statistics"], self.class_names[:10]
        )
        self.rng = np.random.Generator(np.random.PCG64(int(self.cascade["random_seed"])))
        self.bot_id_trajectories: Dict[int, BotIdTrack] = {}
        self.frame_count = 0
        self.real_dt = float(self.position_cfg["default_dt_sec"])
        self.last_frame_monotonic: Optional[float] = None
        self.faction = FACTION.UNKONWN
        self._effective_faction_last: Optional[FACTION] = None
        self.latest_detections = []
        self.latest_lidar_points: Tuple[Tuple[float, float], ...] = ()
        self.latest_lidar_matches: Dict[int, int] = {}
        self._matched_lidar_classes: Dict[int, int] = {}
        self.track_timing_stats = {}
        visual_cfg = config.get("visualization", {})
        self.visualization_max_width = max(
            320, int(visual_cfg.get("output_max_width", 1280))
        )
        self.visualization_max_height = max(
            240, int(visual_cfg.get("output_max_height", 960))
        )

    @staticmethod
    def _validate_config(config: Dict) -> None:
        def required(path: str):
            value = config
            for part in path.split("."):
                if not isinstance(value, dict) or part not in value:
                    raise ValueError(f"{path} 缺失")
                value = value[part]
            return value

        def number(path: str) -> float:
            value = required(path)
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} 必须为数值，当前值={value!r}") from exc
            if not math.isfinite(parsed):
                raise ValueError(f"{path} 必须为有限数值，当前值={value!r}")
            return parsed

        def integer(path: str) -> int:
            value = required(path)
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} 必须为整数，当前值={value!r}") from exc
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(f"{path} 必须为整数，当前值={value!r}")
            return parsed

        def check_range(path: str, lower: float, upper: float) -> None:
            value = number(path)
            if not lower <= value <= upper:
                raise ValueError(f"{path} 必须位于 [{lower},{upper}]，当前值={value}")

        def check_nonnegative(path: str) -> None:
            value = number(path)
            if value < 0.0:
                raise ValueError(f"{path} 必须大于等于 0，当前值={value}")

        def check_positive(path: str) -> None:
            value = number(path)
            if value <= 0.0:
                raise ValueError(f"{path} 必须大于 0，当前值={value}")

        required_names = ["B1", "B2", "B3", "B4", "B7", "R1", "R2", "R3", "R4", "R7", "G1", "G2", "G3", "G4", "G7"]
        names = required("armor_detector.class_names")
        if names != required_names:
            raise ValueError("armor_detector.class_names 必须与 15 类固定顺序完全一致")
        digit = required("digit_detector")
        if not isinstance(digit, dict):
            raise ValueError("digit_detector 必须为配置映射，不能为 None")
        for path in (
            "car_detector.conf_thres",
            "car_detector.iou_thres",
            "armor_detector.conf_thres",
            "armor_detector.iou_thres",
            "digit_detector.digit_confidence_weight",
            "digit_detector.armor_confidence_weight",
            "digit_detector.min_digit_confidence",
            "digit_detector.min_combined_confidence",
            "digit_detector.colored_anchor_min_combined_confidence",
            "digit_detector.gray_anchor_min_combined_confidence",
            "projection_lidar_statistics.min_armor_confidence",
            "projection_lidar_statistics.min_digit_confidence",
            "projection_lidar_statistics.min_combined_confidence",
            "projection_lidar_statistics.max_lidar_ambiguity_ratio",
            "projection_lidar_statistics.max_camera_ambiguity_ratio",
        ):
            check_range(path, 0.0, 1.0)
        for path in (
            "projection_lidar_statistics.enabled",
            "projection_lidar_statistics.include_gray_camera_lidar",
            "ray_projection.consistency_check.enabled",
        ):
            if not isinstance(required(path), bool):
                raise ValueError(f"{path} 必须为 bool")
        weight_sum = number("digit_detector.digit_confidence_weight") + number(
            "digit_detector.armor_confidence_weight"
        )
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                "digit_detector.digit_confidence_weight + "
                "digit_detector.armor_confidence_weight 必须等于 1.0 ± 1e-6"
            )
        cascade = required("tracker.cascade")
        if not isinstance(cascade, dict):
            raise ValueError("tracker.cascade 缺失")
        enemy_only = required("tracker.cascade.guessing.enemy_only")
        if not isinstance(enemy_only, bool) or not enemy_only:
            raise ValueError("tracker.cascade.guessing.enemy_only 第一版必须为 true")
        if required("ray_projection.map_dtype") != "float32":
            raise ValueError("ray_projection.map_dtype 必须为 float32")
        if required("ray_projection.lookup_mode") != "nearest":
            raise ValueError("ray_projection.lookup_mode 必须为 nearest")
        if integer("ray_projection.generation_chunk_rows") < 1:
            raise ValueError("ray_projection.generation_chunk_rows 必须至少为 1")
        if integer("ray_projection.consistency_check.sample_count") < 1:
            raise ValueError(
                "ray_projection.consistency_check.sample_count 必须至少为 1"
            )
        radar_after = integer("tracker.cascade.loss.radar_guess_after_frames")
        fixed_after = integer("tracker.cascade.loss.fixed_guess_after_frames")
        if fixed_after < radar_after or radar_after < 1:
            raise ValueError("tracker.cascade.loss 帧数门限非法")
        min_dt = number("tracker.cascade.position_kalman.min_dt_sec")
        default_dt = number("tracker.cascade.position_kalman.default_dt_sec")
        max_dt = number("tracker.cascade.position_kalman.max_dt_sec")
        if not (0.0 < min_dt <= default_dt <= max_dt):
            raise ValueError("tracker.cascade.position_kalman 的 dt 配置非法")
        queue_size = integer("tracker.cascade.bot_history.queue_size")
        matching_window = integer("tracker.cascade.bot_history.matching_window")
        if queue_size < 1 or not 1 <= matching_window <= queue_size:
            raise ValueError("tracker.cascade.bot_history 队列或窗口配置非法")
        if integer("tracker.cascade.bot_history.association_ttl_frames") < 1:
            raise ValueError("tracker.cascade.bot_history.association_ttl_frames 必须至少为 1")
        tie_bias = number("tracker.cascade.gray_match.enemy_tie_bias")
        side_margin = number("tracker.cascade.gray_match.side_cost_margin")
        if tie_bias >= side_margin:
            raise ValueError("tracker.cascade.gray_match.enemy_tie_bias 必须小于 side_cost_margin")
        for path in (
            "tracker.cascade.colored_triplet_match.max_ray_lidar_distance_m",
            "tracker.cascade.gray_match.max_ray_lidar_distance_m",
            "tracker.cascade.history_lidar_match.max_distance_m",
            "tracker.cascade.ally_fusion.max_lidar_match_distance_m",
            "lidar_fusion.field_length",
            "lidar_fusion.field_width",
            "projection_lidar_statistics.max_match_distance_m",
            "ray_projection.consistency_check.max_error_m",
        ):
            check_positive(path)
        for path in (
            "tracker.cascade.bot_history.decay_tau",
            "tracker.cascade.colored_triplet_match.semantic_cost_weight",
            "tracker.cascade.colored_triplet_match.ray_distance_cost_weight",
            "tracker.cascade.gray_match.semantic_cost_weight",
            "tracker.cascade.gray_match.track_distance_cost_weight",
            "tracker.cascade.gray_match.ray_distance_cost_weight",
            "tracker.cascade.gray_match.unseen_track_cost",
            "tracker.cascade.gray_match.side_cost_margin",
            "tracker.cascade.gray_match.enemy_tie_bias",
            "tracker.cascade.history_lidar_match.unmatched_cost",
            "tracker.cascade.history_lidar_match.invalid_edge_cost",
            "tracker.cascade.ally_fusion.feedback_timeout_sec",
            "tracker.cascade.ally_fusion.feedback_r_std_m",
            "tracker.cascade.ally_fusion.age_r_std_per_sec",
            "tracker.cascade.position_kalman.process_q_std",
            "tracker.cascade.position_kalman.lidar_r_std_m",
            "tracker.cascade.position_kalman.initial_position_std_m",
            "tracker.cascade.position_kalman.initial_velocity_std_mps",
            "tracker.cascade.position_kalman.hard_reset_distance_m",
            "lidar_fusion.max_snapshot_age_sec",
            "projection_lidar_statistics.min_lidar_margin_m",
            "projection_lidar_statistics.min_camera_margin_m",
            "projection_lidar_statistics.min_sample_interval_sec",
        ):
            check_nonnegative(path)
        fallback = required("tracker.cascade.unknown_faction_fallback")
        if fallback not in ("red", "blue"):
            raise ValueError("tracker.cascade.unknown_faction_fallback 只能为 red 或 blue")
        role_priority = required("tracker.cascade.guessing.role_priority")
        expected_roles = ["hero", "standard_3", "standard_4", "sentry", "engineer"]
        if role_priority != expected_roles:
            raise ValueError(
                "tracker.cascade.guessing.role_priority 必须严格等于 "
                f"{expected_roles}"
            )

    def _effective_faction(self, faction: FACTION) -> FACTION:
        if faction in (FACTION.RED, FACTION.BLUE):
            return faction
        fallback = str(self.cascade.get("unknown_faction_fallback", "red")).lower()
        return FACTION.BLUE if fallback == "blue" else FACTION.RED

    def _faction_sets(self, faction: FACTION) -> tuple[Set[int], Set[int]]:
        if faction == FACTION.BLUE:
            ally = set(range(0, 5))
        else:
            ally = set(range(5, 10))
        return ally, set(range(10)) - ally

    def _reset_spatial_for_faction_change(self) -> None:
        self.reset_projection_lidar_statistics("阵营变化")
        for track in self.tracks:
            track.spatial_state = SpatialState.UNINITIALIZED
            track.frame_source = FrameSource.INVALID
            track.output_valid = False
            track.output_position_2d_m = None
            track.last_measurement_position = None
            track.kalman_initialized = False
            track.kalman_filter_2d.reset([0.0, 0.0])
            track.fixed_guess_point = None
            track.continuous_loss_count = 0
            track.unseen_frame_count = 0

    def _prepare_frame_inputs(self, lidar_snapshot, faction: FACTION, now: float) -> List[Tuple[float, float]]:
        if lidar_snapshot is None:
            return []
        received = getattr(lidar_snapshot, "received_monotonic", None)
        points = getattr(lidar_snapshot, "points_xy_m", ())
        if received is None or now - float(received) > float(self.lidar_config["max_snapshot_age_sec"]) or now < float(received):
            return []
        result = []
        field_length = float(self.lidar_config["field_length"])
        field_width = float(self.lidar_config["field_width"])
        for raw in points:
            if len(raw) < 2:
                continue
            x, y = float(raw[0]), float(raw[1])
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if not (0.0 <= x <= field_length and 0.0 <= y <= field_width):
                continue
            if faction == FACTION.BLUE and bool(self.lidar_config.get("mirror_for_blue", True)):
                x, y = field_length - x, field_width - y
            result.append((x, y))
        return result

    @staticmethod
    def _distance(first: Sequence[float], second: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(first[:2], dtype=float) - np.asarray(second[:2], dtype=float)))

    def _predict_position_filters(self, dt: float) -> None:
        for track in self.tracks:
            track.begin_frame()
            if track.kalman_initialized:
                predicted, _ = track.kalman_filter_2d.predict(dt=dt)
                track.output_position_2d_m = list(predicted)

    def _build_car_observations(self, detections, faction: FACTION) -> List[CarObservation]:
        observations = []
        field_length = float(self.lidar_config["field_length"])
        field_width = float(self.lidar_config["field_width"])
        for index, detection in enumerate(detections):
            selected = detection.selected_armor
            semantic_candidates: Tuple[int, ...] = ()
            if selected is not None:
                if 0 <= selected.raw_class_id <= 9:
                    semantic_candidates = (selected.raw_class_id,)
                elif 10 <= selected.raw_class_id <= 14:
                    semantic_candidates = (selected.raw_class_id - 10, selected.raw_class_id - 5)
            ray_world = None
            ray_uwb = None
            if selected is not None and detection.pos_3d is not None:
                raw = tuple(float(value) for value in detection.pos_3d[:3])
                if len(raw) == 3 and all(math.isfinite(value) for value in raw):
                    candidate = solidwork2uwb(raw, faction)
                    if 0.0 <= candidate[0] <= field_length and 0.0 <= candidate[1] <= field_width:
                        ray_world = raw
                        ray_uwb = (float(candidate[0]), float(candidate[1]))
            observations.append(
                CarObservation(
                    car_index=index,
                    car_box=tuple(float(value) for value in detection.car_box),
                    car_confidence=float(detection.car_conf),
                    bot_id=int(detection.bot_id),
                    armor_observations=tuple(detection.armor_observations),
                    selected_armor=selected,
                    semantic_candidates=semantic_candidates,
                    projection_mode=detection.projection_mode,
                    ray_world_3d=ray_world,
                    ray_uwb_2d=ray_uwb,
                )
            )
        return observations

    def _update_bot_histories(self, cars: Sequence[CarObservation]) -> None:
        seen = set()
        for car in cars:
            seen.add(car.bot_id)
            history = self.bot_id_trajectories.get(car.bot_id)
            if history is None:
                history = BotIdTrack(car.bot_id, queue_size=int(self.bot_cfg["queue_size"]))
                self.bot_id_trajectories[car.bot_id] = history
            selected = car.selected_armor
            history.update(
                -1 if selected is None else selected.raw_class_id,
                0.0 if selected is None else selected.combined_confidence,
                frame_index=self.frame_count,
            )
        for bot_id, history in list(self.bot_id_trajectories.items()):
            if bot_id not in seen:
                history.mark_missing()
            if history.last_seen_frame >= 0 and self.frame_count - history.last_seen_frame >= int(self.bot_cfg["association_ttl_frames"]):
                del self.bot_id_trajectories[bot_id]
                for track in self.tracks:
                    if track.associated_bot_id == bot_id:
                        track.associated_bot_id = None

    def _solve_partial_rows(self, costs: np.ndarray, valid: np.ndarray) -> List[Tuple[int, Optional[int], float]]:
        row_count = costs.shape[0]
        col_count = costs.shape[1]
        if row_count == 0:
            return []
        invalid_cost = float(self.history_cfg["invalid_edge_cost"])
        matrix = np.full((row_count, col_count + row_count), invalid_cost, dtype=np.float64)
        if col_count:
            matrix[:, :col_count] = np.where(valid, costs, invalid_cost)
        for row in range(row_count):
            matrix[row, col_count + row] = float(self.history_cfg["unmatched_cost"])
        rows, columns = linear_sum_assignment(matrix)
        result = []
        for row, column in zip(rows.tolist(), columns.tolist()):
            if column < col_count and valid[row, column]:
                result.append((row, column, float(costs[row, column])))
            else:
                result.append((row, None, float(self.history_cfg["unmatched_cost"])))
        return result

    def _solve_disjoint_triplets(self, candidates: Sequence[TripletCandidate]) -> List[TripletMatch]:
        if not candidates:
            return []
        by_car: Dict[int, List[TripletCandidate]] = {}
        for candidate in sorted(candidates, key=lambda item: (item.car_index, item.class_id, item.lidar_index, item.cost)):
            by_car.setdefault(candidate.car_index, []).append(candidate)
        car_indices = sorted(by_car)
        memo = {}

        def key_of(selection: Tuple[TripletCandidate, ...]) -> tuple:
            return (
                -len(selection),
                float(sum(item.cost for item in selection)),
                tuple(sorted((item.class_id, item.car_index, item.lidar_index) for item in selection)),
            )

        def solve(position: int, class_mask: int, lidar_mask: int) -> Tuple[TripletCandidate, ...]:
            state = (position, class_mask, lidar_mask)
            cached = memo.get(state)
            if cached is not None:
                return cached
            if position >= len(car_indices):
                return ()
            best = solve(position + 1, class_mask, lidar_mask)
            car_index = car_indices[position]
            for candidate in by_car[car_index]:
                class_bit = 1 << candidate.class_id
                lidar_bit = 1 << candidate.lidar_index
                if class_mask & class_bit or lidar_mask & lidar_bit:
                    continue
                suffix = solve(position + 1, class_mask | class_bit, lidar_mask | lidar_bit)
                selection = (candidate,) + suffix
                if key_of(selection) < key_of(best):
                    best = selection
            memo[state] = best
            return best

        best = solve(0, 0, 0)
        return [TripletMatch(item.class_id, item.car_index, item.lidar_index, item.cost) for item in sorted(best)]

    def _hard_reset_position(self, track: TrackingState, position: Sequence[float]) -> None:
        track.kalman_filter_2d.hard_reset(
            list(position[:2]),
            initial_position_std_m=float(self.position_cfg["initial_position_std_m"]),
            initial_velocity_std_mps=float(self.position_cfg["initial_velocity_std_mps"]),
        )
        track.kalman_initialized = True

    def _set_measurement_output(
        self,
        track: TrackingState,
        position: Sequence[float],
        source: FrameSource,
        now: float,
        hard_reset: bool = False,
        r_std: Optional[float] = None,
        output_exact_measurement: bool = False,
    ) -> None:
        point = [float(position[0]), float(position[1])]
        if hard_reset or not track.kalman_initialized:
            self._hard_reset_position(track, point)
        else:
            track.kalman_filter_2d.update(point, r_std=r_std)
        track.spatial_state = SpatialState.TRACKING
        track.frame_source = source
        track.output_valid = True
        track.output_position_2d_m = point if output_exact_measurement else track.kalman_filter_2d.get_position()
        track.last_measurement_position = point
        track.last_update_monotonic = now
        track.pos_3d = [track.output_position_2d_m[0], track.output_position_2d_m[1], 0.0]

    def _commit_box(self, track: TrackingState, car: CarObservation) -> None:
        track.associated_bot_id = car.bot_id
        track.last_car_box = list(car.car_box)

    def _match_fresh_ally_to_lidar(
        self,
        ally_snapshot,
        faction: FACTION,
        lidar_points: Sequence[Tuple[float, float]],
        used_classes: Set[int],
        used_lidar: Set[int],
        now: float,
    ) -> None:
        if ally_snapshot is None:
            return
        received = getattr(ally_snapshot, "received_monotonic", None)
        if received is None:
            return
        age = now - float(received)
        timeout = float(self.ally_cfg["feedback_timeout_sec"])
        if age < -1e-9 or age > timeout + 1e-9:
            return
        positions = getattr(ally_snapshot, "positions_m", {}) or {}
        ally_classes, _ = self._faction_sets(faction)
        rows = []
        for class_id in sorted(ally_classes):
            role = self.ROLE_BY_OFFSET[class_id % 5]
            raw = positions.get(role)
            if raw is None or len(raw) < 2:
                continue
            point = (float(raw[0]), float(raw[1]))
            if not self._valid_ally_position(point):
                continue
            rows.append((class_id, point))
        if not rows:
            return
        gate = float(self.ally_cfg["max_lidar_match_distance_m"])
        costs = np.zeros((len(rows), len(lidar_points)), dtype=np.float64)
        valid = np.zeros_like(costs, dtype=bool)
        for row_index, (_class_id, point) in enumerate(rows):
            for lidar_index, lidar_point in enumerate(lidar_points):
                distance = self._distance(point, lidar_point)
                costs[row_index, lidar_index] = distance / gate
                valid[row_index, lidar_index] = distance <= gate
        assignments = {row: column for row, column, _cost in self._solve_partial_rows(costs, valid)}
        feedback_r_std = float(self.ally_cfg["feedback_r_std_m"]) + age * float(
            self.ally_cfg["age_r_std_per_sec"]
        )
        for row_index, (class_id, point) in enumerate(rows):
            lidar_index = assignments.get(row_index)
            track = self.tracks[class_id]
            track.identity_state = IdentityState.CONFIRMED
            source = FrameSource.ALLY_FEEDBACK
            if lidar_index is not None:
                used_lidar.add(lidar_index)
                self._matched_lidar_classes[lidar_index] = class_id
                track.last_lidar_frame = self.frame_count
                source = FrameSource.ALLY_LIDAR
            used_classes.add(class_id)
            if source == FrameSource.ALLY_FEEDBACK and not bool(self.ally_cfg["update_kf_without_lidar"]):
                track.spatial_state = SpatialState.TRACKING
                track.frame_source = source
                track.output_valid = True
                track.output_position_2d_m = [point[0], point[1]]
                track.last_measurement_position = [point[0], point[1]]
                track.last_update_monotonic = now
                track.pos_3d = [point[0], point[1], 0.0]
            else:
                self._set_measurement_output(
                    track,
                    point,
                    source,
                    now,
                    r_std=feedback_r_std,
                    output_exact_measurement=True,
                )

    def _valid_ally_position(self, point: Sequence[float]) -> bool:
        x, y = float(point[0]), float(point[1])
        return (
            math.isfinite(x)
            and math.isfinite(y)
            and (x != 0.0 or y != 0.0)
            and 0.0 <= x <= float(self.lidar_config["field_length"])
            and 0.0 <= y <= float(self.lidar_config["field_width"])
        )

    def _build_colored_triplets(
        self,
        cars: Sequence[CarObservation],
        lidar_points: Sequence[Tuple[float, float]],
        used_classes: Set[int],
        used_cars: Set[int],
        used_lidar: Set[int],
    ) -> List[TripletCandidate]:
        candidates = []
        threshold = float(self.config["digit_detector"]["colored_anchor_min_combined_confidence"])
        gate = float(self.colored_cfg["max_ray_lidar_distance_m"])
        for car in cars:
            selected = car.selected_armor
            if car.car_index in used_cars or selected is None or selected.color == ArmorColor.GRAY:
                continue
            if selected.combined_confidence < threshold or car.ray_uwb_2d is None or len(car.semantic_candidates) != 1:
                continue
            class_id = car.semantic_candidates[0]
            if class_id in used_classes:
                continue
            semantic_cost = 1.0 - selected.combined_confidence
            for lidar_index, lidar_point in enumerate(lidar_points):
                if lidar_index in used_lidar:
                    continue
                distance = self._distance(car.ray_uwb_2d, lidar_point)
                if distance > gate:
                    continue
                cost = (
                    float(self.colored_cfg["semantic_cost_weight"]) * semantic_cost
                    + float(self.colored_cfg["ray_distance_cost_weight"]) * distance / gate
                )
                candidates.append(TripletCandidate(class_id, car.car_index, lidar_index, cost))
        return candidates

    def _commit_colored_matches(
        self,
        matches: Sequence[TripletMatch],
        cars_by_index: Dict[int, CarObservation],
        lidar_points: Sequence[Tuple[float, float]],
        used_classes: Set[int],
        used_cars: Set[int],
        used_lidar: Set[int],
        now: float,
    ) -> Set[int]:
        camera_lidar_indices = set()
        for match in matches:
            track = self.tracks[match.class_id]
            car = cars_by_index[match.car_index]
            point = lidar_points[match.lidar_index]
            jump = track.kalman_initialized and self._distance(track.kalman_filter_2d.get_position(), point) >= float(self.position_cfg["hard_reset_distance_m"])
            hard = track.identity_state in (IdentityState.UNSEEN, IdentityState.GUESSED) or jump
            track.identity_state = IdentityState.CONFIRMED
            self._set_measurement_output(track, point, FrameSource.CAMERA_LIDAR, now, hard_reset=hard, r_std=float(self.position_cfg["lidar_r_std_m"]))
            self._commit_box(track, car)
            track.last_camera_frame = self.frame_count
            track.last_lidar_frame = self.frame_count
            used_classes.add(match.class_id)
            used_cars.add(match.car_index)
            used_lidar.add(match.lidar_index)
            self._matched_lidar_classes[match.lidar_index] = match.class_id
            camera_lidar_indices.add(match.lidar_index)
        return camera_lidar_indices

    def _observe_projection_lidar_matches(
        self,
        matches: Sequence[TripletMatch],
        source: FrameSource,
        cars_by_index: Dict[int, CarObservation],
        lidar_points: Sequence[Tuple[float, float]],
        available_lidar_indices: Sequence[int],
        camera_points: Sequence[Tuple[int, Tuple[float, float]]],
        now: float,
    ) -> None:
        """将本帧相机—雷达直接匹配交给可靠性筛选与在线统计。"""
        for match in matches:
            car = cars_by_index[match.car_index]
            if car.ray_uwb_2d is None:
                continue
            track = self.tracks[match.class_id]
            self.projection_lidar_statistics.observe(
                class_id=match.class_id,
                source=source.value,
                identity_confirmed=track.identity_state == IdentityState.CONFIRMED,
                car_index=match.car_index,
                camera_point=car.ray_uwb_2d,
                lidar_index=match.lidar_index,
                lidar_points=lidar_points,
                available_lidar_indices=available_lidar_indices,
                camera_points=camera_points,
                selected_armor=car.selected_armor,
                now_monotonic=now,
            )

    def reset_projection_lidar_statistics(self, reason: str) -> None:
        self.projection_lidar_statistics.reset(reason)

    def get_projection_lidar_statistics_snapshot(self) -> dict:
        return self.projection_lidar_statistics.snapshot()

    def get_projection_lidar_statistics_text(self) -> str:
        return self.projection_lidar_statistics.format_report()

    def _build_gray_maintenance_triplets(
        self,
        cars: Sequence[CarObservation],
        lidar_points: Sequence[Tuple[float, float]],
        used_classes: Set[int],
        used_cars: Set[int],
        used_lidar: Set[int],
        ally_classes: Set[int],
    ) -> List[TripletCandidate]:
        result = []
        threshold = float(self.config["digit_detector"]["gray_anchor_min_combined_confidence"])
        ray_gate = float(self.gray_cfg["max_ray_lidar_distance_m"])
        track_gate = float(self.history_cfg["max_distance_m"])
        for car in cars:
            selected = car.selected_armor
            if car.car_index in used_cars or selected is None or selected.color != ArmorColor.GRAY:
                continue
            if selected.combined_confidence < threshold or car.ray_uwb_2d is None:
                continue
            for class_id in car.semantic_candidates:
                track = self.tracks[class_id]
                # 己方 0x02BB 过期后，旧己方历史/KF 必须完全退出雷达关联。
                # 当前可靠彩色观测仍会在更早的彩色三元组阶段正常参与匹配。
                if class_id in ally_classes or class_id in used_classes or track.identity_state != IdentityState.CONFIRMED or not track.kalman_initialized:
                    continue
                for lidar_index, point in enumerate(lidar_points):
                    if lidar_index in used_lidar:
                        continue
                    ray_distance = self._distance(car.ray_uwb_2d, point)
                    track_distance = self._distance(track.kalman_filter_2d.get_position(), point)
                    if ray_distance > ray_gate or track_distance > track_gate:
                        continue
                    cost = (
                        float(self.gray_cfg["semantic_cost_weight"]) * (1.0 - selected.combined_confidence)
                        + float(self.gray_cfg["track_distance_cost_weight"]) * track_distance / track_gate
                        + float(self.gray_cfg["ray_distance_cost_weight"]) * ray_distance / ray_gate
                    )
                    result.append(TripletCandidate(class_id, car.car_index, lidar_index, cost))
        return result

    def _commit_gray_maintenance_matches(self, matches, cars_by_index, lidar_points, used_classes, used_cars, used_lidar, now) -> None:
        for match in matches:
            track = self.tracks[match.class_id]
            car = cars_by_index[match.car_index]
            self._set_measurement_output(track, lidar_points[match.lidar_index], FrameSource.GRAY_CAMERA_LIDAR, now, r_std=float(self.position_cfg["lidar_r_std_m"]))
            self._commit_box(track, car)
            track.last_gray_camera_frame = self.frame_count
            track.last_lidar_frame = self.frame_count
            used_classes.add(match.class_id)
            used_cars.add(match.car_index)
            used_lidar.add(match.lidar_index)
            self._matched_lidar_classes[match.lidar_index] = match.class_id

    def _mark_preempted_guesses(self, enemy_classes: Set[int], lidar_points, camera_lidar_indices, used_classes) -> None:
        if not camera_lidar_indices:
            return
        gate = float(self.history_cfg["max_distance_m"])
        for class_id in enemy_classes:
            track = self.tracks[class_id]
            if class_id in used_classes or track.identity_state != IdentityState.GUESSED or not track.kalman_initialized:
                continue
            distances = [(self._distance(track.kalman_filter_2d.get_position(), point), index) for index, point in enumerate(lidar_points)]
            if not distances:
                continue
            distance, closest = min(distances)
            if distance <= gate and closest in camera_lidar_indices:
                track.preempted_this_frame = True

    def _match_enemy_history_to_lidar(self, enemy_classes, lidar_points, used_classes, used_lidar, now) -> None:
        rows = [class_id for class_id in sorted(enemy_classes) if class_id not in used_classes and self.tracks[class_id].kalman_initialized]
        columns = [index for index in range(len(lidar_points)) if index not in used_lidar]
        if not rows or not columns:
            return
        gate = float(self.history_cfg["max_distance_m"])
        costs = np.zeros((len(rows), len(columns)), dtype=np.float64)
        valid = np.zeros_like(costs, dtype=bool)
        for row_index, class_id in enumerate(rows):
            predicted = self.tracks[class_id].kalman_filter_2d.get_position()
            for col_index, lidar_index in enumerate(columns):
                distance = self._distance(predicted, lidar_points[lidar_index])
                costs[row_index, col_index] = distance / gate
                valid[row_index, col_index] = distance <= gate
        for row_index, col_index, _cost in self._solve_partial_rows(costs, valid):
            if col_index is None:
                continue
            class_id = rows[row_index]
            lidar_index = columns[col_index]
            track = self.tracks[class_id]
            self._set_measurement_output(track, lidar_points[lidar_index], FrameSource.LIDAR_HISTORY, now, r_std=float(self.position_cfg["lidar_r_std_m"]))
            track.last_lidar_frame = self.frame_count
            used_classes.add(class_id)
            used_lidar.add(lidar_index)
            # 猜测身份即使依靠历史位置续接到雷达，仍按“猜点”显示为绿色；
            # 只有已经确认身份的历史轨迹才能给雷达点染成阵营色。
            if track.identity_state == IdentityState.CONFIRMED:
                self._matched_lidar_classes[lidar_index] = class_id

    def _match_ambiguous_gray(self, cars, lidar_points, used_classes, used_cars, used_lidar, ally_classes, enemy_classes, now) -> None:
        ray_gate = float(self.gray_cfg["max_ray_lidar_distance_m"])
        track_gate = float(self.history_cfg["max_distance_m"])
        base_candidates: Dict[Tuple[int, int, int], float] = {}
        metadata = {}
        for car in cars:
            selected = car.selected_armor
            if car.car_index in used_cars or selected is None or selected.color != ArmorColor.GRAY or car.ray_uwb_2d is None:
                continue
            if selected.combined_confidence < float(self.config["digit_detector"]["gray_anchor_min_combined_confidence"]):
                continue
            for class_id in car.semantic_candidates:
                track = self.tracks[class_id]
                if class_id in used_classes or track.identity_state not in (IdentityState.UNSEEN, IdentityState.GUESSED):
                    continue
                # 允许没有历史的己方 UNSEEN 候选参加红蓝灰色全局判定；但禁止旧己方
                # GUESSED/KF 轨迹在 0x02BB 过期后继续占用雷达点。
                if class_id in ally_classes and track.kalman_initialized:
                    continue
                for lidar_index, point in enumerate(lidar_points):
                    if lidar_index in used_lidar:
                        continue
                    ray_distance = self._distance(car.ray_uwb_2d, point)
                    if ray_distance > ray_gate:
                        continue
                    if track.identity_state == IdentityState.UNSEEN or not track.kalman_initialized:
                        track_cost = float(self.gray_cfg["unseen_track_cost"])
                    else:
                        track_distance = self._distance(track.kalman_filter_2d.get_position(), point)
                        if track_distance > track_gate:
                            continue
                        track_cost = track_distance / track_gate
                    cost = (
                        float(self.gray_cfg["semantic_cost_weight"]) * (1.0 - selected.combined_confidence)
                        + float(self.gray_cfg["track_distance_cost_weight"]) * track_cost
                        + float(self.gray_cfg["ray_distance_cost_weight"]) * ray_distance / ray_gate
                    )
                    key = (class_id, car.car_index, lidar_index)
                    base_candidates[key] = cost
                    metadata[key] = car
        candidates = []
        margin = float(self.gray_cfg["side_cost_margin"])
        bias = float(self.gray_cfg["enemy_tie_bias"])
        for key, base_cost in base_candidates.items():
            class_id, car_index, lidar_index = key
            opposite = class_id + 5 if class_id < 5 else class_id - 5
            opposite_cost = base_candidates.get((opposite, car_index, lidar_index))
            total = base_cost
            if class_id in enemy_classes and opposite_cost is not None and abs(base_cost - opposite_cost) <= margin:
                total -= bias
            candidates.append(TripletCandidate(class_id, car_index, lidar_index, total))
        for match in self._solve_disjoint_triplets(candidates):
            track = self.tracks[match.class_id]
            car = metadata[(match.class_id, match.car_index, match.lidar_index)]
            track.identity_state = IdentityState.GUESSED
            self._set_measurement_output(track, lidar_points[match.lidar_index], FrameSource.GRAY_RADAR_GUESSED, now, hard_reset=True)
            track.associated_bot_id = car.bot_id
            track.last_car_box = list(car.car_box)
            track.last_gray_camera_frame = self.frame_count
            track.last_lidar_frame = self.frame_count
            used_classes.add(match.class_id)
            used_cars.add(match.car_index)
            used_lidar.add(match.lidar_index)

    def _assign_random_radar_guesses(self, enemy_classes, lidar_points, used_classes, used_lidar, now) -> None:
        available = [index for index in range(len(lidar_points)) if index not in used_lidar]
        if not available:
            return
        radar_after = int(self.loss_cfg["radar_guess_after_frames"])
        role_rank = {role: index for index, role in enumerate(self.guessing_cfg["role_priority"])}
        eligible = []
        for class_id in enemy_classes:
            if class_id in used_classes:
                continue
            track = self.tracks[class_id]
            if track.preempted_this_frame:
                continue
            if track.identity_state == IdentityState.UNSEEN:
                group = 0
            elif track.continuous_loss_count + 1 >= radar_after:
                group = 1
            else:
                continue
            eligible.append((group, -track.continuous_loss_count, role_rank[self.ROLE_BY_OFFSET[class_id % 5]], class_id))
        if not eligible:
            return

        # 先按原有优先级决定本帧有资格获得雷达点的 class，再全局分配雷达点。
        # 红方 class(5~9) 优先 x<=半场线，蓝方 class(0~4) 优先 x>=半场线；
        # 当本方半场没有足够点时仍允许使用另一半场，保证猜点数量不会下降。
        selected = sorted(eligible)[: len(available)]
        field_mid_x = float(self.lidar_config["field_length"]) * 0.5
        costs = np.zeros((len(selected), len(available)), dtype=np.float64)
        tie_break = self.rng.random(costs.shape) * 1e-6
        for row_index, (_group, _loss, _role, class_id) in enumerate(selected):
            is_blue = class_id < 5
            for column_index, lidar_index in enumerate(available):
                point_x = float(lidar_points[lidar_index][0])
                in_preferred_half = (
                    point_x >= field_mid_x if is_blue else point_x <= field_mid_x
                )
                costs[row_index, column_index] = (
                    0.0 if in_preferred_half else 1.0
                ) + tie_break[row_index, column_index]

        rows, columns = linear_sum_assignment(costs)
        for row_index, column_index in zip(rows.tolist(), columns.tolist()):
            _group, _loss, _role, class_id = selected[row_index]
            lidar_index = available[column_index]
            track = self.tracks[class_id]
            track.identity_state = IdentityState.GUESSED
            self._set_measurement_output(track, lidar_points[lidar_index], FrameSource.RADAR_GUESSED, now, hard_reset=True)
            track.last_lidar_frame = self.frame_count
            used_classes.add(class_id)
            used_lidar.add(lidar_index)

    def _handle_unmatched_tracks(self, enemy_classes: Set[int], used_classes: Set[int]) -> None:
        fixed_after = int(self.loss_cfg["fixed_guess_after_frames"])
        radar_after = int(self.loss_cfg["radar_guess_after_frames"])
        for track in self.tracks:
            if track.class_id in used_classes:
                continue
            if track.class_id not in enemy_classes:
                track.spatial_state = SpatialState.UNINITIALIZED if not track.kalman_initialized else SpatialState.LOST
                track.frame_source = FrameSource.INVALID
                track.output_valid = False
                track.output_position_2d_m = None
                continue
            pending = track.unseen_frame_count + 1 if track.identity_state == IdentityState.UNSEEN else track.continuous_loss_count + 1
            if pending >= fixed_after:
                point = self.point_guesser.predict_points(
                    track,
                    ref_color="blue" if self.faction == FACTION.BLUE else "red",
                )
                if point is None:
                    track.frame_source = FrameSource.INVALID
                    track.output_valid = False
                    track.output_position_2d_m = None
                    continue
                track.fixed_guess_point = [float(point[0]), float(point[1])]
                track.spatial_state = SpatialState.LOST
                track.frame_source = FrameSource.FIXED_GUESSED
                track.output_valid = True
                track.output_position_2d_m = list(track.fixed_guess_point)
                track.pos_3d = [track.fixed_guess_point[0], track.fixed_guess_point[1], 0.0]
                continue
            if track.kalman_initialized:
                track.spatial_state = SpatialState.COASTING if pending < radar_after else SpatialState.LOST
                track.frame_source = FrameSource.KF_PREDICTED
                track.output_valid = True
                track.output_position_2d_m = track.kalman_filter_2d.get_position()
                track.pos_3d = [track.output_position_2d_m[0], track.output_position_2d_m[1], 0.0]
            else:
                track.spatial_state = SpatialState.UNINITIALIZED
                track.frame_source = FrameSource.INVALID
                track.output_valid = False
                track.output_position_2d_m = None

    def _finalize_outputs(self) -> None:
        lidar_sources = {
            FrameSource.CAMERA_LIDAR,
            FrameSource.GRAY_CAMERA_LIDAR,
            FrameSource.ALLY_LIDAR,
            FrameSource.LIDAR_HISTORY,
            FrameSource.GRAY_RADAR_GUESSED,
            FrameSource.RADAR_GUESSED,
        }
        loss_reset_sources = lidar_sources | {FrameSource.ALLY_FEEDBACK}
        for track in self.tracks:
            source = track.frame_source
            track.lidar_miss_count = 0 if source in lidar_sources else track.lidar_miss_count + 1
            if source in loss_reset_sources:
                track.continuous_loss_count = 0
            elif track.identity_state != IdentityState.UNSEEN:
                track.continuous_loss_count += 1
            if track.identity_state == IdentityState.UNSEEN and source not in lidar_sources:
                track.unseen_frame_count += 1
            elif track.identity_state != IdentityState.UNSEEN:
                track.unseen_frame_count = 0
            if track.output_position_2d_m is not None:
                point = track.output_position_2d_m
                track.output_valid = track.output_valid and all(math.isfinite(float(value)) for value in point[:2])

    def process_observations(
        self,
        cars: Sequence[CarObservation],
        lidar_snapshot=None,
        ally_snapshot=None,
        faction: FACTION = FACTION.RED,
        now_monotonic: Optional[float] = None,
    ) -> List[TrackingState]:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        effective_faction = self._effective_faction(faction)
        self.faction = effective_faction
        if self._effective_faction_last is not None and effective_faction != self._effective_faction_last:
            self._reset_spatial_for_faction_change()
        self._effective_faction_last = effective_faction
        self.frame_count += 1
        if self.last_frame_monotonic is None:
            dt = float(self.position_cfg["default_dt_sec"])
        else:
            dt = min(float(self.position_cfg["max_dt_sec"]), max(float(self.position_cfg["min_dt_sec"]), now - self.last_frame_monotonic))
        self.last_frame_monotonic = now
        self.real_dt = dt
        self._matched_lidar_classes = {}
        stage_started = time.perf_counter()
        lidar_points = self._prepare_frame_inputs(lidar_snapshot, effective_faction, now)
        self.latest_lidar_points = tuple(lidar_points)
        self._record_timing_stat(self.track_timing_stats, "10.1 地面雷达输入快照整理（原子模块10.1）", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        self._predict_position_filters(dt)
        self._record_timing_stat(self.track_timing_stats, "10.2 全目标二维卡尔曼预测（原子模块10.2）", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        self._update_bot_histories(cars)
        self._record_timing_stat(self.track_timing_stats, "10.3 BoT-ID语义历史更新（原子模块10.3）", time.perf_counter() - stage_started)
        ally_classes, enemy_classes = self._faction_sets(effective_faction)
        used_classes: Set[int] = set()
        used_cars: Set[int] = set()
        used_lidar: Set[int] = set()
        cars_by_index = {car.car_index: car for car in cars}

        stage_started = time.perf_counter()
        self._match_fresh_ally_to_lidar(ally_snapshot, effective_faction, lidar_points, used_classes, used_lidar, now)
        self._record_timing_stat(self.track_timing_stats, "10.4 己方哨兵反馈匹配与雷达点剔除（原子模块10.4）", time.perf_counter() - stage_started)
        statistics_lidar_indices = tuple(
            index for index in range(len(lidar_points)) if index not in used_lidar
        )
        statistics_camera_points = tuple(
            (car.car_index, car.ray_uwb_2d)
            for car in cars
            if car.ray_uwb_2d is not None
        )
        stage_started = time.perf_counter()
        colored = self._solve_disjoint_triplets(self._build_colored_triplets(cars, lidar_points, used_classes, used_cars, used_lidar))
        camera_lidar_indices = self._commit_colored_matches(colored, cars_by_index, lidar_points, used_classes, used_cars, used_lidar, now)
        self._observe_projection_lidar_matches(
            colored,
            FrameSource.CAMERA_LIDAR,
            cars_by_index,
            lidar_points,
            statistics_lidar_indices,
            statistics_camera_points,
            now,
        )
        self._record_timing_stat(self.track_timing_stats, "10.5 彩色相机雷达三元组匹配（原子模块10.5）", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        gray_maintenance = self._solve_disjoint_triplets(
            self._build_gray_maintenance_triplets(cars, lidar_points, used_classes, used_cars, used_lidar, ally_classes)
        )
        self._commit_gray_maintenance_matches(gray_maintenance, cars_by_index, lidar_points, used_classes, used_cars, used_lidar, now)
        self._observe_projection_lidar_matches(
            gray_maintenance,
            FrameSource.GRAY_CAMERA_LIDAR,
            cars_by_index,
            lidar_points,
            statistics_lidar_indices,
            statistics_camera_points,
            now,
        )
        self._record_timing_stat(self.track_timing_stats, "10.6 灰色装甲身份维持匹配（原子模块10.6）", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        self._mark_preempted_guesses(enemy_classes, lidar_points, camera_lidar_indices, used_classes)
        self._record_timing_stat(self.track_timing_stats, "10.7 旧猜测抢占标记（原子模块10.7）", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        self._match_enemy_history_to_lidar(enemy_classes, lidar_points, used_classes, used_lidar, now)
        self._record_timing_stat(self.track_timing_stats, "10.8 历史轨迹雷达匈牙利匹配（原子模块10.8）", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        self._match_ambiguous_gray(cars, lidar_points, used_classes, used_cars, used_lidar, ally_classes, enemy_classes, now)
        self._record_timing_stat(self.track_timing_stats, "10.9 灰色装甲模糊猜测匹配（原子模块10.9）", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        self._assign_random_radar_guesses(enemy_classes, lidar_points, used_classes, used_lidar, now)
        self._record_timing_stat(self.track_timing_stats, "10.10 纯雷达目标猜测（原子模块10.10）", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        self._handle_unmatched_tracks(enemy_classes, used_classes)
        self._record_timing_stat(self.track_timing_stats, "10.11 未匹配目标丢失与固定点处理（原子模块10.11）", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        self._finalize_outputs()
        # 只导出真实匹配关系；灰甲猜测、纯雷达猜测等不会写入该映射。
        self.latest_lidar_matches = dict(self._matched_lidar_classes)
        self._record_timing_stat(self.track_timing_stats, "10.12 跟踪状态输出有效性整理（原子模块10.12）", time.perf_counter() - stage_started)
        return self.tracks

    def track(
        self,
        img: np.ndarray,
        lidar_snapshot=None,
        ally_snapshot=None,
        faction: FACTION = FACTION.RED,
        radar_data: Optional[Sequence[Sequence[float]]] = None,
    ) -> tuple[List[TrackingState], Optional[np.ndarray], Optional[np.ndarray]]:
        track_started = time.perf_counter()
        with self._timing_lock:
            self.track_frame_timing_ms = {}
            self.detector_frame_timing_ms = {}
        now = time.monotonic()
        if lidar_snapshot is None and radar_data is not None:
            lidar_snapshot = SimpleNamespace(points_xy_m=tuple(tuple(point[:2]) for point in radar_data), received_monotonic=now, ros_stamp_sec=None)

        stage = time.perf_counter()
        detections, detect_vis_img = self.detect(img)
        self._record_timing_stat(self.track_timing_stats, "车辆与装甲检测", time.perf_counter() - stage)
        self.latest_detections = list(detections)
        effective_faction = self._effective_faction(faction)
        stage = time.perf_counter()
        cars = self._build_car_observations(detections, effective_faction)
        self._record_timing_stat(self.track_timing_stats, "观测预处理与投影", time.perf_counter() - stage)

        state_started = time.perf_counter()
        self.process_observations(cars, lidar_snapshot, ally_snapshot, faction=effective_faction, now_monotonic=now)
        self._record_timing_stat(self.track_timing_stats, "匹配状态机总计", time.perf_counter() - state_started)

        self._record_timing_stat(self.track_timing_stats, "追踪总计", time.perf_counter() - track_started)
        return self.tracks, detect_vis_img, None

    def warmup(self, warmup_num: int = 20, img_path: str = "./tracker/warmup_image.jpg") -> None:
        image = cv2.imread(img_path)
        if image is None:
            logger.warning("预热图像不存在，跳过 tracker 预热: %s", img_path)
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        xy_map_manager = self.xy_map_manager
        self.xy_map_manager = None
        try:
            for _ in range(max(0, int(warmup_num))):
                self.detect(image)
        finally:
            self.xy_map_manager = xy_map_manager

    def visualize_results(self, img: np.ndarray, detections) -> np.ndarray:
        source_height, source_width = img.shape[:2]
        scale = min(1.0, self.visualization_max_width / float(source_width), self.visualization_max_height / float(source_height))
        if scale < 1.0:
            vis_img = cv2.resize(img, (max(1, int(round(source_width * scale))), max(1, int(round(source_height * scale)))), interpolation=cv2.INTER_AREA)
        else:
            vis_img = img.copy()
        thickness = max(1, int(round(self.vis_box_thickness * scale)))
        font_scale = max(0.35, self.vis_label_font_scale * scale)
        for detection in detections:
            box = [int(round(value * scale)) for value in detection.car_box]
            cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness)
            cv2.putText(vis_img, f"Car {detection.car_conf:.2f} ID:{detection.bot_id}", (box[0], max(10, box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        for track in self.tracks:
            if not track.output_valid or track.last_car_box is None:
                continue
            box = [int(round(value * scale)) for value in track.last_car_box]
            color = (0, 0, 255) if track.class_id < 5 else (255, 0, 0)
            cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), color, thickness)
            label = f"{track.name} {track.identity_state.value}/{track.frame_source.value}"
            cv2.putText(vis_img, label, (box[0], max(10, box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return vis_img
