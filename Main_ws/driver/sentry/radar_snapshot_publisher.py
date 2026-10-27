"""Publish a committed radar frame to the sentry decision ROS port."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


_ROLES = (1, 2, 3, 4, 7)
_MEASURED_SOURCES = {"camera_lidar", "gray_camera_lidar", "lidar_history"}


@dataclass(frozen=True)
class EnemySlot:
    robot_id: int = 0
    robot_type: int = 0
    known: bool = False
    visible: bool = False
    x_m: float = 0.0
    y_m: float = 0.0


def _enum_value(value) -> str:
    return str(getattr(value, "value", value)).lower()


def build_enemy_slots(completed) -> tuple[EnemySlot, ...]:
    """Use only identified, measured enemy positions from this committed frame."""
    faction = _enum_value(completed.faction)
    if faction not in ("red", "blue") or completed.error:
        return (EnemySlot(),) * 5

    first_class = 0 if faction == "red" else 5
    tracks = {int(track.class_id): track for track in completed.tracks}
    slots = []
    for slot, role in enumerate(_ROLES):
        track = tracks.get(first_class + slot)
        robot_id = (100 if faction == "red" else 0) + role
        position = getattr(track, "output_position_2d_m", None)
        measured = (
            track is not None
            and bool(track.output_valid)
            and _enum_value(track.identity_state) == "confirmed"
            and _enum_value(track.frame_source) in _MEASURED_SOURCES
            and position is not None
            and len(position) >= 2
            and all(math.isfinite(float(value)) for value in position[:2])
        )
        if measured:
            slots.append(
                EnemySlot(robot_id, role, True, True, float(position[0]), float(position[1]))
            )
        else:
            slots.append(EnemySlot(robot_id=robot_id, robot_type=role))
    return tuple(slots)


def _frame_age_ms(completed, now_wall_s: float) -> float:
    processing_age = max(0.0, float(completed.processing_age_ms))
    input_age = getattr(completed, "input_frame_age_ms", None)
    if input_age is not None and math.isfinite(float(input_age)) and input_age >= 0.0:
        processing_age = max(processing_age, float(input_age))
    try:
        camera_age = (now_wall_s - float(completed.frame_stamp)) * 1000.0
    except (TypeError, ValueError):
        return processing_age
    if not math.isfinite(camera_age) or camera_age < -100.0 or camera_age > 3600000.0:
        return processing_age
    return max(processing_age, camera_age)


def _source_stamp_s(completed, now_wall_s: float, source_age_ms: float) -> float:
    try:
        stamp_s = float(completed.frame_stamp)
        camera_age_ms = (now_wall_s - stamp_s) * 1000.0
    except (TypeError, ValueError):
        camera_age_ms = -1.0
    if math.isfinite(camera_age_ms) and -100.0 <= camera_age_ms <= 3600000.0:
        return stamp_s
    return now_wall_s - source_age_ms / 1000.0


class SentryRadarSnapshotPublisher:
    """ROS publisher; construction fails clearly when the interface is not built."""

    def __init__(self, topic: str, frame_id: str):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rm2026_vision_interface.msg import (
            SentryRadarEnemy,
            SentryRadarMate,
            SentryRadarOutpost,
            SentryRadarSnapshot,
        )

        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("rm2026_sentry_radar_pub")
        self._publisher = self._node.create_publisher(
            SentryRadarSnapshot,
            topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._frame_id = frame_id
        self._Enemy = SentryRadarEnemy
        self._Mate = SentryRadarMate
        self._Outpost = SentryRadarOutpost
        self._Snapshot = SentryRadarSnapshot

    def publish(self, completed) -> None:
        now_wall_s = time.time()
        message = self._Snapshot()
        message.header.frame_id = self._frame_id
        source_age_ms = _frame_age_ms(completed, now_wall_s)
        stamp_s = _source_stamp_s(completed, now_wall_s, source_age_ms)
        message.header.stamp.sec = int(stamp_s)
        message.header.stamp.nanosec = int((stamp_s % 1.0) * 1e9)
        message.pipeline_epoch = int(completed.pipeline_epoch)
        message.sequence = int(completed.commit_sequence)
        faction = _enum_value(completed.faction)
        message.faction = {
            "red": self._Snapshot.FACTION_RED,
            "blue": self._Snapshot.FACTION_BLUE,
        }.get(faction, self._Snapshot.FACTION_UNKNOWN)
        message.source_age_ms = source_age_ms
        message.enemies = [
            self._Enemy(
                robot_id=slot.robot_id,
                robot_type=slot.robot_type,
                known=slot.known,
                visible=slot.visible,
                x_m=slot.x_m,
                y_m=slot.y_m,
                alive_known=False,
                alive=True,
            )
            for slot in build_enemy_slots(completed)
        ]
        message.mates = [self._Mate(known=False) for _ in range(5)]
        message.outposts = [self._Outpost(known=False, hp=-1) for _ in range(2)]
        message.kills = -1
        self._publisher.publish(message)

    def stop(self) -> None:
        self._node.destroy_node()
