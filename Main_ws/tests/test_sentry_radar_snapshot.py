from types import SimpleNamespace

import pytest

from driver.sentry.radar_snapshot_publisher import (
    SentryRadarSnapshotPublisher,
    _frame_age_ms,
    build_enemy_slots,
)


def frame(faction, tracks=(), error=None):
    return SimpleNamespace(
        faction=faction,
        tracks=tracks,
        error=error,
        processing_age_ms=40.0,
        input_frame_age_ms=None,
        frame_stamp=100.0,
        pipeline_epoch=2,
        commit_sequence=42,
    )


def track(class_id, source="camera_lidar", identity="confirmed", position=(5.5, 6.0)):
    return SimpleNamespace(
        class_id=class_id,
        frame_source=source,
        identity_state=identity,
        output_valid=True,
        output_position_2d_m=position,
    )


def test_red_maps_blue_official_ids_and_keeps_metres():
    slots = build_enemy_slots(frame("red", (track(0), track(4, position=(7.0, 8.0)))))
    assert [slot.robot_id for slot in slots] == [101, 102, 103, 104, 107]
    assert [slot.robot_type for slot in slots] == [1, 2, 3, 4, 7]
    assert slots[0].known and slots[0].visible
    assert (slots[0].x_m, slots[0].y_m) == (5.5, 6.0)
    assert not slots[1].known
    assert slots[4].known


def test_blue_maps_red_official_ids():
    slots = build_enemy_slots(frame("blue", (track(5), track(9))))
    assert [slot.robot_id for slot in slots] == [1, 2, 3, 4, 7]
    assert slots[0].known and slots[4].known


def test_guesses_predictions_and_invalid_positions_are_unknown():
    tracks = (
        track(0, source="radar_guessed"),
        track(1, source="kf_predicted"),
        track(2, identity="guessed"),
        track(3, position=(float("nan"), 1.0)),
        track(4, source="lidar_history"),
    )
    slots = build_enemy_slots(frame("red", tracks))
    assert [slot.known for slot in slots] == [False, False, False, False, True]
    assert [slot.visible for slot in slots] == [False, False, False, False, True]


def test_unknown_faction_or_failed_frame_has_no_known_targets():
    assert not any(slot.known for slot in build_enemy_slots(frame("unknown", (track(0),))))
    assert not any(slot.known for slot in build_enemy_slots(frame("red", (track(0),), "error")))


def test_source_age_includes_processing_and_camera_delay():
    assert _frame_age_ms(frame("red"), 100.1) >= 99.9
    assert _frame_age_ms(frame("red"), 100.01) == 40.0
    assert _frame_age_ms(frame("red"), 5000.0) == 40.0
    delayed = frame("red")
    delayed.input_frame_age_ms = 75.0
    assert _frame_age_ms(delayed, 5000.0) == 75.0


def test_ros_snapshot_contains_fixed_slots_and_unknown_truth():
    messages = pytest.importorskip("rm2026_vision_interface.msg")
    publisher = object.__new__(SentryRadarSnapshotPublisher)
    publisher._frame_id = "radar_field"
    publisher._Enemy = messages.SentryRadarEnemy
    publisher._Mate = messages.SentryRadarMate
    publisher._Outpost = messages.SentryRadarOutpost
    publisher._Snapshot = messages.SentryRadarSnapshot
    captured = []
    publisher._publisher = SimpleNamespace(publish=captured.append)

    publisher.publish(frame("red", (track(0),)))

    message = captured[0]
    assert message.header.frame_id == "radar_field"
    assert (message.pipeline_epoch, message.sequence) == (2, 42)
    assert message.faction == message.FACTION_RED
    assert len(message.enemies) == 5
    assert message.enemies[0].robot_id == 101 and message.enemies[0].known
    assert not any(mate.known for mate in message.mates)
    assert not any(outpost.known for outpost in message.outposts)
    assert message.kills == -1
