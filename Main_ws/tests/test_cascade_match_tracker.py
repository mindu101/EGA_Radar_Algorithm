from __future__ import annotations

import copy
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml

from driver.referee.referee_comm import FACTION
from tracker.CascadeMatchTracker.detector import BaseDetector
from tracker.CascadeMatchTracker.tracker import CascadeMatchTracker
from tracker.CascadeMatchTracker.type import (
    ArmorColor,
    ArmorObservation,
    CarObservation,
    FrameSource,
    IdentityState,
    ProjectionMode,
)
from transform.xy_map import XYMapManager


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with (ROOT / "config/params.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def lidar(points, received=10.0):
    return SimpleNamespace(
        points_xy_m=tuple(tuple(point) for point in points),
        ros_stamp_sec=None,
        received_monotonic=received,
    )


def ally(positions=None, received=10.0):
    values = {
        "hero": None,
        "engineer": None,
        "standard_3": None,
        "standard_4": None,
        "sentry": None,
    }
    values.update(positions or {})
    return SimpleNamespace(positions_m=values, received_monotonic=received)


def car(index, raw_class_id, ray, bot_id=None, confidence=0.9):
    if raw_class_id < 5:
        color = ArmorColor.BLUE
        digit = raw_class_id
        candidates = (raw_class_id,)
    elif raw_class_id < 10:
        color = ArmorColor.RED
        digit = raw_class_id - 5
        candidates = (raw_class_id,)
    else:
        color = ArmorColor.GRAY
        digit = raw_class_id - 10
        candidates = (raw_class_id - 10, raw_class_id - 5)
    armor = ArmorObservation(
        armor_box=(1.0, 1.0, 10.0, 8.0),
        armor_confidence=confidence,
        digit_index=digit,
        digit_confidence=confidence,
        color=color,
        raw_class_id=raw_class_id,
        combined_confidence=confidence,
    )
    ray_tuple = None if ray is None else tuple(ray)
    return CarObservation(
        car_index=index,
        car_box=(index * 100.0, 0.0, index * 100.0 + 80.0, 100.0),
        car_confidence=0.9,
        bot_id=index + 1 if bot_id is None else bot_id,
        armor_observations=(armor,),
        selected_armor=armor,
        semantic_candidates=candidates,
        projection_mode=ProjectionMode.DIRECT,
        ray_world_3d=None if ray_tuple is None else (ray_tuple[0], ray_tuple[1], 0.0),
        ray_uwb_2d=ray_tuple,
    )


class CascadeStateMachineTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.tracker = CascadeMatchTracker.create_for_test(copy.deepcopy(self.config))

    def process(self, cars=(), points=(), ally_snapshot=None, now=10.0, faction=FACTION.RED, lidar_received=None):
        received = now if lidar_received is None else lidar_received
        return self.tracker.process_observations(
            list(cars),
            lidar(points, received=received),
            ally_snapshot,
            faction=faction,
            now_monotonic=now,
        )

    def test_colored_camera_and_lidar_confirms_identity(self):
        tracks = self.process([car(0, 0, (1.0, 1.0))], [(1.1, 1.0)])
        self.assertEqual(tracks[0].identity_state, IdentityState.CONFIRMED)
        self.assertEqual(tracks[0].frame_source, FrameSource.CAMERA_LIDAR)
        self.assertEqual(self.tracker.latest_lidar_matches, {0: 0})
        statistics = self.tracker.get_projection_lidar_statistics_snapshot()
        self.assertEqual(statistics["global"]["count"], 1)
        self.assertEqual(statistics["by_class"]["B1"]["count"], 1)

    def test_duplicate_colored_digit_uses_one_car(self):
        tracks = self.process(
            [car(0, 0, (1.0, 1.0)), car(1, 0, (5.0, 5.0))],
            [(1.0, 1.0), (5.0, 5.0)],
        )
        self.assertEqual(tracks[0].frame_source, FrameSource.CAMERA_LIDAR)
        matched_boxes = [track.last_car_box for track in tracks if track.frame_source == FrameSource.CAMERA_LIDAR]
        self.assertEqual(len(matched_boxes), 1)

    def test_joint_solver_skips_car_without_legal_lidar(self):
        tracks = self.process(
            [car(0, 0, (20.0, 10.0)), car(1, 0, (2.0, 2.0))],
            [(2.1, 2.0)],
        )
        self.assertEqual(tracks[0].associated_bot_id, 2)

    def test_bot_history_does_not_affect_colored_match_cost(self):
        # 先让 bot 1 的语义历史偏向 B2、bot 2 偏向 B1。
        for frame in range(3):
            self.process(
                [
                    car(0, 1, (1.0, 1.0), bot_id=1),
                    car(1, 0, (1.0, 1.0), bot_id=2),
                ],
                [],
                now=10.0 + frame * 0.1,
            )

        tracks = self.process(
            [
                car(0, 0, (1.0, 1.0), bot_id=1),
                car(1, 0, (1.0, 1.0), bot_id=2),
            ],
            [(1.0, 1.0)],
            now=10.3,
        )

        # 两个候选的语义与几何代价完全相同，应按 car_index 确定性选择，
        # 不再因为 bot 2 的历史更偏向 B1 而改变结果。
        self.assertEqual(tracks[0].associated_bot_id, 1)

    def test_camera_without_lidar_does_not_initialize_kf(self):
        tracks = self.process([car(0, 0, (1.0, 1.0))], [])
        self.assertFalse(tracks[0].kalman_initialized)
        self.assertEqual(tracks[0].frame_source, FrameSource.INVALID)

    def test_confirmed_history_matches_lidar(self):
        self.process([car(0, 0, (1.0, 1.0))], [(1.0, 1.0)], now=10.0)
        tracks = self.process([], [(1.2, 1.0)], now=10.1)
        self.assertEqual(tracks[0].frame_source, FrameSource.LIDAR_HISTORY)
        self.assertEqual(self.tracker.latest_lidar_matches, {0: 0})

    def test_history_gate_is_broken_before_hungarian(self):
        self.process([car(0, 0, (1.0, 1.0))], [(1.0, 1.0)], now=10.0)
        tracks = self.process([], [(10.0, 10.0)], now=10.1)
        self.assertNotEqual(tracks[0].frame_source, FrameSource.LIDAR_HISTORY)

    def test_unseen_enemy_immediately_guesses_remaining_radar(self):
        tracks = self.process([], [(2.0, 2.0)])
        guessed = [track for track in tracks[:5] if track.frame_source == FrameSource.RADAR_GUESSED]
        self.assertEqual(len(guessed), 1)
        self.assertEqual(self.tracker.latest_lidar_matches, {})

    def test_short_loss_uses_kf_prediction(self):
        self.process([car(0, 0, (1.0, 1.0))], [(1.0, 1.0)], now=10.0)
        tracks = self.process([], [], now=10.1)
        self.assertEqual(tracks[0].frame_source, FrameSource.KF_PREDICTED)

    def test_tenth_loss_can_randomly_reacquire_far_radar(self):
        self.process([car(0, 0, (1.0, 1.0))], [(1.0, 1.0)], now=10.0)
        for index in range(1, 10):
            self.process([], [], now=10.0 + index * 0.1)
        tracks = self.process(
            [],
            [(20.0, 10.0), (21.0, 10.0), (22.0, 10.0), (23.0, 10.0), (24.0, 10.0)],
            now=11.0,
        )
        self.assertEqual(tracks[0].frame_source, FrameSource.RADAR_GUESSED)
        self.assertEqual(tracks[0].identity_state, IdentityState.GUESSED)

    def test_thirtieth_loss_uses_fixed_point_without_kf_update(self):
        self.process([car(0, 0, (1.0, 1.0))], [(1.0, 1.0)], now=10.0)
        state_before = None
        for index in range(1, 31):
            if index == 30:
                state_before = self.tracker.tracks[0].kalman_filter_2d.kf.x.copy()
            tracks = self.process([], [], now=10.0 + index * 0.1)
        self.assertEqual(tracks[0].frame_source, FrameSource.FIXED_GUESSED)
        np.testing.assert_allclose(tracks[0].kalman_filter_2d.kf.x, state_before)

    def test_gray_tie_prefers_enemy(self):
        tracks = self.process([car(0, 10, (2.0, 2.0))], [(2.0, 2.0)])
        self.assertEqual(tracks[0].frame_source, FrameSource.GRAY_RADAR_GUESSED)
        self.assertNotEqual(tracks[5].frame_source, FrameSource.GRAY_RADAR_GUESSED)

    def test_two_gray_cars_are_solved_globally(self):
        tracks = self.process(
            [car(0, 10, (2.0, 2.0)), car(1, 10, (8.0, 8.0))],
            [(2.0, 2.0), (8.0, 8.0)],
        )
        matched = [track for track in (tracks[0], tracks[5]) if track.frame_source == FrameSource.GRAY_RADAR_GUESSED]
        self.assertEqual(len(matched), 2)

    def test_gray_does_not_force_invalid_second_edge(self):
        tracks = self.process(
            [car(0, 10, (2.0, 2.0)), car(1, 10, (20.0, 14.0))],
            [(2.0, 2.0)],
        )
        matched = [track for track in (tracks[0], tracks[5]) if track.frame_source == FrameSource.GRAY_RADAR_GUESSED]
        self.assertEqual(len(matched), 1)

    def test_gray_maintains_confirmed_without_colored_timestamp(self):
        self.process([car(0, 0, (2.0, 2.0))], [(2.0, 2.0)], now=10.0)
        colored_frame = self.tracker.tracks[0].last_camera_frame
        tracks = self.process([car(0, 10, (2.1, 2.0))], [(2.1, 2.0)], now=10.1)
        self.assertEqual(tracks[0].frame_source, FrameSource.GRAY_CAMERA_LIDAR)
        self.assertEqual(tracks[0].last_camera_frame, colored_frame)

    def test_fresh_ally_consumes_lidar_before_camera(self):
        tracks = self.process(
            [car(0, 0, (10.1, 10.0))],
            [(10.1, 10.0)],
            ally({"hero": (10.0, 10.0)}, received=10.0),
        )
        self.assertEqual(tracks[5].frame_source, FrameSource.ALLY_LIDAR)
        self.assertEqual(tracks[5].output_position_2d_m, [10.0, 10.0])
        self.assertEqual(tracks[0].frame_source, FrameSource.INVALID)

    def test_fresh_ally_without_near_lidar_does_not_consume_enemy_point(self):
        tracks = self.process(
            [car(0, 0, (2.0, 2.0))],
            [(2.0, 2.0)],
            ally({"hero": (10.0, 10.0)}, received=10.0),
        )
        self.assertEqual(tracks[5].frame_source, FrameSource.ALLY_FEEDBACK)
        self.assertEqual(tracks[0].frame_source, FrameSource.CAMERA_LIDAR)

    def test_ally_freshness_boundary(self):
        tracks = self.process([], [], ally({"hero": (10.0, 10.0)}, received=9.85), now=10.0)
        self.assertEqual(tracks[5].frame_source, FrameSource.ALLY_FEEDBACK)
        tracker = CascadeMatchTracker.create_for_test(copy.deepcopy(self.config))
        tracks = tracker.process_observations([], lidar([], 10.0), ally({"hero": (10.0, 10.0)}, received=9.85 - 1e-6), FACTION.RED, 10.0)
        self.assertEqual(tracks[5].frame_source, FrameSource.INVALID)

    def test_stale_ally_history_cannot_consume_radar_through_gray_maintenance(self):
        self.process([], [], ally({"hero": (5.0, 5.0)}, received=10.0), now=10.0)
        tracks = self.process([car(0, 10, (5.1, 5.0))], [(5.1, 5.0)], now=10.2)
        self.assertEqual(tracks[5].frame_source, FrameSource.INVALID)
        self.assertEqual(tracks[0].frame_source, FrameSource.GRAY_RADAR_GUESSED)

    def test_stale_ally_history_still_allows_current_colored_camera_lidar(self):
        self.process([], [], ally({"hero": (5.0, 5.0)}, received=10.0), now=10.0)
        tracks = self.process([car(0, 5, (5.1, 5.0))], [(5.1, 5.0)], now=10.2)
        self.assertEqual(tracks[5].frame_source, FrameSource.CAMERA_LIDAR)

    def test_stale_lidar_snapshot_is_empty(self):
        tracks = self.process([], [(2.0, 2.0)], now=10.0, lidar_received=9.79)
        self.assertFalse(any(track.output_valid for track in tracks[:5]))

    def test_faction_change_resets_projection_lidar_statistics(self):
        self.process([car(0, 0, (1.0, 1.0))], [(1.0, 1.0)], now=10.0)
        self.assertEqual(
            self.tracker.get_projection_lidar_statistics_snapshot()["global"]["count"],
            1,
        )
        self.process([], [], now=10.1, faction=FACTION.BLUE)
        snapshot = self.tracker.get_projection_lidar_statistics_snapshot()
        self.assertEqual(snapshot["global"]["count"], 0)
        self.assertEqual(snapshot["last_reset_reason"], "阵营变化")

    def test_camera_preemption_prevents_same_frame_chain_guess(self):
        self.process([], [(5.0, 5.0)], now=10.0)
        guessed = next(track for track in self.tracker.tracks[:5] if track.frame_source == FrameSource.RADAR_GUESSED)
        target_class = 1 if guessed.class_id != 1 else 2
        tracks = self.process([car(0, target_class, (5.0, 5.0))], [(5.0, 5.0)], now=10.1)
        self.assertEqual(tracks[guessed.class_id].frame_source, FrameSource.KF_PREDICTED)

    def test_random_assignment_is_reproducible(self):
        first = self.process([], [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
        first_result = [(track.class_id, track.output_position_2d_m) for track in first if track.frame_source == FrameSource.RADAR_GUESSED]
        other = CascadeMatchTracker.create_for_test(copy.deepcopy(self.config))
        second = other.process_observations([], lidar([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], 10.0), None, FACTION.RED, 10.0)
        second_result = [(track.class_id, track.output_position_2d_m) for track in second if track.frame_source == FrameSource.RADAR_GUESSED]
        self.assertEqual(first_result, second_result)

    def test_blue_robot_guess_prefers_blue_half(self):
        used_classes = {1, 2, 3, 4}
        used_lidar = set()
        points = [(2.0, 2.0), (20.0, 2.0)]
        self.tracker._assign_random_radar_guesses(
            set(range(5)), points, used_classes, used_lidar, now=10.0
        )
        self.assertEqual(self.tracker.tracks[0].output_position_2d_m, [20.0, 2.0])
        self.assertTrue(self.tracker.tracks[0].is_guessed_output)

    def test_red_robot_guess_prefers_red_half(self):
        used_classes = {5, 7, 8, 9}
        used_lidar = set()
        points = [(2.0, 2.0), (20.0, 2.0)]
        self.tracker._assign_random_radar_guesses(
            set(range(5, 10)), points, used_classes, used_lidar, now=10.0
        )
        self.assertEqual(self.tracker.tracks[6].output_position_2d_m, [2.0, 2.0])
        self.assertTrue(self.tracker.tracks[6].is_guessed_output)

    def test_guessed_identity_remains_marked_during_history_tracking(self):
        tracks = self.process([], [(20.0, 2.0)])
        guessed = next(
            track for track in tracks if track.frame_source == FrameSource.RADAR_GUESSED
        )
        tracks = self.process([], [(20.1, 2.0)], now=10.1)
        self.assertEqual(tracks[guessed.class_id].frame_source, FrameSource.LIDAR_HISTORY)
        self.assertTrue(tracks[guessed.class_id].is_guessed_output)
        self.assertEqual(self.tracker.latest_lidar_matches, {})

    def test_multiple_armor_colors_choose_weighted_best_instead_of_discarding_car(self):
        detector = BaseDetector.__new__(BaseDetector)
        detector.ignored_digit_index = 5
        detector.min_digit_confidence = 0.55
        detector.min_combined_confidence = 0.55
        weaker = ArmorObservation((1, 1, 5, 5), 0.95, 0, 0.60, ArmorColor.BLUE, 0, 0.705)
        stronger = ArmorObservation((6, 1, 10, 5), 0.80, 1, 0.90, ArmorColor.RED, 6, 0.87)
        self.assertIs(detector._select_armor((weaker, stronger)), stronger)

    def test_missing_config_reports_full_yaml_path(self):
        broken = copy.deepcopy(self.config)
        del broken["tracker"]["cascade"]["gray_match"]["side_cost_margin"]
        with self.assertRaisesRegex(ValueError, "tracker.cascade.gray_match.side_cost_margin"):
            CascadeMatchTracker.create_for_test(broken)

class FakeTensor:
    def __init__(self, value, dtype=None):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


class FakeScene:
    def __init__(self, fail=False):
        self.fail = fail

    def cast_rays(self, tensor):
        if self.fail:
            raise RuntimeError("synthetic cast failure")
        return {"t_hit": FakeTensor(np.ones(len(tensor.value), dtype=np.float32))}


class FakeO3D:
    class core:
        Tensor = FakeTensor

        class Dtype:
            Float32 = "float32"


class XYMapManagerTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.config["ray_projection"]["generation_chunk_rows"] = 2
        self.config["ray_projection"]["consistency_check"]["enabled"] = False

    @staticmethod
    def converter(fail=False):
        return SimpleNamespace(
            camera_matrix=np.eye(3, dtype=float),
            dist_coeffs=np.zeros(5, dtype=float),
            R=np.eye(3, dtype=float),
            T=np.zeros((3, 1), dtype=float),
            scene=FakeScene(fail=fail),
        )

    @staticmethod
    def wait_terminal(manager, timeout=2.0):
        deadline = time.monotonic() + timeout
        while manager.status not in (manager.STATUS_READY, manager.STATUS_FAILED) and time.monotonic() < deadline:
            time.sleep(0.01)

    def test_disabled_mode_uses_direct_projection(self):
        manager = XYMapManager(self.config)
        calls = []
        point, mode = manager.project_pixel(1.0, 2.0, (4, 3), lambda pixel: calls.append(pixel) or (1.0, 2.0, 3.0))
        self.assertEqual(mode, "direct")
        self.assertEqual(len(calls), 1)
        self.assertEqual(point, (1.0, 2.0, 3.0))
        manager.close()

    def test_checked_before_calibration_waits(self):
        manager = XYMapManager(self.config)
        manager.set_enabled(True)
        self.assertEqual(manager.status, manager.STATUS_WAITING)
        manager.close()

    @mock.patch("transform.xy_map._get_open3d", return_value=FakeO3D)
    def test_ready_map_uses_lookup_only(self, _patched):
        manager = XYMapManager(self.config)
        manager.set_frame_size(4, 3)
        manager.set_calibration(self.converter())
        manager.set_enabled(True)
        self.wait_terminal(manager)
        self.assertEqual(manager.status, manager.STATUS_READY)
        point, mode = manager.project_pixel(1.0, 1.0, (4, 3), lambda _pixel: self.fail("不应回退逐框投影"))
        self.assertEqual(mode, "xy_map")
        np.testing.assert_allclose(point[:2], (1.0, 1.0), atol=1e-6)
        manager.close()

    @mock.patch("transform.xy_map._get_open3d", return_value=FakeO3D)
    def test_invalid_ready_pixel_does_not_fallback(self, _patched):
        manager = XYMapManager(self.config)
        manager.set_frame_size(4, 3)
        manager.set_calibration(self.converter())
        manager.set_enabled(True)
        self.wait_terminal(manager)
        with manager._lock:
            manager._valid_mask[1, 1] = False
        point, mode = manager.project_pixel(1.0, 1.0, (4, 3), lambda _pixel: self.fail("无效映射不得回退"))
        self.assertIsNone(point)
        self.assertEqual(mode, "xy_map")
        manager.close()

    @mock.patch("transform.xy_map._get_open3d", return_value=FakeO3D)
    def test_generation_failure_disables_checkbox_state(self, _patched):
        with mock.patch("transform.xy_map.logger.exception"):
            manager = XYMapManager(self.config)
            manager.set_frame_size(4, 3)
            manager.set_calibration(self.converter(fail=True))
            manager.set_enabled(True)
            self.wait_terminal(manager)
        self.assertEqual(manager.status, manager.STATUS_FAILED)
        self.assertFalse(manager.enabled)
        manager.close()

    @mock.patch("transform.xy_map._get_open3d", return_value=FakeO3D)
    def test_unchecking_ready_map_immediately_restores_direct_projection(self, _patched):
        manager = XYMapManager(self.config)
        manager.set_frame_size(4, 3)
        manager.set_calibration(self.converter())
        manager.set_enabled(True)
        self.wait_terminal(manager)
        manager.set_enabled(False)
        calls = []
        point, mode = manager.project_pixel(1.0, 1.0, (4, 3), lambda pixel: calls.append(pixel) or (9.0, 8.0, 0.0))
        self.assertEqual(mode, "direct")
        self.assertEqual(point, (9.0, 8.0, 0.0))
        self.assertEqual(len(calls), 1)
        manager.close()

    @mock.patch("transform.xy_map._get_open3d", return_value=FakeO3D)
    def test_resolution_change_rebuilds_without_scaling_intrinsics(self, _patched):
        manager = XYMapManager(self.config)
        manager.set_frame_size(4, 3)
        manager.set_calibration(self.converter())
        manager.set_enabled(True)
        self.wait_terminal(manager)
        manager.set_frame_size(8, 6)
        self.wait_terminal(manager)
        self.assertEqual(manager.last_metrics["xy_map_width"], 8)
        point, mode = manager.project_pixel(2.0, 2.0, (8, 6), lambda _pixel: self.fail("应使用重建后的映射"))
        self.assertEqual(mode, "xy_map")
        np.testing.assert_allclose(point[:2], (2.0, 2.0), atol=1e-6)
        manager.close()

    @mock.patch("transform.xy_map._get_open3d", return_value=FakeO3D)
    def test_consistency_check_passes_after_required_real_car_samples(self, _patched):
        config = copy.deepcopy(self.config)
        config["ray_projection"]["consistency_check"].update(
            {"enabled": True, "sample_count": 2, "max_error_m": 0.05}
        )
        manager = XYMapManager(config)
        manager.set_frame_size(4, 3)
        manager.set_calibration(self.converter())
        manager.set_enabled(True)
        self.wait_terminal(manager)

        for _ in range(2):
            point, mode = manager.project_pixel(
                1.0,
                1.0,
                (4, 3),
                lambda _pixel: (1.0, 1.0, 0.0),
            )
            self.assertEqual(mode, "direct")
            np.testing.assert_allclose(point[:2], (1.0, 1.0), atol=1e-6)

        self.assertTrue(manager.enabled)
        self.assertEqual(manager.status, manager.STATUS_READY)
        self.assertIn("一致性检查通过", manager.status_text)
        events = manager.consume_consistency_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["level"], "info")
        self.assertAlmostEqual(events[0]["max_error_m"], 0.0)
        point, mode = manager.project_pixel(
            1.0,
            1.0,
            (4, 3),
            lambda _pixel: self.fail("检查通过后不应继续逐框投影"),
        )
        self.assertEqual(mode, "xy_map")
        np.testing.assert_allclose(point[:2], (1.0, 1.0), atol=1e-6)
        manager.close()

    @mock.patch("transform.xy_map._get_open3d", return_value=FakeO3D)
    def test_consistency_check_failure_disables_xy_maps(self, _patched):
        config = copy.deepcopy(self.config)
        config["ray_projection"]["consistency_check"].update(
            {"enabled": True, "sample_count": 2, "max_error_m": 0.05}
        )
        manager = XYMapManager(config)
        manager.set_frame_size(4, 3)
        manager.set_calibration(self.converter())
        manager.set_enabled(True)
        self.wait_terminal(manager)

        for _ in range(2):
            manager.project_pixel(
                1.0,
                1.0,
                (4, 3),
                lambda _pixel: (2.0, 2.0, 0.0),
            )

        self.assertFalse(manager.enabled)
        self.assertEqual(manager.status, manager.STATUS_CONSISTENCY_FAILED)
        events = manager.consume_consistency_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["level"], "warning")
        self.assertGreater(events[0]["max_error_m"], 0.05)
        manager.close()

    @mock.patch("transform.xy_map._get_open3d", return_value=FakeO3D)
    def test_recalibration_restarts_consistency_check(self, _patched):
        config = copy.deepcopy(self.config)
        config["ray_projection"]["consistency_check"].update(
            {"enabled": True, "sample_count": 1, "max_error_m": 0.05}
        )
        manager = XYMapManager(config)
        manager.set_frame_size(4, 3)
        manager.set_calibration(self.converter())
        manager.set_enabled(True)
        self.wait_terminal(manager)
        manager.project_pixel(
            1.0,
            1.0,
            (4, 3),
            lambda _pixel: (1.0, 1.0, 0.0),
        )
        self.assertIn("一致性检查通过", manager.status_text)

        recalibrated = self.converter()
        recalibrated.T[0, 0] = 1.0
        manager.set_calibration(recalibrated)
        self.wait_terminal(manager)
        self.assertIn("一致性检查 0/1", manager.status_text)
        manager.close()


if __name__ == "__main__":
    unittest.main()
