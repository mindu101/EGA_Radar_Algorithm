# -*- coding: utf-8 -*-
"""异步叠图线程的低依赖回归测试。"""

import time
import unittest
from types import SimpleNamespace

import numpy as np

from utils.async_visualization import AsyncTrackingVisualizer, snapshot_detections


def _config(enabled=True, update_hz=100.0):
    return {
        "visualization": {
            "enabled": enabled,
            "update_hz": update_hz,
            "output_max_width": 320,
            "output_max_height": 240,
            "box_thickness": 2,
            "label_font_scale": 0.5,
            "label_font_thickness": 1,
            "label_offset_y": 8,
        },
        "debug": {
            "timing_debug_enabled": True,
            "timing_ema_alpha": 0.2,
        },
    }


def _detection(bot_id=7):
    selected_armor = SimpleNamespace(
        armor_confidence=0.82,
        digit_confidence=0.93,
        combined_confidence=0.90,
    )
    return SimpleNamespace(
        car_box=[50, 30, 250, 180],
        car_conf=0.91,
        bot_id=bot_id,
        class_id=2,
        class_conf=0.90,
        armor_box=[30, 40, 100, 90],
        selected_armor=selected_armor,
    )


def _track():
    return SimpleNamespace(
        is_active=True,
        car_box=[60, 35, 245, 175],
        class_id=2,
        name="B3",
        bot_id=7,
        state=SimpleNamespace(value="confirmed"),
    )


def _new_state_machine_track():
    return SimpleNamespace(
        output_valid=True,
        last_car_box=[60, 35, 245, 175],
        class_id=7,
        name="R3",
        identity_state=SimpleNamespace(value="confirmed"),
        frame_source=SimpleNamespace(value="camera_lidar"),
    )


def _wait_for_sequence(visualizer, minimum_sequence, timeout_sec=1.0):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        image, stamp, sequence = visualizer.get_latest()
        if sequence >= minimum_sequence:
            return image, stamp, sequence
        time.sleep(0.005)
    raise AssertionError(f"异步叠图未在 {timeout_sec:.1f}s 内完成")


class AsyncTrackingVisualizerTest(unittest.TestCase):
    def test_armor_confidences_are_snapshotted(self):
        snapshot = snapshot_detections([_detection()])[0]
        self.assertEqual(snapshot.armor_confidence, 0.82)
        self.assertEqual(snapshot.digit_confidence, 0.93)
        self.assertEqual(snapshot.combined_confidence, 0.90)

    def test_render_result_is_downsampled_and_read_only(self):
        published = []
        visualizer = AsyncTrackingVisualizer(
            _config(),
            on_image_ready=lambda image, stamp, sequence: published.append(
                (image, stamp, sequence)
            ),
        )
        self.addCleanup(visualizer.close)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertTrue(
            visualizer.submit(frame, [_detection()], [_track()], source_stamp=12.5)
        )
        image, stamp, sequence = _wait_for_sequence(visualizer, 1)

        self.assertEqual(image.shape, (240, 320, 3))
        self.assertFalse(image.flags.writeable)
        self.assertGreater(int(image.max()), 0)
        self.assertEqual(stamp, 12.5)
        self.assertEqual(sequence, 1)
        self.assertEqual(len(published), 1)
        self.assertIs(published[0][0], image)

        stats = visualizer.get_timing_stats()
        self.assertIn("后台合并可视化", stats)
        self.assertGreaterEqual(stats["后台合并可视化"]["last_ms"], 0.0)

    def test_pending_tasks_are_overwritten_by_latest_frame(self):
        visualizer = AsyncTrackingVisualizer(_config(update_hz=20.0))
        self.addCleanup(visualizer.close)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        self.assertTrue(visualizer.submit(frame, [_detection(0)], [], source_stamp=0))
        _wait_for_sequence(visualizer, 1)

        for stamp in range(1, 21):
            self.assertTrue(
                visualizer.submit(
                    frame, [_detection(stamp)], [], source_stamp=stamp
                )
            )

        _, latest_stamp, _ = _wait_for_sequence(visualizer, 2)
        diagnostics = visualizer.get_diagnostics()
        self.assertEqual(latest_stamp, 20.0)
        self.assertGreater(diagnostics["overwritten"], 0)
        self.assertLess(diagnostics["rendered"], diagnostics["submitted"])

    def test_new_state_machine_track_fields_are_snapshotted(self):
        visualizer = AsyncTrackingVisualizer(_config())
        self.addCleanup(visualizer.close)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        self.assertTrue(
            visualizer.submit(frame, [], [_new_state_machine_track()], source_stamp=1)
        )
        image, _, _ = _wait_for_sequence(visualizer, 1)
        self.assertGreater(int(image.max()), 0)

    def test_disabled_visualizer_does_not_start_or_accept_tasks(self):
        visualizer = AsyncTrackingVisualizer(_config(enabled=False))
        self.addCleanup(visualizer.close)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        self.assertFalse(visualizer.submit(frame, [], [], source_stamp=1.0))
        self.assertEqual(visualizer.get_latest(), (None, None, 0))
        self.assertEqual(visualizer.get_diagnostics()["submitted"], 0)

    def test_close_is_idempotent_and_rejects_new_tasks(self):
        visualizer = AsyncTrackingVisualizer(_config())
        visualizer.close()
        visualizer.close()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertFalse(visualizer.submit(frame, [], [], source_stamp=1.0))


if __name__ == "__main__":
    unittest.main()
