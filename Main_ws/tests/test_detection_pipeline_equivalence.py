from __future__ import annotations

import unittest
from threading import RLock

import numpy as np

from tracker.CascadeMatchTracker.detector import BaseDetector


class _FakeCarDetector:
    def __init__(self):
        self.last_timing_stats = {}

    def predict(self, image):
        self.last_timing_stats = {
            "车辆神经网络推理与BoT-SORT": 0.001,
            "车辆检测阶段总时间": 0.001,
        }
        return [(0, [2, 2, 18, 18], 0.8, 9)], None


class _FakeArmorDetector:
    def __init__(self):
        self.last_timing_stats = {}

    def predict_batch(self, images):
        self.last_timing_stats = {
            "装甲板位置神经网络推理与结果回传": 0.001,
            "数字神经网络推理与结果回传": 0.001,
        }
        return [[(1, [2, 2, 10, 8], 0.9, 0.8, 1)] for _ in images], None


class _FakeNumberedCarDetector(_FakeCarDetector):
    def predict(self, image):
        del image
        self.last_timing_stats = {
            "03 车辆检测与BoT-SORT（模块03总时间）": 0.001,
            "03.2.2 车辆神经网络TensorRT推理（原子模块03.2.2）": 0.0007,
        }
        return [(0, [2, 2, 18, 18], 0.8, 9)], None


class _FakeDetailedArmorDetector(_FakeArmorDetector):
    def predict_batch(self, images):
        detections, annotated = super().predict_batch(images)
        self.last_timing_stats.update(
            {
                "06 装甲位置检测（模块06总时间）": 0.001,
                "08 数字识别（模块08总时间）": 0.001,
                "装甲与数字识别阶段总时间": 0.002,
            }
        )
        return detections, annotated


class _FakeProjection:
    @staticmethod
    def pixel_to_world(pixel):
        return [float(pixel[0]), float(pixel[1]), 1.0]


def fake_detector() -> BaseDetector:
    detector = BaseDetector.__new__(BaseDetector)
    detector.car_detector = _FakeCarDetector()
    detector.armor_detector = _FakeArmorDetector()
    detector.xy_map_manager = None
    detector.pixel_world_transform = _FakeProjection()
    detector.digit_weight = 0.7
    detector.armor_weight = 0.3
    detector.min_digit_confidence = 0.3
    detector.min_combined_confidence = 0.55
    detector.ignored_digit_index = 5
    detector.visualize = False
    detector.separate_detection_visualization = False
    detector.timing_debug_enabled = True
    detector.timing_alpha = 0.2
    detector._timing_lock = RLock()
    detector.detector_timing_stats = {}
    detector.detector_frame_timing_ms = {}
    detector.track_frame_timing_ms = {}
    return detector


class PipelineEquivalenceTest(unittest.TestCase):
    def test_split_detection_matches_synchronous_composition(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        synchronous = fake_detector()
        serial_results, _ = synchronous.detect(image)

        split = fake_detector()
        cars, _ = split.detect_cars(image)
        split_results, _, _ = split.detect_armors(image, cars, 123.0)

        self.assertEqual(len(serial_results), len(split_results))
        left = serial_results[0]
        right = split_results[0]
        self.assertEqual(left.class_id, right.class_id)
        self.assertEqual(left.bot_id, right.bot_id)
        self.assertEqual(left.car_box, right.car_box)
        self.assertEqual(left.armor_box, right.armor_box)
        self.assertAlmostEqual(left.class_conf, right.class_conf)
        self.assertEqual(right.time_stamp, 123.0)

    def test_detailed_timings_do_not_repeat_parent_entries(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        detector = fake_detector()
        detector.car_detector = _FakeNumberedCarDetector()
        detector.armor_detector = _FakeDetailedArmorDetector()

        cars, car_timings = detector.detect_cars(image)
        _, _, armor_timings = detector.detect_armors(image, cars, 123.0)

        car_parent = "03 车辆检测与BoT-SORT（模块03总时间）"
        self.assertEqual(detector.detector_timing_stats[car_parent]["count"], 1)
        self.assertEqual(sum(name == car_parent for name in car_timings), 1)
        self.assertNotIn(
            "06 装甲位置检测含结果回传（模块06总时间）",
            armor_timings,
        )
        self.assertNotIn(
            "08 数字神经网络识别含结果回传（模块08总时间）",
            armor_timings,
        )
        self.assertEqual(
            sum(name.startswith("06-08 ") for name in armor_timings),
            1,
        )


if __name__ == "__main__":
    unittest.main()
