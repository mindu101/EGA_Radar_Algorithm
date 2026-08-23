import unittest

import numpy as np

from model.armor_detector import TwoStepArmorDetectorClassifier
from model.yolov12.image_utils import rgb_to_ultralytics_bgr_view


class TestDetectionColorContract(unittest.TestCase):
    def test_rgb_to_ultralytics_bgr_uses_shared_memory_view(self):
        rgb = np.array([[[11, 22, 33], [44, 55, 66]]], dtype=np.uint8)

        bgr = rgb_to_ultralytics_bgr_view(rgb)

        np.testing.assert_array_equal(
            bgr,
            np.array([[[33, 22, 11], [66, 55, 44]]], dtype=np.uint8),
        )
        self.assertTrue(np.shares_memory(rgb, bgr))
        # Ultralytics 会在 NumPy 预处理中再做一次 BGR->RGB，模型最终看到原始 RGB。
        np.testing.assert_array_equal(bgr[..., ::-1], rgb)

    def test_armor_color_class_mapping(self):
        compose = TwoStepArmorDetectorClassifier._compose_armor_class_id

        for pattern_id in range(5):
            self.assertEqual(compose(0, pattern_id), 10 + pattern_id)  # 灰色/死亡
            self.assertEqual(compose(1, pattern_id), 5 + pattern_id)   # 红方
            self.assertEqual(compose(2, pattern_id), pattern_id)       # 蓝方

    def test_invalid_armor_class_is_rejected(self):
        compose = TwoStepArmorDetectorClassifier._compose_armor_class_id

        self.assertIsNone(compose(3, 0))
        self.assertIsNone(compose(1, 5))
        self.assertIsNone(compose(1, -1))


if __name__ == "__main__":
    unittest.main()
