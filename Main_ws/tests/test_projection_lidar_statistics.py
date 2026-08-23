from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from tracker.CascadeMatchTracker.projection_lidar_statistics import (
    ProjectionLidarStatistics,
)


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["B1", "B2", "B3", "B4", "B7", "R1", "R2", "R3", "R4", "R7"]


def load_statistics_config() -> dict:
    with (ROOT / "config/params.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)["projection_lidar_statistics"]


def armor(confidence=0.9):
    return SimpleNamespace(
        armor_confidence=confidence,
        digit_confidence=confidence,
        combined_confidence=confidence,
    )


def observe(
    statistics,
    *,
    camera_point,
    lidar_points,
    lidar_index=0,
    camera_points=None,
    available_lidar_indices=None,
    now=10.0,
    class_id=0,
    source="camera_lidar",
):
    return statistics.observe(
        class_id=class_id,
        source=source,
        identity_confirmed=True,
        car_index=0,
        camera_point=camera_point,
        lidar_index=lidar_index,
        lidar_points=lidar_points,
        available_lidar_indices=(
            tuple(range(len(lidar_points)))
            if available_lidar_indices is None
            else available_lidar_indices
        ),
        camera_points=(
            [(0, camera_point)] if camera_points is None else camera_points
        ),
        selected_armor=armor(),
        now_monotonic=now,
    )


class ProjectionLidarStatisticsTest(unittest.TestCase):
    def setUp(self):
        self.config = load_statistics_config()
        self.statistics = ProjectionLidarStatistics(self.config, CLASS_NAMES)

    def test_welford_updates_global_and_class_statistics(self):
        self.assertTrue(
            observe(
                self.statistics,
                camera_point=(1.0, 1.0),
                lidar_points=[(1.1, 1.2)],
                now=10.0,
            )
        )
        self.assertTrue(
            observe(
                self.statistics,
                camera_point=(2.0, 2.0),
                lidar_points=[(2.3, 2.4)],
                now=10.3,
            )
        )
        snapshot = self.statistics.snapshot()
        self.assertEqual(snapshot["global"]["count"], 2)
        self.assertEqual(snapshot["by_faction"]["blue"]["count"], 2)
        self.assertEqual(snapshot["by_faction"]["red"]["count"], 0)
        self.assertEqual(snapshot["by_class"]["B1"]["count"], 2)
        np.testing.assert_allclose(snapshot["global"]["mean"], [0.2, 0.3])
        np.testing.assert_allclose(
            snapshot["by_faction"]["blue"]["mean"], [0.2, 0.3]
        )
        np.testing.assert_allclose(
            snapshot["global"]["covariance"], [[0.02, 0.02], [0.02, 0.02]]
        )

    def test_blue_and_red_statistics_are_updated_independently(self):
        self.assertTrue(
            observe(
                self.statistics,
                camera_point=(1.0, 1.0),
                lidar_points=[(1.1, 1.2)],
                class_id=0,
                now=10.0,
            )
        )
        self.assertTrue(
            observe(
                self.statistics,
                camera_point=(2.0, 2.0),
                lidar_points=[(2.3, 2.4)],
                class_id=6,
                now=10.0,
            )
        )
        snapshot = self.statistics.snapshot()
        self.assertEqual(snapshot["global"]["count"], 2)
        self.assertEqual(snapshot["by_faction"]["blue"]["count"], 1)
        self.assertEqual(snapshot["by_faction"]["red"]["count"], 1)
        np.testing.assert_allclose(
            snapshot["by_faction"]["blue"]["mean"], [0.1, 0.2]
        )
        np.testing.assert_allclose(
            snapshot["by_faction"]["red"]["mean"], [0.3, 0.4]
        )

    def test_report_separates_factions_and_formats_covariance_on_two_lines(self):
        observe(
            self.statistics,
            camera_point=(1.0, 1.0),
            lidar_points=[(1.1, 1.2)],
            now=10.0,
        )
        observe(
            self.statistics,
            camera_point=(2.0, 2.0),
            lidar_points=[(2.3, 2.4)],
            now=10.3,
        )
        report = self.statistics.format_report()
        self.assertIn("所有机器人总计", report)
        self.assertIn("蓝方总计", report)
        self.assertIn("红方总计", report)
        self.assertIn("\n================================\n", report)
        self.assertIn("  B1: n=2\n", report)
        self.assertIn("\n\n  B2: n=0\n", report)
        self.assertRegex(
            report,
            r"cov=\[\[[^\n]+\],\n\s+\[[^\n]+\]\]",
        )

    def test_sample_interval_rejects_correlated_frame(self):
        self.assertTrue(
            observe(
                self.statistics,
                camera_point=(1.0, 1.0),
                lidar_points=[(1.1, 1.1)],
                now=10.0,
            )
        )
        self.assertFalse(
            observe(
                self.statistics,
                camera_point=(1.0, 1.0),
                lidar_points=[(1.1, 1.1)],
                now=10.1,
            )
        )
        self.assertEqual(
            self.statistics.snapshot()["rejections"]["sample_interval"], 1
        )

    def test_closer_competitor_rejects_non_mutual_nearest_pair(self):
        accepted = observe(
            self.statistics,
            camera_point=(1.0, 1.0),
            lidar_points=[(1.4, 1.0), (1.1, 1.0)],
            lidar_index=0,
        )
        self.assertFalse(accepted)
        self.assertEqual(
            self.statistics.snapshot()["rejections"]["not_mutual_nearest"], 1
        )

    def test_ratio_uses_d1_divided_by_second_distance(self):
        config = copy.deepcopy(self.config)
        config["min_lidar_margin_m"] = 0.0
        config["min_camera_margin_m"] = 0.0
        config["max_lidar_ambiguity_ratio"] = 0.5
        statistics = ProjectionLidarStatistics(config, CLASS_NAMES)
        accepted = observe(
            statistics,
            camera_point=(0.0, 0.0),
            lidar_points=[(0.4, 0.0), (0.6, 0.0)],
            lidar_index=0,
        )
        self.assertFalse(accepted)
        self.assertEqual(statistics.snapshot()["rejections"]["lidar_ratio"], 1)

    def test_gray_source_is_controlled_by_config(self):
        self.assertFalse(
            observe(
                self.statistics,
                camera_point=(1.0, 1.0),
                lidar_points=[(1.1, 1.1)],
                source="gray_camera_lidar",
            )
        )
        self.assertEqual(
            self.statistics.snapshot()["rejections"]["gray_disabled"], 1
        )

    def test_reset_clears_global_class_and_counters(self):
        observe(
            self.statistics,
            camera_point=(1.0, 1.0),
            lidar_points=[(1.1, 1.1)],
        )
        self.statistics.reset("外参更新")
        snapshot = self.statistics.snapshot()
        self.assertEqual(snapshot["global"]["count"], 0)
        self.assertEqual(snapshot["by_faction"]["blue"]["count"], 0)
        self.assertEqual(snapshot["by_faction"]["red"]["count"], 0)
        self.assertEqual(snapshot["by_class"]["B1"]["count"], 0)
        self.assertEqual(snapshot["evaluated"], 0)
        self.assertEqual(snapshot["last_reset_reason"], "外参更新")


if __name__ == "__main__":
    unittest.main()
