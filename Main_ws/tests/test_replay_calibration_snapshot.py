# -*- coding: utf-8 -*-
"""同步回放标定快照加载测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
workspace_path = str(WORKSPACE_ROOT)
if workspace_path in sys.path:
    sys.path.remove(workspace_path)
sys.path.insert(0, workspace_path)

from tools.synchronized_replay import (
    ReplayConfigurationError,
    _merge_recorded_calibration,
)


class ReplayCalibrationSnapshotTest(unittest.TestCase):
    def test_snapshot_supplies_intrinsics_and_extrinsics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera_dir = root / "camera"
            run_dir = root / "run"
            camera_dir.mkdir()
            run_dir.mkdir()
            snapshot = {
                "version": 1,
                "intrinsics": {
                    "image_width": 4024,
                    "image_height": 3036,
                    # 故意与 params 不同，用于证明 replay 强制使用录像内参。
                    "K": [
                        [1000.0, 0.0, 500.0],
                        [0.0, 1001.0, 400.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "dist_coeffs": [0.1, 0.2, 0.0, 0.0, 0.0],
                },
                "extrinsics": {
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "translation_vector": [[1.0], [2.0], [3.0]],
                },
            }
            with (camera_dir / "calibration_snapshot.yaml").open(
                "w", encoding="utf-8"
            ) as file:
                yaml.safe_dump(snapshot, file, sort_keys=False)

            params_k = [
                [2000.0, 0.0, 1000.0],
                [0.0, 2001.0, 800.0],
                [0.0, 0.0, 1.0],
            ]
            params_dist = [0.01, 0.02, 0.0, 0.0, 0.0]
            params = {
                "transform": {
                    "K": params_k,
                    "dist_coeffs": params_dist,
                }
            }
            _extrinsics_path, snapshot_info = _merge_recorded_calibration(
                params, camera_dir, run_dir
            )

            self.assertEqual(params["transform"]["K"], snapshot["intrinsics"]["K"])
            self.assertEqual(
                params["transform"]["dist_coeffs"],
                snapshot["intrinsics"]["dist_coeffs"],
            )
            self.assertEqual(params["transform"]["R"][0], [1.0, 0.0, 0.0])
            self.assertEqual(params["transform"]["t"], [[1.0], [2.0], [3.0]])
            self.assertEqual(snapshot_info["image_width"], 4024)
            self.assertEqual(snapshot_info["image_height"], 3036)

    def test_missing_calibration_snapshot_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera_dir = root / "camera"
            run_dir = root / "run"
            camera_dir.mkdir()
            run_dir.mkdir()
            params = {
                "transform": {
                    "K": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "dist_coeffs": [0.0] * 5,
                }
            }
            with self.assertRaisesRegex(
                ReplayConfigurationError, "缺少 calibration_snapshot.yaml"
            ):
                _merge_recorded_calibration(params, camera_dir, run_dir)

    def test_incomplete_snapshot_intrinsics_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera_dir = root / "camera"
            run_dir = root / "run"
            camera_dir.mkdir()
            run_dir.mkdir()
            snapshot = {
                "version": 1,
                "intrinsics": {
                    "image_width": 4024,
                    "image_height": 3036,
                    "dist_coeffs": [0.0] * 5,
                },
                "extrinsics": {
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "translation_vector": [[1.0], [2.0], [3.0]],
                },
            }
            with (camera_dir / "calibration_snapshot.yaml").open(
                "w", encoding="utf-8"
            ) as file:
                yaml.safe_dump(snapshot, file, sort_keys=False)

            params = {
                "transform": {
                    "K": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "dist_coeffs": [0.0] * 5,
                }
            }
            with self.assertRaisesRegex(
                ReplayConfigurationError,
                r"calibration_snapshot.yaml::intrinsics.K",
            ):
                _merge_recorded_calibration(params, camera_dir, run_dir)


if __name__ == "__main__":
    unittest.main()
