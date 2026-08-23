# -*- coding: utf-8 -*-
"""同步回放相机在 rosbag 启动前的首帧预览测试。"""

import unittest

import numpy as np

from driver.replay_camera.timestamp_replay_camera import (
    TimestampReplayCamera,
    _FrameRecord,
)


class _FakeBuffer:
    def get_latest(self, group_id, timeout):
        return group_id, float(timeout)


class ReplayCameraPreviewTest(unittest.TestCase):
    def setUp(self):
        self.camera = TimestampReplayCamera.__new__(TimestampReplayCamera)
        self.camera._preview_frame = np.zeros((4, 6, 3), dtype=np.uint8)
        self.camera._preview_record = _FrameRecord(frame_index=12, timestamp=123.5)
        self.camera._published = 0
        self.camera._buffer = _FakeBuffer()

    def test_ui_repeatedly_receives_preview_before_clock_playback(self):
        first = self.camera.get_image_latest("radar_station", timeout=0.0)
        second = self.camera.get_image_latest("radar_station", timeout=0.0)
        self.assertIs(first[0], self.camera._preview_frame)
        self.assertIs(second[0], self.camera._preview_frame)
        self.assertEqual(first[1], 123.5)

    def test_tracker_never_receives_preview_frame(self):
        self.assertEqual(
            self.camera.get_image_latest("tracker", timeout=0.1),
            ("tracker", 0.1),
        )

    def test_ui_switches_to_replay_buffer_after_first_real_frame(self):
        self.camera._published = 1
        self.assertEqual(
            self.camera.get_image_latest("radar_station", timeout=0.2),
            ("radar_station", 0.2),
        )


if __name__ == "__main__":
    unittest.main()
