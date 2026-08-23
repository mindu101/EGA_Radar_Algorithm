# -*- coding: utf-8 -*-
"""异步相机录像器的低依赖回归测试。"""

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from utils.async_camera_recorder import AsyncCameraRecorder


class _FakeCamera:
    def __init__(self):
        self.listeners = {}
        self.next_token = 1

    def add_frame_listener(self, callback):
        token = self.next_token
        self.next_token += 1
        self.listeners[token] = callback
        return token

    def remove_frame_listener(self, token):
        self.listeners.pop(token, None)

    def emit(self, frame, stamp):
        for callback in list(self.listeners.values()):
            callback(frame, stamp)


class _SizedFakeCamera(_FakeCamera):
    width = 32
    height = 24


class _FakeWriter:
    def __init__(self, sink, delay_sec=0.0, **_kwargs):
        self.sink = sink
        self.delay_sec = delay_sec
        self.closed = False

    def write(self, frame):
        if self.delay_sec:
            time.sleep(self.delay_sec)
        self.sink.append((frame.shape, int(frame[0, 0, 0])))

    def close(self):
        self.closed = True

    def abort(self):
        self.closed = True


def _make_recorder(root, camera, sink, fps=30.0, queue_size=16, delay_sec=0.0):
    root = Path(root)

    def factory(**kwargs):
        return _FakeWriter(sink=sink, delay_sec=delay_sec, **kwargs)

    return AsyncCameraRecorder(
        camera=camera,
        output_path=str(root / "camera_record.mp4"),
        timestamp_path=str(root / "camera_timestamps.csv"),
        metrics_path=str(root / "recording_metrics.json"),
        target_fps=fps,
        color_order="rgb",
        queue_size=queue_size,
        encoder_factory=factory,
    )


class AsyncCameraRecorderTest(unittest.TestCase):
    def test_encoder_is_warmed_before_capture_listener_is_registered(self):
        with tempfile.TemporaryDirectory() as directory:
            camera = _SizedFakeCamera()
            encoded = []
            listener_state_during_init = []

            def factory(**kwargs):
                listener_state_during_init.append(bool(camera.listeners))
                time.sleep(0.02)
                return _FakeWriter(sink=encoded, **kwargs)

            root = Path(directory)
            recorder = AsyncCameraRecorder(
                camera=camera,
                output_path=str(root / "camera_record.mp4"),
                timestamp_path=str(root / "camera_timestamps.csv"),
                metrics_path=str(root / "recording_metrics.json"),
                target_fps=30.0,
                color_order="rgb",
                encoder_factory=factory,
            )
            recorder.start()
            camera.emit(np.zeros((24, 32, 3), dtype=np.uint8), 50.0)
            diagnostics = recorder.stop()

            self.assertEqual(listener_state_during_init, [False])
            self.assertEqual(diagnostics["encoded"], 1)
            self.assertGreaterEqual(diagnostics["encoder_init_ms"], 15.0)

    def test_video_rows_and_metrics_are_one_to_one(self):
        with tempfile.TemporaryDirectory() as directory:
            camera = _FakeCamera()
            encoded = []
            recorder = _make_recorder(directory, camera, encoded, fps=30.0)
            recorder.start()

            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            for index in range(10):
                frame[0, 0, 0] = index
                camera.emit(frame.copy(), 100.0 + index / 30.0)
            diagnostics = recorder.stop()

            with open(
                Path(directory) / "camera_timestamps.csv",
                newline="",
                encoding="utf-8",
            ) as file:
                rows = list(csv.DictReader(file))
            metrics = json.loads(
                (Path(directory) / "recording_metrics.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(len(encoded), 10)
            self.assertEqual(len(rows), 10)
            self.assertEqual([int(row["frame_index"]) for row in rows], list(range(10)))
            self.assertEqual(diagnostics["encoded"], 10)
            self.assertEqual(metrics["encoded"], 10)
            self.assertEqual(metrics["queue_overflow_dropped"], 0)
            self.assertEqual(metrics["status"], "stopped")
            self.assertFalse(camera.listeners)

    def test_60hz_source_is_sampled_to_about_30hz(self):
        with tempfile.TemporaryDirectory() as directory:
            camera = _FakeCamera()
            encoded = []
            recorder = _make_recorder(
                directory, camera, encoded, fps=30.0, queue_size=128
            )
            recorder.start()

            frame = np.zeros((16, 20, 3), dtype=np.uint8)
            for index in range(60):
                camera.emit(frame, 200.0 + index / 60.0)
            diagnostics = recorder.stop()

            self.assertGreaterEqual(diagnostics["encoded"], 29)
            self.assertLessEqual(diagnostics["encoded"], 31)
            self.assertEqual(
                diagnostics["capture_seen"],
                diagnostics["encoded"] + diagnostics["sampled_out"],
            )

    def test_full_queue_is_counted_without_breaking_csv_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            camera = _FakeCamera()
            encoded = []
            recorder = _make_recorder(
                directory,
                camera,
                encoded,
                fps=1000.0,
                queue_size=2,
                delay_sec=0.01,
            )
            recorder.start()

            frame = np.zeros((12, 16, 3), dtype=np.uint8)
            for index in range(40):
                camera.emit(frame, 300.0 + index / 1000.0)
            diagnostics = recorder.stop()

            with open(
                Path(directory) / "camera_timestamps.csv",
                newline="",
                encoding="utf-8",
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertGreater(diagnostics["queue_overflow_dropped"], 0)
            self.assertEqual(len(rows), diagnostics["encoded"])
            self.assertEqual(len(encoded), diagnostics["encoded"])
            self.assertEqual(
                diagnostics["enqueued"],
                diagnostics["encoded"],
            )


if __name__ == "__main__":
    unittest.main()
