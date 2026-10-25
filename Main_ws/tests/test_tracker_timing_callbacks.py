import unittest
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Results
from ultralytics.trackers import register_tracker

from model.yolov12.predictor_with_tracker import PredictorWithTracker


class TestTrackerTimingCallbacks(unittest.TestCase):
    def test_callback_refresh_updates_tracker_once_per_frame(self):
        calls = []

        def update(*args, **kwargs):
            calls.append(1)
            return np.empty((0, 8))

        tracker = SimpleNamespace(update=update, tracked_stracks=[])
        predictor = SimpleNamespace(
            trackers=[tracker],
            args=SimpleNamespace(mode="track", task="detect"),
            dataset=SimpleNamespace(mode="image"),
        )
        model = SimpleNamespace(
            predictor=predictor,
            callbacks={"on_predict_start": [], "on_predict_postprocess_end": []},
        )
        model.add_callback = lambda event, callback: model.callbacks[event].append(callback)
        detector = PredictorWithTracker.__new__(PredictorWithTracker)
        detector.model = model
        detector._last_tracker_callback_s = 0.0
        detector._last_botsort_update_s = 0.0

        for _ in range(3):
            register_tracker(model, persist=True)
            detector._install_tracker_timing_hooks()
            # Also exercise repeated installation without a callback refresh.
            detector._install_tracker_timing_hooks()
            callbacks = model.callbacks["on_predict_postprocess_end"]
            self.assertEqual(len(callbacks), 1)
            predictor.results = [Results(
                np.zeros((10, 10, 3), dtype=np.uint8),
                path="frame.jpg",
                names={0: "car"},
                boxes=np.array([[1, 1, 5, 5, .9, 0]], dtype=np.float32),
            )]
            callbacks[0](predictor)

        self.assertEqual(len(calls), 3)
        self.assertGreater(detector._last_tracker_callback_s, 0.0)
        self.assertGreater(detector._last_botsort_update_s, 0.0)


if __name__ == "__main__":
    unittest.main()
