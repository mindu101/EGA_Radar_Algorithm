import time
import unittest
from functools import partial
from types import SimpleNamespace

from model.yolov12.predictor_with_tracker import PredictorWithTracker


class _FakeTracker:
    def update(self, *args, **kwargs):
        time.sleep(0.001)
        return []


def _fake_tracker_callback(predictor, persist=False):
    del persist
    predictor.trackers[0].update(None)


# 模拟 Ultralytics register_tracker 注册的 partial 回调身份。
_fake_tracker_callback.__module__ = "ultralytics.trackers.track"
_fake_tracker_callback.__name__ = "on_predict_postprocess_end"


class TimingHookTest(unittest.TestCase):
    def test_botsort_update_is_measured_separately_from_callback(self):
        tracker = _FakeTracker()
        fake_predictor = SimpleNamespace(trackers=[tracker])
        callback = partial(_fake_tracker_callback, persist=True)
        fake_model = SimpleNamespace(
            predictor=fake_predictor,
            callbacks={"on_predict_postprocess_end": [callback]},
        )

        predictor = PredictorWithTracker.__new__(PredictorWithTracker)
        predictor.model = fake_model
        predictor.last_timing_stats = {}
        predictor._timing_hooks_installed = False
        predictor._last_tracker_callback_s = 0.0
        predictor._last_botsort_update_s = 0.0

        predictor._install_tracker_timing_hooks()
        fake_model.callbacks["on_predict_postprocess_end"][0](fake_predictor)

        self.assertTrue(predictor._timing_hooks_installed)
        self.assertGreater(predictor._last_botsort_update_s, 0.0)
        self.assertGreaterEqual(
            predictor._last_tracker_callback_s,
            predictor._last_botsort_update_s,
        )

if __name__ == "__main__":
    unittest.main()
