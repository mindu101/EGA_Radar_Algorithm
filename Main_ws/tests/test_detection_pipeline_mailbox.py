from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from tracker.CascadeMatchTracker.pipeline import (
    LatestPacketMailbox,
    PipelineDiagnostics,
    StageAPacket,
    freeze_mapping,
)


def packet(sequence: int) -> StageAPacket:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    return StageAPacket(
        pipeline_epoch=1,
        source_sequence=sequence,
        frame=frame,
        frame_stamp=float(sequence),
        frame_width=2,
        frame_height=2,
        input_frame_age_ms=0.0,
        ingress_monotonic=time.monotonic(),
        observation_monotonic=time.monotonic(),
        published_monotonic=time.monotonic(),
        faction="red",
        lidar_snapshot=None,
        ally_snapshot=None,
        lidar_fly_points=(),
        outpost_alive=1,
        cars=(),
        stage_a_timings_ms=freeze_mapping(),
    )


class LatestPacketMailboxTest(unittest.TestCase):
    def test_publish_is_latest_wins_and_nonblocking(self):
        mailbox = LatestPacketMailbox()
        self.assertFalse(mailbox.publish_latest(packet(1)))
        self.assertTrue(mailbox.publish_latest(packet(2)))
        self.assertEqual(mailbox.take_latest().source_sequence, 2)

    def test_close_discards_pending_and_wakes_waiter(self):
        mailbox = LatestPacketMailbox()
        result = []

        def wait_packet():
            result.append(mailbox.take_latest())

        worker = threading.Thread(target=wait_packet)
        worker.start()
        time.sleep(0.01)
        self.assertIsNone(mailbox.close())
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [None])

    def test_clear_returns_pending_packet(self):
        diagnostics = PipelineDiagnostics()
        mailbox = LatestPacketMailbox(diagnostics)
        mailbox.publish_latest(packet(7))
        self.assertEqual(mailbox.clear("epoch").source_sequence, 7)
        snapshot = diagnostics.snapshot()
        self.assertEqual(snapshot.published_total, 1)
        self.assertEqual(snapshot.epoch_drop_total, 1)
        mailbox.close()

    def test_diagnostics_change_with_mailbox_state(self):
        diagnostics = PipelineDiagnostics()
        mailbox = LatestPacketMailbox(diagnostics)
        mailbox.publish_latest(packet(1))
        mailbox.publish_latest(packet(2))
        mailbox.take_latest()
        snapshot = diagnostics.snapshot()
        self.assertEqual(snapshot.published_total, 2)
        self.assertEqual(snapshot.pending_replaced_total, 1)
        self.assertEqual(snapshot.taken_total, 1)


if __name__ == "__main__":
    unittest.main()
