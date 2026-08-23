from __future__ import annotations

import unittest

from tracker.CascadeMatchTracker.pipeline import (
    PipelineDiagnostics,
    freeze_mapping,
)


class PipelineStateTest(unittest.TestCase):
    def test_source_and_commit_sequences_have_distinct_semantics(self):
        diagnostics = PipelineDiagnostics()
        self.assertEqual(diagnostics.next_source_sequence(), 1)
        self.assertEqual(diagnostics.next_source_sequence(), 2)

        commit_sequence, interval = diagnostics.record_commit(10.0)
        self.assertEqual(commit_sequence, 1)
        self.assertIsNone(interval)
        commit_sequence, interval = diagnostics.record_commit(10.04)
        self.assertEqual(commit_sequence, 2)
        self.assertAlmostEqual(interval, 0.04)

        snapshot = diagnostics.snapshot()
        self.assertEqual(snapshot.source_sequence, 2)
        self.assertEqual(snapshot.commit_sequence, 2)
        self.assertEqual(snapshot.committed_total, snapshot.commit_sequence)

    def test_frozen_timing_mapping_rejects_mutation(self):
        timings = freeze_mapping({"A": 1.0})
        with self.assertRaises(TypeError):
            timings["A"] = 2.0

    def test_drop_counters_are_thread_safe_api_fields(self):
        diagnostics = PipelineDiagnostics()
        diagnostics.increment("pending_replaced_total", 2)
        diagnostics.increment("stale_before_stage_b_total")
        snapshot = diagnostics.snapshot()
        self.assertEqual(snapshot.pending_replaced_total, 2)
        self.assertEqual(snapshot.stale_before_stage_b_total, 1)


if __name__ == "__main__":
    unittest.main()
