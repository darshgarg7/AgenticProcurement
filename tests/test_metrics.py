"""
Unit tests for MetricsTracker (evaluation/metrics.py).

Validates:
  - Default state is correct
  - Counter increments (queries, delays)
  - Purchase recording and exceedance flag   [Proposal §7]
  - get_stats returns correct dictionary
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from evaluation.metrics import MetricsTracker


class TestDefaultState(unittest.TestCase):

    def test_initial_values(self):
        t = MetricsTracker()
        self.assertEqual(t.queries, 0)
        self.assertEqual(t.delays, 0)
        self.assertFalse(t.purchased)
        self.assertEqual(t.realized_regret, 0.0)
        self.assertFalse(t.exceeded_regret)


class TestCounters(unittest.TestCase):

    def test_query_counter(self):
        t = MetricsTracker()
        for _ in range(5):
            t.record_query()
        self.assertEqual(t.queries, 5)

    def test_delay_counter(self):
        t = MetricsTracker()
        for _ in range(3):
            t.record_delay()
        self.assertEqual(t.delays, 3)


class TestPurchaseRecording(unittest.TestCase):

    def test_purchase_below_threshold(self):
        t = MetricsTracker()
        t.record_purchase(realized_regret=0.5, estimated_wc_regret=0.8, threshold=1.0)
        self.assertTrue(t.purchased)
        self.assertAlmostEqual(t.realized_regret, 0.5)
        self.assertAlmostEqual(t.estimated_worst_case_regret, 0.8)
        self.assertFalse(t.exceeded_regret, "Regret 0.5 should not exceed threshold 1.0")

    def test_purchase_above_threshold(self):
        t = MetricsTracker()
        t.record_purchase(realized_regret=1.5, estimated_wc_regret=2.0, threshold=1.0)
        self.assertTrue(t.exceeded_regret, "Regret 1.5 should exceed threshold 1.0")

    def test_purchase_at_exact_threshold(self):
        t = MetricsTracker()
        t.record_purchase(realized_regret=1.0, estimated_wc_regret=1.0, threshold=1.0)
        # 1.0 > 1.0 is False, so should NOT exceed
        self.assertFalse(t.exceeded_regret)


class TestGetStats(unittest.TestCase):

    def test_stats_dict_keys(self):
        t = MetricsTracker()
        stats = t.get_stats()
        expected_keys = {'queries', 'delays', 'purchased', 'realized_regret',
                         'estimated_worst_case_regret', 'exceeded_regret'}
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_stats_reflects_state(self):
        t = MetricsTracker()
        t.record_query()
        t.record_query()
        t.record_delay()
        t.record_purchase(realized_regret=0.3, estimated_wc_regret=0.6, threshold=1.0)
        stats = t.get_stats()
        self.assertEqual(stats['queries'], 2)
        self.assertEqual(stats['delays'], 1)
        self.assertTrue(stats['purchased'])
        self.assertAlmostEqual(stats['realized_regret'], 0.3)


if __name__ == '__main__':
    unittest.main()
