"""
Unit tests for StochasticMarket (environment/simulator.py).

Validates:
  - Stock-out rate is consistent with alpha parameter   [Proposal §4.2 Eq.9]
  - Price fluctuation stays within expected bounds       [Proposal §4.2]
  - Feature extraction produces L2-normalized rows       [Implementation detail]
  - Edge case: empty market after all items stocked out  [Robustness]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import unittest
from environment.simulator import StochasticMarket
from config.settings import EnvConfig

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'products.csv')


class TestMarketInitialization(unittest.TestCase):

    def test_loads_csv(self):
        env = StochasticMarket(config=EnvConfig(data_path=DATA_PATH))
        self.assertGreater(len(env.items), 0, "Market should load items from CSV")

    def test_all_items_available_initially(self):
        env = StochasticMarket(config=EnvConfig(data_path=DATA_PATH))
        self.assertEqual(len(env.available_indices), len(env.items))


class TestObservation(unittest.TestCase):

    def test_feature_shape(self):
        env = StochasticMarket(config=EnvConfig(data_path=DATA_PATH))
        obs = env.observe()
        n_items = len(env.available_indices)
        self.assertEqual(obs.features.shape[0], n_items)
        self.assertGreater(obs.features.shape[1], 0)

    def test_features_are_l2_normalized(self):
        """Each row should have L2 norm ≈ 1.0 after normalization."""
        env = StochasticMarket(config=EnvConfig(data_path=DATA_PATH))
        obs = env.observe()
        norms = np.linalg.norm(obs.features, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(norms)), decimal=5)

    def test_item_ids_match_features(self):
        env = StochasticMarket(config=EnvConfig(data_path=DATA_PATH))
        obs = env.observe()
        self.assertEqual(len(obs.item_ids), obs.features.shape[0])


class TestStockOutDynamics(unittest.TestCase):
    """Proposal §4.2 Eq.9: P(z_{i,t+1}=0 | z_{i,t}=1) = alpha."""

    def test_stockout_rate_approximates_alpha(self):
        """Over many steps, fraction of items lost per step ≈ alpha."""
        alpha = 0.05
        env = StochasticMarket(config=EnvConfig(data_path=DATA_PATH, alpha=alpha))
        initial_count = len(env.available_indices)

        total_removed = 0
        total_at_risk = 0
        np.random.seed(123)

        for _ in range(100):
            before = len(env.available_indices)
            env.step()
            after = len(env.available_indices)
            removed = before - after
            total_removed += removed
            total_at_risk += before

        empirical_rate = total_removed / total_at_risk if total_at_risk > 0 else 0
        # Should be roughly alpha; allow generous tolerance for stochasticity
        self.assertAlmostEqual(empirical_rate, alpha, delta=0.02,
                               msg=f"Empirical stock-out rate {empirical_rate:.3f} should be near alpha={alpha}")

    def test_items_decrease_over_time(self):
        """With alpha > 0, available items should generally decrease."""
        env = StochasticMarket(config=EnvConfig(data_path=DATA_PATH, alpha=0.05))
        initial = len(env.available_indices)
        np.random.seed(42)
        for _ in range(50):
            env.step()
        self.assertLess(len(env.available_indices), initial)

    def test_zero_alpha_no_stockout(self):
        """With alpha=0, no items should ever go out of stock."""
        env = StochasticMarket(config=EnvConfig(data_path=DATA_PATH, alpha=0.0))
        initial = len(env.available_indices)
        for _ in range(20):
            env.step()
        self.assertEqual(len(env.available_indices), initial)


class TestPriceFluctuation(unittest.TestCase):

    def test_prices_change_after_step(self):
        """Prices should change (with high probability) when fluctuation > 0."""
        env = StochasticMarket(config=EnvConfig(data_path=DATA_PATH, alpha=0.0, price_fluctuation=0.1))
        prices_before = env.items['price_norm'].copy()
        np.random.seed(42)
        env.step()
        prices_after = env.items['price_norm']
        # At least some prices should differ
        self.assertFalse(np.allclose(prices_before, prices_after),
                         "Prices should change after a step with nonzero fluctuation")

    def test_prices_stay_positive(self):
        """Prices must remain > 0 even after many steps."""
        env = StochasticMarket(config=EnvConfig(data_path=DATA_PATH, alpha=0.0, price_fluctuation=0.2))
        np.random.seed(42)
        for _ in range(50):
            env.step()
        self.assertTrue((env.items['price_norm'] > 0).all(),
                        "All prices must remain positive")


class TestEmptyMarket(unittest.TestCase):

    def test_observe_empty_market(self):
        """If all items are stocked out, observe should return empty arrays."""
        env = StochasticMarket(config=EnvConfig(data_path=DATA_PATH, alpha=1.0))
        env.step()  # alpha=1 → everything goes out of stock
        obs = env.observe()
        self.assertEqual(len(obs.item_ids), 0)
        self.assertEqual(obs.features.shape[0], 0)

    def test_step_on_empty_market(self):
        """Stepping on an empty market should not crash."""
        env = StochasticMarket(config=EnvConfig(data_path=DATA_PATH, alpha=1.0))
        env.step()
        env.step()  # should not raise


if __name__ == '__main__':
    unittest.main()
