import os
import unittest

import numpy as np

from config.settings import EngineConfig
from core.interfaces import Observation, Purchase, QueryUser, Search, Wait
from decision.delegation_engine import DelegationEngine
from experiments.run_full_experiments import run_episode
from models.bayesian_user import BayesianPreferenceModel

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "products.csv")


class TestSeededReproducibility(unittest.TestCase):
    def test_full_episode_is_reproducible_with_seeded_rng(self):
        d = 8
        rng = np.random.RandomState(123)
        true_theta = rng.randn(d)
        true_theta /= np.linalg.norm(true_theta)
        prior_m0 = true_theta + rng.normal(0, 0.4, d)
        prior_S0 = np.eye(d) * 0.5

        stats1 = run_episode(
            true_theta,
            DATA_PATH,
            prior_m0,
            prior_S0,
            engine_config=EngineConfig(eps_reg=0.8, eps_var=0.8, tau_util=0.0),
            max_steps=20,
            rng=np.random.RandomState(999),
        )
        stats2 = run_episode(
            true_theta,
            DATA_PATH,
            prior_m0,
            prior_S0,
            engine_config=EngineConfig(eps_reg=0.8, eps_var=0.8, tau_util=0.0),
            max_steps=20,
            rng=np.random.RandomState(999),
        )

        self.assertEqual(stats1, stats2)


class TestDecisionDiagnostics(unittest.TestCase):
    def test_decide_with_diagnostics_matches_action_contract(self):
        model = BayesianPreferenceModel(
            d=2,
            m0=np.array([1.0, 0.0]),
            S0=np.eye(2) * 0.001,
        )
        engine = DelegationEngine(
            model,
            EngineConfig(eps_reg=5.0, eps_var=5.0, tau_util=0.0, num_samples=100),
            rng=np.random.RandomState(7),
        )
        obs = Observation(
            item_ids=["A", "B"],
            features=np.array([[0.1, 0.9], [0.9, 0.1]]),
        )

        action, diagnostics = engine.decide_with_diagnostics(obs)

        self.assertIsInstance(action, (Purchase, QueryUser, Search, Wait))
        self.assertEqual(diagnostics.best_idx, 1)
        self.assertEqual(diagnostics.expected_utils.shape, (2,))
        self.assertEqual(diagnostics.epistemic_uncertainties.shape, (2,))
        self.assertEqual(diagnostics.worst_case_regrets.shape, (2,))


if __name__ == "__main__":
    unittest.main()
