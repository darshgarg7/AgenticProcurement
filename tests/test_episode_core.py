import os
import unittest

import numpy as np

from config.settings import EngineConfig
from core.episode import make_episode_context, run_agent_episode, step_agent_episode

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "products.csv")


class TestEpisodeCore(unittest.TestCase):
    def _params(self, seed=5):
        rng = np.random.RandomState(seed)
        true_theta = rng.randn(8)
        true_theta /= np.linalg.norm(true_theta)
        prior_m0 = true_theta + rng.normal(0, 0.4, 8)
        prior_S0 = np.eye(8) * 0.5
        return true_theta, prior_m0, prior_S0

    def test_step_payload_contains_audit_fields(self):
        true_theta, prior_m0, prior_S0 = self._params()
        context = make_episode_context(
            true_theta,
            DATA_PATH,
            prior_m0,
            prior_S0,
            engine_config=EngineConfig(eps_reg=0.8, eps_var=0.8, tau_util=0.0),
            rng=np.random.RandomState(10),
        )

        step = step_agent_episode(context, 0)

        self.assertIn(step.action, {"Purchase", "QueryUser", "Wait", "Search"})
        self.assertEqual(step.step, 1)
        self.assertIsInstance(step.available_item_ids, list)
        self.assertIsInstance(step.top_item_ids, list)

    def test_run_agent_episode_respects_step_limit(self):
        true_theta, prior_m0, prior_S0 = self._params()
        stats = run_agent_episode(
            true_theta,
            DATA_PATH,
            prior_m0,
            prior_S0,
            engine_config=EngineConfig(eps_reg=0.0, eps_var=0.0, tau_util=10.0),
            max_steps=3,
            rng=np.random.RandomState(11),
        )

        self.assertLessEqual(stats["steps_taken"], 3)
        self.assertEqual(len(stats["step_data"]), stats["steps_taken"])


if __name__ == "__main__":
    unittest.main()
