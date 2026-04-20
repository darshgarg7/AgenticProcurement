"""
Unit tests for DelegationEngine (decision/delegation_engine.py).

Validates:
  - Purchase when safety gate is satisfied (regret ≤ ε_reg AND var ≤ ε_epi)  [Proposal §4.4 Eq.16]
  - Defer (QueryUser/Search/Wait) when thresholds are violated                [Proposal §4.4]
  - Worst-case regret computation is non-negative and correct shape            [Proposal §4.3 Eq.11]
  - QueryUser chosen when information gain exceeds search IG                   [Proposal §4.5 Eq.17]
  - Search/Wait fallback when IG is low                                        [Proposal Algorithm 1]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import unittest
from decision.delegation_engine import DelegationEngine
from models.bayesian_user import BayesianPreferenceModel
from core.interfaces import Observation, Purchase, QueryUser, Search, Wait
from config.settings import EngineConfig, ModelConfig


def make_obs(features, item_ids=None):
    """Helper to create an Observation."""
    if item_ids is None:
        item_ids = list(range(features.shape[0]))
    return Observation(item_ids=item_ids, features=features)


class TestSafetyGatePurchase(unittest.TestCase):
    """Agent MUST purchase when all three conditions of Eq.16 are met."""

    def test_purchase_when_gate_satisfied(self):
        """Low uncertainty + low regret + high utility → Purchase."""
        d = 3
        true_theta = np.array([1.0, 0.0, 0.0])
        # Tight posterior around true theta → low epistemic uncertainty
        model = BayesianPreferenceModel(d=d, m0=true_theta, S0=np.eye(d) * 0.001,
                                         config=ModelConfig(sigma2=0.05))
        config = EngineConfig(eps_reg=2.0, eps_var=1.0, tau_util=0.0)
        engine = DelegationEngine(model, config)

        X = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])
        obs = make_obs(X)
        action = engine.decide(obs)
        self.assertIsInstance(action, Purchase,
                              "Should purchase when uncertainty and regret are low")

    def test_purchase_selects_highest_utility(self):
        """When purchasing, the agent should pick the item with highest expected utility."""
        d = 2
        m0 = np.array([1.0, 0.0])
        model = BayesianPreferenceModel(d=d, m0=m0, S0=np.eye(d) * 0.001)
        config = EngineConfig(eps_reg=5.0, eps_var=5.0, tau_util=0.0)
        engine = DelegationEngine(model, config)

        X = np.array([[0.2, 0.8],   # low utility under m0
                       [0.9, 0.1]])  # high utility under m0
        obs = make_obs(X, item_ids=['A', 'B'])
        action = engine.decide(obs)
        self.assertIsInstance(action, Purchase)
        self.assertEqual(action.item_id, 'B')


class TestSafetyGateDefer(unittest.TestCase):
    """Agent must NOT purchase when thresholds are violated."""

    def test_defer_when_uncertainty_high(self):
        """Broad prior → high epistemic uncertainty → defer."""
        d = 4
        model = BayesianPreferenceModel(d=d, S0=np.eye(d) * 100.0)
        config = EngineConfig(eps_reg=10.0, eps_var=0.001, tau_util=0.0)  # very strict var threshold
        engine = DelegationEngine(model, config)

        X = np.random.randn(5, d)
        obs = make_obs(X)
        action = engine.decide(obs)
        self.assertNotIsInstance(action, Purchase,
                                 "Should defer when epistemic uncertainty exceeds threshold")

    def test_defer_when_utility_below_tau(self):
        """When all items have utility below tau_util, defer."""
        d = 2
        m0 = np.array([0.0, 0.0])
        model = BayesianPreferenceModel(d=d, m0=m0, S0=np.eye(d) * 0.001)
        config = EngineConfig(eps_reg=10.0, eps_var=10.0, tau_util=100.0)  # impossibly high threshold
        engine = DelegationEngine(model, config)

        X = np.array([[0.1, 0.1], [0.2, 0.2]])
        obs = make_obs(X)
        action = engine.decide(obs)
        self.assertNotIsInstance(action, Purchase,
                                 "Should defer when utility is below tau_util")


class TestWorstCaseRegret(unittest.TestCase):
    """Proposal §4.3: regret must be non-negative and well-shaped."""

    def test_regret_shape(self):
        d = 3
        model = BayesianPreferenceModel(d=d)
        config = EngineConfig(num_samples=50)
        engine = DelegationEngine(model, config)

        X = np.random.randn(10, d)
        regrets = engine.compute_worst_case_regret(X)
        self.assertEqual(regrets.shape, (10,))

    def test_regret_non_negative(self):
        d = 3
        model = BayesianPreferenceModel(d=d)
        engine = DelegationEngine(model, EngineConfig(num_samples=200))

        X = np.random.randn(10, d)
        regrets = engine.compute_worst_case_regret(X)
        self.assertTrue((regrets >= -1e-10).all(),
                        "Worst-case regret should be non-negative")

    def test_best_item_has_lowest_regret(self):
        """The item with highest mean utility should tend to have lowest regret."""
        d = 2
        m0 = np.array([1.0, 0.0])
        model = BayesianPreferenceModel(d=d, m0=m0, S0=np.eye(d) * 0.001)
        engine = DelegationEngine(model, EngineConfig(num_samples=500))

        X = np.array([[1.0, 0.0],   # clearly best under m0
                       [0.0, 1.0],
                       [-1.0, 0.0]])
        regrets = engine.compute_worst_case_regret(X)
        self.assertEqual(np.argmin(regrets), 0,
                         "Best item under tight posterior should have lowest regret")


class TestQueryUserLogic(unittest.TestCase):
    """When deferring, QueryUser should be chosen when IG is high."""

    def test_query_chosen_over_search_when_uncertain(self):
        """Moderate uncertainty → IG(query) > base_search_ig → QueryUser."""
        d = 3
        model = BayesianPreferenceModel(d=d, S0=np.eye(d) * 5.0,
                                         config=ModelConfig(sigma2=0.05))
        # Set thresholds so purchase is impossible, but IG is high
        config = EngineConfig(eps_reg=0.0, eps_var=0.0, tau_util=0.0,
                              base_search_ig=0.001)
        engine = DelegationEngine(model, config)

        X = np.array([[1.0, 0.0, 0.0]])
        obs = make_obs(X)
        action = engine.decide(obs)
        self.assertIsInstance(action, QueryUser,
                              "Should query user when information gain is high")


class TestEmptyObservation(unittest.TestCase):

    def test_search_on_empty(self):
        d = 3
        model = BayesianPreferenceModel(d=d)
        engine = DelegationEngine(model)
        obs = Observation(item_ids=[], features=np.empty((0, d)))
        action = engine.decide(obs)
        self.assertIsInstance(action, Search,
                              "Should search when no items are available")


if __name__ == '__main__':
    unittest.main()
