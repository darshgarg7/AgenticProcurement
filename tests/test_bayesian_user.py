"""
Unit tests for BayesianPreferenceModel (models/bayesian_user.py).

Validates:
  - Posterior convergence toward true theta after repeated observations  [Proposal §4.1]
  - Epistemic uncertainty (x^T S x) strictly decreases after update      [Proposal §4.1, §4.4]
  - Expected utility m^T x matches hand-computed values                   [Proposal §4.1 Eq.8]
  - sample_theta draws are distributed around the posterior mean          [Proposal §4.3]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import unittest
from models.bayesian_user import BayesianPreferenceModel
from config.settings import ModelConfig


class TestBayesianPrior(unittest.TestCase):
    """Verify initial state is set correctly."""

    def test_default_prior(self):
        model = BayesianPreferenceModel(d=4)
        np.testing.assert_array_equal(model.m, np.zeros(4))
        np.testing.assert_array_equal(model.S, np.eye(4))

    def test_custom_prior(self):
        m0 = np.array([1.0, 2.0, 3.0])
        S0 = np.eye(3) * 0.5
        model = BayesianPreferenceModel(d=3, m0=m0, S0=S0)
        np.testing.assert_array_almost_equal(model.m, m0)
        np.testing.assert_array_almost_equal(model.S, S0)


class TestBayesianUpdate(unittest.TestCase):
    """Verify conjugate Gaussian update correctness."""

    def setUp(self):
        self.d = 4
        self.true_theta = np.array([0.5, -0.3, 0.8, 0.1])
        self.sigma2 = 0.05
        self.config = ModelConfig(sigma2=self.sigma2)

    def test_posterior_converges_to_true_theta(self):
        """After many noisy observations, posterior mean should approach true theta."""
        model = BayesianPreferenceModel(d=self.d, config=self.config)
        rng = np.random.RandomState(42)

        for _ in range(200):
            x = rng.randn(self.d)
            x /= np.linalg.norm(x)
            y = self.true_theta @ x + rng.normal(0, np.sqrt(self.sigma2))
            model.update(x, y)

        np.testing.assert_array_almost_equal(model.m, self.true_theta, decimal=1)

    def test_epistemic_uncertainty_decreases(self):
        """Epistemic variance for a fixed item must decrease after every update."""
        model = BayesianPreferenceModel(d=self.d, config=self.config)
        x_test = np.array([[0.5, 0.5, 0.5, 0.5]])
        rng = np.random.RandomState(7)

        prev_var = model.epistemic_uncertainty(x_test)[0]
        for _ in range(10):
            x_obs = rng.randn(self.d)
            x_obs /= np.linalg.norm(x_obs)
            y = self.true_theta @ x_obs + rng.normal(0, np.sqrt(self.sigma2))
            model.update(x_obs, y)

            curr_var = model.epistemic_uncertainty(x_test)[0]
            self.assertLess(curr_var, prev_var,
                            "Epistemic uncertainty must strictly decrease after an update")
            prev_var = curr_var

    def test_epistemic_uncertainty_approaches_zero(self):
        """After very many observations, epistemic uncertainty should be near zero."""
        model = BayesianPreferenceModel(d=self.d, config=self.config)
        rng = np.random.RandomState(99)

        for _ in range(500):
            x = rng.randn(self.d)
            x /= np.linalg.norm(x)
            y = self.true_theta @ x + rng.normal(0, np.sqrt(self.sigma2))
            model.update(x, y)

        x_test = np.array([[1.0, 0.0, 0.0, 0.0]])
        var = model.epistemic_uncertainty(x_test)[0]
        self.assertLess(var, 0.01, "Variance should be near zero after 500 observations")


class TestExpectedUtility(unittest.TestCase):
    """Verify expected utility is m^T x (Proposal Eq. 8 mean term)."""

    def test_utility_equals_dot_product(self):
        m0 = np.array([1.0, 2.0, 3.0])
        model = BayesianPreferenceModel(d=3, m0=m0)
        X = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.5, 0.5, 0.0]])
        expected = X @ m0
        np.testing.assert_array_almost_equal(model.expected_utility(X), expected)

    def test_utility_rank_order(self):
        """Higher-weight features should yield higher utility."""
        m0 = np.array([0.1, 0.9])  # strongly prefers feature 2
        model = BayesianPreferenceModel(d=2, m0=m0)
        X = np.array([[1.0, 0.0],   # high feature 1
                       [0.0, 1.0]])  # high feature 2
        utils = model.expected_utility(X)
        self.assertGreater(utils[1], utils[0])


class TestSampleTheta(unittest.TestCase):
    """Verify theta samples are consistent with posterior."""

    def test_sample_shape(self):
        model = BayesianPreferenceModel(d=5)
        samples = model.sample_theta(n=100)
        self.assertEqual(samples.shape, (100, 5))

    def test_sample_mean_near_posterior(self):
        m0 = np.array([1.0, -1.0, 0.5])
        model = BayesianPreferenceModel(d=3, m0=m0, S0=np.eye(3) * 0.01)
        samples = model.sample_theta(n=5000)
        np.testing.assert_array_almost_equal(samples.mean(axis=0), m0, decimal=1)

    def test_sample_covariance_near_posterior(self):
        S0 = np.diag([0.1, 0.2, 0.3])
        model = BayesianPreferenceModel(d=3, S0=S0)
        samples = model.sample_theta(n=10000)
        np.testing.assert_array_almost_equal(np.cov(samples.T), S0, decimal=1)


if __name__ == '__main__':
    unittest.main()
