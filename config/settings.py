"""Typed configuration objects for the procurement agent."""

from dataclasses import dataclass

import numpy as np


@dataclass
class EngineConfig:
    """Decision-engine thresholds and Monte Carlo planning parameters."""

    eps_reg: float = 1.0
    eps_var: float = 0.8
    tau_util: float = 0.5
    num_samples: int = 100
    base_search_ig: float = 0.1
    mc_rollouts: int = 20          # Monte Carlo rollouts for Wait value estimation
    discount_factor: float = 0.95  # temporal discount for Wait rollouts
    confidence_percentile: float = 95.0
    wait_stockout_alpha: float = 0.02
    wait_price_fluctuation: float = 0.05
    empty_market_value_factor: float = 0.5

    def __post_init__(self) -> None:
        """Validate engine hyperparameters after dataclass initialization."""
        _require_non_negative("eps_reg", self.eps_reg)
        _require_non_negative("eps_var", self.eps_var)
        _require_positive_int("num_samples", self.num_samples)
        _require_non_negative("base_search_ig", self.base_search_ig)
        _require_positive_int("mc_rollouts", self.mc_rollouts)
        _require_between("discount_factor", self.discount_factor, 0.0, 1.0)
        _require_between("confidence_percentile", self.confidence_percentile, 0.0, 100.0)
        _require_between("wait_stockout_alpha", self.wait_stockout_alpha, 0.0, 1.0)
        _require_non_negative("wait_price_fluctuation", self.wait_price_fluctuation)
        _require_between("empty_market_value_factor", self.empty_market_value_factor, 0.0, 1.0)

@dataclass
class EnvConfig:
    """Market-simulator data path and transition dynamics."""

    data_path: str = "data/products.csv"
    alpha: float = 0.02
    price_fluctuation: float = 0.05

    def __post_init__(self) -> None:
        """Validate environment settings."""
        if not self.data_path:
            raise ValueError("data_path must be a non-empty path")
        _require_between("alpha", self.alpha, 0.0, 1.0)
        _require_non_negative("price_fluctuation", self.price_fluctuation)
    
@dataclass
class ModelConfig:
    """Bayesian preference-model noise parameters."""

    sigma2: float = 0.05

    def __post_init__(self) -> None:
        """Validate observation-noise variance."""
        _require_positive("sigma2", self.sigma2)

@dataclass
class PersonaConfig:
    """Defines a synthetic user persona via prior over theta_U."""
    name: str = "balanced"
    true_theta: np.ndarray | None = None
    prior_mean: np.ndarray | None = None
    prior_cov: np.ndarray | None = None

    @staticmethod
    def budget_shopper(d: int = 8, seed: int = 0) -> 'PersonaConfig':
        """Strongly prefers low price (negative weight on price_norm)."""
        rng = np.random.RandomState(seed)
        true_theta = np.zeros(d)
        true_theta[0] = -0.8   # price_norm: strongly negative (prefers cheap)
        true_theta[1] = 0.3    # rating_norm
        true_theta[2] = 0.2    # quality_norm
        true_theta /= np.linalg.norm(true_theta)
        prior_mean = true_theta + rng.normal(0, 0.3, d)
        prior_cov = np.eye(d) * 0.5
        return PersonaConfig(name="budget_shopper", true_theta=true_theta,
                             prior_mean=prior_mean, prior_cov=prior_cov)

    @staticmethod
    def quality_maximizer(d: int = 8, seed: int = 0) -> 'PersonaConfig':
        """Strongly prefers high quality and rating, tolerant on price."""
        rng = np.random.RandomState(seed)
        true_theta = np.zeros(d)
        true_theta[0] = 0.1    # price_norm: near-indifferent
        true_theta[1] = 0.7    # rating_norm: strongly positive
        true_theta[2] = 0.6    # quality_norm: strongly positive
        true_theta /= np.linalg.norm(true_theta)
        prior_mean = true_theta + rng.normal(0, 0.3, d)
        prior_cov = np.eye(d) * 0.5
        return PersonaConfig(name="quality_maximizer", true_theta=true_theta,
                             prior_mean=prior_mean, prior_cov=prior_cov)

    @staticmethod
    def balanced(d: int = 8, seed: int = 0) -> 'PersonaConfig':
        """Equal weighting across price, rating, quality."""
        rng = np.random.RandomState(seed)
        true_theta = np.zeros(d)
        true_theta[0] = -0.4   # price_norm: moderately prefers cheap
        true_theta[1] = 0.5    # rating_norm
        true_theta[2] = 0.5    # quality_norm
        true_theta /= np.linalg.norm(true_theta)
        prior_mean = true_theta + rng.normal(0, 0.3, d)
        prior_cov = np.eye(d) * 0.5
        return PersonaConfig(name="balanced", true_theta=true_theta,
                             prior_mean=prior_mean, prior_cov=prior_cov)


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _require_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _require_between(name: str, value: float, lower: float, upper: float) -> None:
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}], got {value}")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
