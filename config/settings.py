from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

@dataclass
class EngineConfig:
    eps_reg: float = 1.0
    eps_var: float = 0.8
    tau_util: float = 0.5
    num_samples: int = 100
    base_search_ig: float = 0.1
    mc_rollouts: int = 20          # Monte Carlo rollouts for Wait value estimation
    discount_factor: float = 0.95  # temporal discount for Wait rollouts

@dataclass
class EnvConfig:
    data_path: str = "data/products.csv"
    alpha: float = 0.02
    price_fluctuation: float = 0.05
    
@dataclass
class ModelConfig:
    sigma2: float = 0.05

@dataclass
class PersonaConfig:
    """Defines a synthetic user persona via prior over theta_U."""
    name: str = "balanced"
    true_theta: Optional[np.ndarray] = None
    prior_mean: Optional[np.ndarray] = None
    prior_cov: Optional[np.ndarray] = None

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
