"""Bayesian linear preference model for user utility inference."""

from typing import Any

import numpy as np
import numpy.typing as npt

from config.settings import ModelConfig
from core.interfaces import IPreferenceModel


class BayesianPreferenceModel(IPreferenceModel):
    """Maintain a Gaussian posterior over linear user utility weights."""

    def __init__(self, d: int, m0: npt.NDArray[np.float64] | None = None, S0: npt.NDArray[np.float64] | None = None, config: ModelConfig | None = None):
        """Create a preference model with optional prior mean and covariance."""
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        self.d = d
        self.m = np.zeros(d) if m0 is None else np.array(m0, dtype=np.float64)
        self.S = np.eye(d) if S0 is None else np.array(S0, dtype=np.float64)
        if self.m.shape != (d,):
            raise ValueError(f"m0 must have shape ({d},), got {self.m.shape}")
        if self.S.shape != (d, d):
            raise ValueError(f"S0 must have shape ({d}, {d}), got {self.S.shape}")
        if config is None:
            config = ModelConfig()
        self.sigma2 = config.sigma2

    def update(self, x: npt.NDArray[np.float64], y: float) -> None:
        """Condition the posterior on one noisy utility observation."""
        x_vec = self._as_feature_vector(x)

        # Kalman-form Bayesian linear regression update. This avoids repeated
        # matrix inversion and preserves symmetry of the posterior covariance.
        Sx = self.S @ x_vec
        innovation_var = self.sigma2 + float(x_vec @ Sx)
        if innovation_var <= 0:
            raise ValueError("Posterior update produced a non-positive innovation variance")
        gain = Sx / innovation_var
        residual = float(y) - float(x_vec @ self.m)

        self.m = self.m + gain * residual
        self.S = self.S - np.outer(gain, Sx)
        self.S = 0.5 * (self.S + self.S.T)

    def expected_utility(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return posterior-mean utility for each feature row."""
        features = self._as_feature_matrix(X)
        return features @ self.m

    def epistemic_uncertainty(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return reducible posterior utility variance for each feature row."""
        features = self._as_feature_matrix(X)
        # Vectorized variance: sum(X * (X @ S), axis=1) == diag(X @ S @ X.T)
        return np.sum(features * (features @ self.S), axis=1)

    def sample_theta(self, n: int = 1, rng: Any | None = None) -> npt.NDArray[np.float64]:
        """Draw preference-weight samples from the current posterior."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        random = np.random if rng is None else rng
        return random.multivariate_normal(self.m, self.S, size=n)

    def _as_feature_vector(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Validate and coerce one feature vector."""
        vector = np.asarray(x, dtype=np.float64).reshape(-1)
        if vector.shape != (self.d,):
            raise ValueError(f"feature vector must have shape ({self.d},), got {vector.shape}")
        return vector

    def _as_feature_matrix(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Validate and coerce a feature matrix."""
        features = np.asarray(X, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2 or features.shape[1] != self.d:
            raise ValueError(
                f"feature matrix must have shape (n, {self.d}), got {features.shape}"
            )
        return features
