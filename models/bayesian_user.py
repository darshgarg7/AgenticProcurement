import numpy as np
import numpy.typing as npt
from typing import Optional
from core.interfaces import IPreferenceModel
from config.settings import ModelConfig

class BayesianPreferenceModel(IPreferenceModel):
    def __init__(self, d: int, m0: Optional[npt.NDArray[np.float64]] = None, S0: Optional[npt.NDArray[np.float64]] = None, config: ModelConfig = ModelConfig()):
        self.d = d
        self.m = np.zeros(d) if m0 is None else np.array(m0, dtype=np.float64)
        self.S = np.eye(d) if S0 is None else np.array(S0, dtype=np.float64)
        self.sigma2 = config.sigma2

    def update(self, x: npt.NDArray[np.float64], y: float) -> None:
        x_col = x.reshape(-1, 1)
        S_inv = np.linalg.inv(self.S)
        S_inv_new = S_inv + (x_col @ x_col.T) / self.sigma2
        self.S = np.linalg.inv(S_inv_new)
        self.m = self.S @ (S_inv @ self.m + x_col.flatten() * y / self.sigma2)

    def expected_utility(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return X @ self.m

    def epistemic_uncertainty(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Vectorized variance: sum(X * (X @ S), axis=1) == diag(X @ S @ X.T)
        return np.sum(X * (X @ self.S), axis=1)

    def sample_theta(self, n: int = 1) -> npt.NDArray[np.float64]:
        return np.random.multivariate_normal(self.m, self.S, size=n)
