"""Decision policy for uncertainty-aware delegated procurement."""

from typing import Any

import numpy as np
import numpy.typing as npt

from config.settings import EngineConfig
from core.interfaces import (
    Action,
    DecisionDiagnostics,
    IDecisionEngine,
    IPreferenceModel,
    Observation,
    Purchase,
    QueryUser,
    Search,
    Wait,
)


class DelegationEngine(IDecisionEngine):
    """Select purchase, query, wait, or search actions from belief diagnostics."""

    def __init__(
        self,
        model: IPreferenceModel,
        config: EngineConfig | None = None,
        rng: Any | None = None,
    ):
        """Initialize the engine with a preference model and random source."""
        self.model = model
        self.config = EngineConfig() if config is None else config
        self.rng = np.random if rng is None else rng
        
    def compute_worst_case_regret(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Approximate item regret using posterior samples of user preferences."""
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D feature matrix, got shape {X.shape}")
        if X.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)

        thetas = self.model.sample_theta(self.config.num_samples, rng=self.rng) # (N, d)
        
        U = thetas @ X.T # (N, M)
        max_U = np.max(U, axis=1, keepdims=True) # (N, 1)
        regret = max_U - U # (N, M)
        
        return np.percentile(regret, self.config.confidence_percentile, axis=0) # (M,)

    def estimate_wait_value(self, obs: Observation) -> float:
        """Estimate the one-step value of Wait via Monte Carlo rollouts.
        
        Simulates future price/availability transitions and computes
        the expected best utility after waiting. [Proposal §4.2, Algorithm 1]
        """
        if obs.features.shape[0] == 0:
            return -float('inf')
        
        current_best = float(np.max(self.model.expected_utility(obs.features)))
        future_values = []
        
        for _ in range(self.config.mc_rollouts):
            n_items = obs.features.shape[0]
            survive_mask = self.rng.uniform(size=n_items) > self.config.wait_stockout_alpha
            if not survive_mask.any():
                future_values.append(current_best * self.config.empty_market_value_factor)
                continue

            future_features = obs.features[survive_mask].copy()
            noise = self.rng.normal(
                loc=1.0,
                scale=self.config.wait_price_fluctuation,
                size=future_features.shape[0],
            )
            future_features[:, 0] *= noise
            norms = np.linalg.norm(future_features, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            future_features = future_features / norms
            
            future_utils = self.model.expected_utility(future_features)
            future_values.append(float(np.max(future_utils)))
        
        discounted_future = self.config.discount_factor * np.mean(future_values)
        return discounted_future

    def evaluate(self, obs: Observation) -> DecisionDiagnostics:
        """Compute utilities, uncertainty, and sampled regret for an observation."""
        M = obs.features.shape[0]
        if M == 0:
            return DecisionDiagnostics(
                expected_utils=np.empty((0,), dtype=np.float64),
                epistemic_uncertainties=np.empty((0,), dtype=np.float64),
                worst_case_regrets=np.empty((0,), dtype=np.float64),
                best_idx=None,
            )
        
        X = obs.features
            
        expected_utils = self.model.expected_utility(X) # (M,)
        epistemic_uncs = self.model.epistemic_uncertainty(X) # (M,)
        worst_regrets = self.compute_worst_case_regret(X) # (M,)

        return DecisionDiagnostics(
            expected_utils=expected_utils,
            epistemic_uncertainties=epistemic_uncs,
            worst_case_regrets=worst_regrets,
            best_idx=int(np.argmax(expected_utils)),
        )

    def decide_with_diagnostics(
        self,
        obs: Observation,
        verbose: bool = False,
    ) -> tuple[Action, DecisionDiagnostics]:
        """Choose an action and return the diagnostics used to justify it."""
        diagnostics = self.evaluate(obs)

        if diagnostics.best_idx is None:
            return Search(), diagnostics

        expected_utils = diagnostics.expected_utils
        epistemic_uncs = diagnostics.epistemic_uncertainties
        worst_regrets = diagnostics.worst_case_regrets
        
        if verbose:
            top_indices = np.argsort(expected_utils)[::-1][:3]
            top_ids = [obs.item_ids[i] for i in top_indices]
            print(f"Available items: {top_ids}")
            print("")
            for i in top_indices:
                print(f"Item {obs.item_ids[i]}:")
                print(f"  expected_utility={expected_utils[i]:.2f}")
                print(f"  epistemic_uncertainty={epistemic_uncs[i]:.2f}")
                print(f"  estimated_worst_case_regret={worst_regrets[i]:.2f}")
            print("")
            
        # 3. Let i* = argmax expected utility
        best_idx = diagnostics.best_idx
        best_id = obs.item_ids[best_idx]
        
        mu_star = expected_utils[best_idx]
        reg_star = worst_regrets[best_idx]
        var_star = epistemic_uncs[best_idx]
        
        action = None
        # 4. If mu >= tau_util, R <= eps_reg, and var <= eps_var -> Purchase
        if mu_star >= self.config.tau_util and reg_star <= self.config.eps_reg and var_star <= self.config.eps_var:
            action = Purchase(item_id=best_id)
            if verbose:
                print(f"Decision: Purchase Item {best_id}\n")
        else:
            # 5. Else: Estimate the expected Information Gain
            sigma2 = getattr(self.model, 'sigma2', 0.05)
            ig_query = 0.5 * np.log(1 + var_star / sigma2)
            ig_search = self.config.base_search_ig
            
            # 6. Estimate Wait value via MC rollouts [Proposal Algorithm 1]
            wait_value = self.estimate_wait_value(obs)
            current_value = float(mu_star)
            wait_advantage = wait_value - current_value
            diagnostics.ig_query = float(ig_query)
            diagnostics.wait_value = float(wait_value)
            diagnostics.wait_advantage = float(wait_advantage)
            
            if ig_query > ig_search and ig_query > max(wait_advantage, 0):
                action = QueryUser(item_id=best_id)
                if verbose:
                    print(f"Decision: QueryUser (IG={ig_query:.3f})\n")
            elif wait_advantage > ig_search:
                action = Wait()
                if verbose:
                    print(
                        f"Decision: Wait "
                        f"(wait_value={wait_value:.3f}, current={current_value:.3f})\n"
                    )
            else:
                action = Search()
                if verbose:
                    print("Decision: Search\n")
                    
        return action, diagnostics

    def decide(self, obs: Observation, verbose: bool = False) -> Action:
        """Choose the next action without returning diagnostics."""
        action, _ = self.decide_with_diagnostics(obs, verbose=verbose)
        return action
