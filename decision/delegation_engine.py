import numpy as np
import numpy.typing as npt
import copy

from core.interfaces import IDecisionEngine, IPreferenceModel, Observation, Action, Purchase, QueryUser, Wait, Search
from config.settings import EngineConfig

class DelegationEngine(IDecisionEngine):
    def __init__(self, model: IPreferenceModel, config: EngineConfig | None = None):
        self.model = model
        self.config = EngineConfig() if config is None else config
        
    def compute_worst_case_regret(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        thetas = self.model.sample_theta(self.config.num_samples) # (N, d)
        
        U = thetas @ X.T # (N, M)
        max_U = np.max(U, axis=1, keepdims=True) # (N, 1)
        regret = max_U - U # (N, M)
        
        return np.percentile(regret, 95, axis=0) # (M,)

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
            # Simulate: some items may go out of stock, prices may change
            n_items = obs.features.shape[0]
            # Randomly drop items (stock-out simulation)
            survive_mask = np.random.rand(n_items) > 0.02  # approximate alpha
            if not survive_mask.any():
                future_values.append(current_best * 0.5)  # penalty for empty market
                continue
            future_features = obs.features[survive_mask].copy()
            # Simulate price fluctuation on the price dimension
            noise = np.random.normal(1.0, 0.05, size=future_features.shape[0])
            future_features[:, 0] *= noise
            # Re-normalize rows
            norms = np.linalg.norm(future_features, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            future_features = future_features / norms
            
            future_utils = self.model.expected_utility(future_features)
            future_values.append(float(np.max(future_utils)))
        
        discounted_future = self.config.discount_factor * np.mean(future_values)
        return discounted_future

    def decide(self, obs: Observation, verbose: bool = False) -> Action:
        M = obs.features.shape[0]
        if M == 0:
            return Search()
        
        X = obs.features
            
        expected_utils = self.model.expected_utility(X) # (M,)
        epistemic_uncs = self.model.epistemic_uncertainty(X) # (M,)
        worst_regrets = self.compute_worst_case_regret(X) # (M,)
        
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
        best_idx = int(np.argmax(expected_utils))
        best_id = obs.item_ids[best_idx]
        
        mu_star = expected_utils[best_idx]
        reg_star = worst_regrets[best_idx]
        var_star = epistemic_uncs[best_idx]
        
        action = None
        # 4. If mu >= tau_util, R <= eps_reg, and var <= eps_var -> Purchase
        if mu_star >= self.config.tau_util and reg_star <= self.config.eps_reg and var_star <= self.config.eps_var:
            action = Purchase(item_id=best_id)
            if verbose: print(f"Decision: Purchase Item {best_id}\n")
        else:
            # 5. Else: Estimate the expected Information Gain
            sigma2 = getattr(self.model, 'sigma2', 0.05)
            ig_query = 0.5 * np.log(1 + var_star / sigma2)
            ig_search = self.config.base_search_ig
            
            # 6. Estimate Wait value via MC rollouts [Proposal Algorithm 1]
            wait_value = self.estimate_wait_value(obs)
            current_value = float(mu_star)
            wait_advantage = wait_value - current_value
            
            if ig_query > ig_search and ig_query > max(wait_advantage, 0):
                action = QueryUser(item_id=best_id)
                if verbose: print(f"Decision: QueryUser (IG={ig_query:.3f})\n")
            elif wait_advantage > ig_search:
                action = Wait()
                if verbose: print(f"Decision: Wait (wait_value={wait_value:.3f}, current={current_value:.3f})\n")
            else:
                action = Search()
                if verbose: print(f"Decision: Search\n")
                    
        return action
