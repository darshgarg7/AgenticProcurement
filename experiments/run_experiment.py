import numpy as np
import numpy.typing as npt
from typing import Dict, Any, List

from environment.simulator import StochasticMarket
from decision.delegation_engine import DelegationEngine
from models.bayesian_user import BayesianPreferenceModel
from evaluation.metrics import MetricsTracker
from core.interfaces import Purchase, QueryUser, Wait, Search
from config.settings import EngineConfig, EnvConfig, ModelConfig

def true_utility(x: npt.NDArray[np.float64], true_theta: npt.NDArray[np.float64]) -> float:
    return float(np.dot(true_theta, x))

def compute_realized_regret(obs, true_theta, chosen_idx_in_obs: int) -> float:
    utils = obs.features @ true_theta
    best_u = np.max(utils)
    chosen_u = utils[chosen_idx_in_obs]
    return best_u - chosen_u

def run_episode(true_theta: npt.NDArray[np.float64], data_path: str, prior_m0: npt.NDArray[np.float64], prior_S0: npt.NDArray[np.float64], max_steps: int = 20, verbose: bool = False) -> Dict[str, Any]:
    d = len(true_theta)
    
    env_config = EnvConfig(data_path=data_path)
    engine_config = EngineConfig(eps_reg=1.0, eps_var=0.8, tau_util=0.0)
    model_config = ModelConfig(sigma2=0.05)
    
    model = BayesianPreferenceModel(d=d, m0=prior_m0, S0=prior_S0, config=model_config)
    engine = DelegationEngine(model, config=engine_config)
    env = StochasticMarket(config=env_config)
    tracker = MetricsTracker()
    
    if verbose:
        print("\n" + "="*40)
        print(" 1. Step-by-Step Output (what happens in one run)")
        print("="*40)
        
    for step in range(max_steps):
        obs = env.observe()
        if verbose:
            print(f"Step {step + 1}:")
            
        action = engine.decide(obs, verbose=verbose)
        
        if isinstance(action, Purchase):
            idx = obs.item_ids.index(action.item_id)
            
            # The agent prints the top 3 items by expected utility in step-by-step logs.
            # We track this explicit consideration set to define our regret bounds.
            expected_utils = engine.model.expected_utility(obs.features)
            top_indices = np.argsort(expected_utils)[::-1][:3]
            purchase_available_items = [obs.item_ids[i] for i in top_indices]
            purchase_available_features = obs.features[top_indices]
            
            # Validation Check
            assert action.item_id in purchase_available_items, "Validation Error: purchased_item is not in purchase_available_items"
            
            # Compute realized regret uniquely over the stored consideration set
            utils_consideration = purchase_available_features @ true_theta
            best_idx_consideration = int(np.argmax(utils_consideration))
            true_best_item = purchase_available_items[best_idx_consideration]
            best_u = utils_consideration[best_idx_consideration]
            
            purchased_idx_in_consideration = purchase_available_items.index(action.item_id)
            chosen_u = utils_consideration[purchased_idx_in_consideration]
            realized_reg = float(best_u - chosen_u)
            
            worst_regs = engine.compute_worst_case_regret(obs.features)
            estimated_wc_reg = worst_regs[idx]
            
            tracker.record_purchase(realized_regret=realized_reg, estimated_wc_regret=estimated_wc_reg, threshold=engine_config.eps_reg)
            
            if verbose:
                print("\n🧠 What this shows")
                print("The agent is not blindly picking the highest expected utility")
                print("It is balancing: \n  uncertainty\n  estimated_worst_case_regret")
                print("It may delay decisions → that's expected")
                
                print("\n📊 2. Final Output (per episode)\n")
                print("At the end of one run:\n")
                print(f"Final Decision: Purchased Item {action.item_id}")
                print(f"Final available items at purchase: {purchase_available_items}")
                print(f"True best among final available items: Item {true_best_item}")
                print(f"Realized regret under true θ: {realized_reg:.2f}\n")
                print(f"Total queries: {tracker.queries}")
                print(f"Steps taken: {step + 1}\n")
            break
            
        elif isinstance(action, QueryUser):
            idx = obs.item_ids.index(action.item_id)
            x = obs.features[idx]
            
            y_obs = true_utility(x, true_theta) + np.random.normal(0, np.sqrt(model_config.sigma2))
            model.update(x, y_obs)
            tracker.record_query()
            if verbose: print("Updated beliefs after user feedback")
            
        elif isinstance(action, Wait) or isinstance(action, Search):
            tracker.record_delay()
            env.step()
            
    return tracker.get_stats()

def run_baseline_episode(true_theta: npt.NDArray[np.float64], data_path: str, prior_m0: npt.NDArray[np.float64], prior_S0: npt.NDArray[np.float64]) -> Dict[str, Any]:
    d = len(true_theta)
    
    model = BayesianPreferenceModel(d=d, m0=prior_m0, S0=prior_S0)
    engine = DelegationEngine(model)
    env = StochasticMarket(config=EnvConfig(data_path=data_path))
    
    obs = env.observe()
    if len(obs.features) == 0:
        return {'purchased': False, 'queries': 0, 'delays': 0, 'realized_regret': 0.0, 'estimated_worst_case_regret': 0.0, 'exceeded_regret': False}
        
    expected_utils = model.expected_utility(obs.features)
    best_idx = int(np.argmax(expected_utils))
    
    realized_reg = compute_realized_regret(obs, true_theta, best_idx)
    
    return {
        'queries': 0,
        'delays': 0,
        'purchased': True,
        'realized_regret': realized_reg,
        'estimated_worst_case_regret': 0.0,
        'exceeded_regret': realized_reg > 1.0 # Default threshold 1.0 same as agent default
    }

def run_experiment(data_path: str, num_episodes: int = 50, d: int = 8) -> None:
    agent_results = []
    baseline_results = []
    
    for ep in range(num_episodes):
        true_theta = np.random.randn(d)
        true_theta /= np.linalg.norm(true_theta)
        
        prior_m0 = true_theta + np.random.normal(0, 0.4, d)
        prior_S0 = np.eye(d) * 0.5
        
        is_verbose = (ep == 0)
        
        agent_stats = run_episode(true_theta, data_path, prior_m0, prior_S0, verbose=is_verbose)
        baseline_stats = run_baseline_episode(true_theta, data_path, prior_m0, prior_S0)
        
        agent_results.append(agent_stats)
        baseline_results.append(baseline_stats)
        
    def aggregate(results):
        purchases = [r for r in results if r['purchased']]
        return {
            'purchase_rate': len(purchases) / len(results) if results else 0,
            'avg_queries': np.mean([r['queries'] for r in results]) if results else 0,
            'avg_realized_regret': np.mean([r['realized_regret'] for r in purchases]) if purchases else 0.0,
            'max_est_wc_regret': np.max([r['estimated_worst_case_regret'] for r in purchases]) if purchases else 0.0,
            'exceedance_rate': np.mean([1 if r['exceeded_regret'] else 0 for r in purchases]) if purchases else 0.0
        }
        
    base_agg = aggregate(baseline_results)
    agent_agg = aggregate(agent_results)
        
    print("📈 3. Aggregated Metrics (IMPORTANT)\n")
    print("Core metrics:")
    print("1. Average Realized Regret")
    print(f"Average Realized Regret: {agent_agg['avg_realized_regret']:.2f}")
    print("\n👉 how far off your decisions are on average\n")
    
    print("2. Max Estimated Worst-Case Regret Observed")
    print(f"Max Estimated Worst-Case Regret: {agent_agg['max_est_wc_regret']:.2f}")
    print("\n👉 robustness measure\n")
    
    print("3. Regret Exceedance Rate")
    print(f"% of decisions with regret > threshold: {agent_agg['exceedance_rate']*100:.0f}%")
    print("\n👉 safety of your system\n")
    
    print("4. Query Rate")
    print(f"Average Queries per Episode: {agent_agg['avg_queries']:.1f}")
    print("\n👉 how often the agent asks for help\n")
    
    print("5. Purchase Rate")
    print(f"Purchase Rate: {agent_agg['purchase_rate']*100:.0f}%")
    print("\n👉 how often the agent commits\n")
    
    print("⚖️ 4. Comparison Output (VERY IMPORTANT)\n")
    print("You should compare against baseline:\n")
    print("Baseline (greedy):")
    print(f"Average Realized Regret: {base_agg['avg_realized_regret']:.2f}")
    print(f"Regret Exceedance: {base_agg['exceedance_rate']*100:.0f}%\n")
    
    print("Our Model:")
    print(f"Average Realized Regret: {agent_agg['avg_realized_regret']:.2f}")
