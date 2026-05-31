"""Compact experiment runner used by the top-level script."""

from typing import Any

import numpy as np
import numpy.typing as npt

from config.settings import EngineConfig
from core.episode import (
    run_agent_episode,
)
from core.episode import (
    run_baseline_episode as run_greedy_baseline_episode,
)


def run_episode(
    true_theta: npt.NDArray[np.float64],
    data_path: str,
    prior_m0: npt.NDArray[np.float64],
    prior_S0: npt.NDArray[np.float64],
    max_steps: int = 20,
    verbose: bool = False,
    rng=None,
) -> dict[str, Any]:
    """Run one agent episode with the compact runner defaults."""
    engine_config = EngineConfig(eps_reg=1.0, eps_var=0.8, tau_util=0.0)
    stats = run_agent_episode(
        true_theta=true_theta,
        data_path=data_path,
        prior_m0=prior_m0,
        prior_S0=prior_S0,
        engine_config=engine_config,
        max_steps=max_steps,
        rng=rng,
        top_k=3,
    )

    if verbose:
        print("\n" + "="*40)
        print(" 1. Step-by-Step Output (what happens in one run)")
        print("="*40)
        for step in stats["step_data"]:
            print(
                f"Step {step['step']}: {step['action']} "
                f"item={step['item_id']} regret={step['realized_regret']}"
            )

    return stats

def run_baseline_episode(true_theta: npt.NDArray[np.float64], data_path: str, prior_m0: npt.NDArray[np.float64], prior_S0: npt.NDArray[np.float64]) -> dict[str, Any]:
    """Run the greedy first-step baseline for one compact-runner episode."""
    return run_greedy_baseline_episode(true_theta, data_path, prior_m0, prior_S0, eps_reg=1.0)

def run_experiment(data_path: str, num_episodes: int = 50, d: int = 8) -> None:
    """Run a small agent-vs-baseline experiment and print aggregate metrics."""
    agent_results = []
    baseline_results = []
    
    for ep in range(num_episodes):
        rng = np.random.RandomState(ep)
        true_theta = rng.randn(d)
        true_theta /= np.linalg.norm(true_theta)
        
        prior_m0 = true_theta + rng.normal(0, 0.4, d)
        prior_S0 = np.eye(d) * 0.5
        
        is_verbose = (ep == 0)
        
        agent_stats = run_episode(true_theta, data_path, prior_m0, prior_S0, verbose=is_verbose, rng=rng)
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
