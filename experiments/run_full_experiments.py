"""
Full experiment suite for the project.

Runs:
  1. Agent vs Baseline comparison (100 episodes, multiple seeds)
  2. Persona experiments (budget_shopper, quality_maximizer, balanced)
  3. Ablation studies:
     a. No epistemic uncertainty gate
     b. No minimax regret gate
     c. No market dynamics (alpha=0, price_fluctuation=0)
  4. Tightened threshold experiments (eps_reg=0.3)

All results saved to results/ as JSON.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from typing import Dict, Any, List

from environment.simulator import StochasticMarket
from decision.delegation_engine import DelegationEngine
from models.bayesian_user import BayesianPreferenceModel
from evaluation.metrics import MetricsTracker
from core.interfaces import Purchase, QueryUser, Wait, Search
from config.settings import EngineConfig, EnvConfig, ModelConfig, PersonaConfig

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'products.csv')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Core episode runners ────────────────────────────────────────────────────

def run_episode(true_theta, data_path, prior_m0, prior_S0,
                engine_config=None, env_config=None, model_config=None,
                max_steps=20) -> Dict[str, Any]:
    d = len(true_theta)
    if engine_config is None:
        engine_config = EngineConfig(eps_reg=0.8, eps_var=0.8, tau_util=0.0)
    if env_config is None:
        env_config = EnvConfig(data_path=data_path)
    if model_config is None:
        model_config = ModelConfig(sigma2=0.05)

    model = BayesianPreferenceModel(d=d, m0=prior_m0, S0=prior_S0, config=model_config)
    engine = DelegationEngine(model, config=engine_config)
    env = StochasticMarket(config=env_config)
    tracker = MetricsTracker()

    step_data = []

    for step in range(max_steps):
        obs = env.observe()
        if obs.features.shape[0] == 0:
            env.step()
            tracker.record_delay()
            step_data.append({'step': step, 'action': 'Search', 'epi_unc': None, 'exp_util': None})
            continue

        epi_unc = model.epistemic_uncertainty(obs.features)
        exp_util = model.expected_utility(obs.features)
        best_idx = int(np.argmax(exp_util))

        action = engine.decide(obs)

        if isinstance(action, Purchase):
            idx = obs.item_ids.index(action.item_id)
            utils = obs.features @ true_theta
            best_u = float(np.max(utils))
            chosen_u = float(utils[idx])
            realized_reg = best_u - chosen_u

            worst_regs = engine.compute_worst_case_regret(obs.features)
            estimated_wc_reg = float(worst_regs[idx])

            tracker.record_purchase(realized_regret=realized_reg,
                                    estimated_wc_regret=estimated_wc_reg,
                                    threshold=engine_config.eps_reg)

            step_data.append({
                'step': step, 'action': 'Purchase',
                'epi_unc': float(epi_unc[best_idx]),
                'exp_util': float(exp_util[best_idx]),
            })
            break

        elif isinstance(action, QueryUser):
            idx = obs.item_ids.index(action.item_id)
            x = obs.features[idx]
            y = float(true_theta @ x) + np.random.normal(0, np.sqrt(model_config.sigma2))
            model.update(x, y)
            tracker.record_query()
            step_data.append({
                'step': step, 'action': 'QueryUser',
                'epi_unc': float(epi_unc[best_idx]),
                'exp_util': float(exp_util[best_idx]),
            })

        elif isinstance(action, Wait):
            tracker.record_delay()
            env.step()
            step_data.append({
                'step': step, 'action': 'Wait',
                'epi_unc': float(epi_unc[best_idx]),
                'exp_util': float(exp_util[best_idx]),
            })

        else:  # Search
            tracker.record_delay()
            env.step()
            step_data.append({
                'step': step, 'action': 'Search',
                'epi_unc': float(epi_unc[best_idx]),
                'exp_util': float(exp_util[best_idx]),
            })

    stats = tracker.get_stats()
    stats['steps_taken'] = len(step_data)
    stats['step_data'] = step_data
    return stats


def run_baseline_episode(true_theta, data_path, prior_m0, prior_S0,
                         eps_reg=0.3) -> Dict[str, Any]:
    d = len(true_theta)
    model = BayesianPreferenceModel(d=d, m0=prior_m0, S0=prior_S0)
    env = StochasticMarket(config=EnvConfig(data_path=data_path))
    obs = env.observe()

    if obs.features.shape[0] == 0:
        return {'purchased': False, 'queries': 0, 'delays': 0,
                'realized_regret': 0.0, 'estimated_worst_case_regret': 0.0,
                'exceeded_regret': False}

    exp_utils = model.expected_utility(obs.features)
    best_idx = int(np.argmax(exp_utils))
    utils = obs.features @ true_theta
    best_u = float(np.max(utils))
    chosen_u = float(utils[best_idx])
    realized_reg = best_u - chosen_u

    return {
        'queries': 0, 'delays': 0, 'purchased': True,
        'realized_regret': realized_reg,
        'estimated_worst_case_regret': 0.0,
        'exceeded_regret': realized_reg > eps_reg
    }


# ── Aggregation helper ──────────────────────────────────────────────────────

def aggregate(results: List[Dict]) -> Dict[str, float]:
    purchases = [r for r in results if r['purchased']]
    return {
        'num_episodes': len(results),
        'purchase_rate': len(purchases) / len(results) if results else 0,
        'avg_queries': float(np.mean([r['queries'] for r in results])) if results else 0,
        'avg_delays': float(np.mean([r['delays'] for r in results])) if results else 0,
        'avg_realized_regret': float(np.mean([r['realized_regret'] for r in purchases])) if purchases else 0.0,
        'std_realized_regret': float(np.std([r['realized_regret'] for r in purchases])) if purchases else 0.0,
        'max_realized_regret': float(np.max([r['realized_regret'] for r in purchases])) if purchases else 0.0,
        'avg_est_wc_regret': float(np.mean([r['estimated_worst_case_regret'] for r in purchases])) if purchases else 0.0,
        'exceedance_rate': float(np.mean([1 if r['exceeded_regret'] else 0 for r in purchases])) if purchases else 0.0,
    }


# ── Experiment 1: Agent vs Baseline (tightened threshold) ───────────────────

def experiment_agent_vs_baseline(num_episodes=100, d=8, eps_reg=0.3):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1: Agent vs Baseline (eps_reg={eps_reg}, {num_episodes} episodes)")
    print(f"{'='*60}")

    agent_results = []
    baseline_results = []

    for seed in range(num_episodes):
        rng = np.random.RandomState(seed)
        true_theta = rng.randn(d)
        true_theta /= np.linalg.norm(true_theta)
        prior_m0 = true_theta + rng.normal(0, 0.4, d)
        prior_S0 = np.eye(d) * 0.5

        engine_config = EngineConfig(eps_reg=eps_reg, eps_var=0.8, tau_util=0.0)

        agent_stats = run_episode(true_theta, DATA_PATH, prior_m0, prior_S0,
                                  engine_config=engine_config,
                                  max_steps=30)
        baseline_stats = run_baseline_episode(true_theta, DATA_PATH, prior_m0, prior_S0,
                                              eps_reg=eps_reg)

        agent_results.append(agent_stats)
        baseline_results.append(baseline_stats)

    agent_agg = aggregate(agent_results)
    baseline_agg = aggregate(baseline_results)

    print(f"\nAgent:    avg_regret={agent_agg['avg_realized_regret']:.4f}, "
          f"purchase_rate={agent_agg['purchase_rate']*100:.0f}%, "
          f"exceedance={agent_agg['exceedance_rate']*100:.1f}%, "
          f"avg_queries={agent_agg['avg_queries']:.1f}")
    print(f"Baseline: avg_regret={baseline_agg['avg_realized_regret']:.4f}, "
          f"purchase_rate={baseline_agg['purchase_rate']*100:.0f}%, "
          f"exceedance={baseline_agg['exceedance_rate']*100:.1f}%")

    return {
        'agent': {'results': agent_results, 'aggregate': agent_agg},
        'baseline': {'results': baseline_results, 'aggregate': baseline_agg},
        'config': {'eps_reg': eps_reg, 'eps_var': 0.8, 'num_episodes': num_episodes}
    }


# ── Experiment 2: Persona experiments ───────────────────────────────────────

def experiment_personas(num_episodes_per_persona=100, d=8, eps_reg=0.3):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2: Persona Experiments ({num_episodes_per_persona} episodes each)")
    print(f"{'='*60}")

    persona_factories = [
        PersonaConfig.budget_shopper,
        PersonaConfig.quality_maximizer,
        PersonaConfig.balanced,
    ]

    all_persona_results = {}

    for factory in persona_factories:
        persona_agent_results = []
        persona_baseline_results = []

        for seed in range(num_episodes_per_persona):
            persona = factory(d=d, seed=seed)

            engine_config = EngineConfig(eps_reg=eps_reg, eps_var=0.8, tau_util=0.0)

            agent_stats = run_episode(persona.true_theta, DATA_PATH,
                                      persona.prior_mean, persona.prior_cov,
                                      engine_config=engine_config, max_steps=30)
            baseline_stats = run_baseline_episode(persona.true_theta, DATA_PATH,
                                                  persona.prior_mean, persona.prior_cov,
                                                  eps_reg=eps_reg)

            persona_agent_results.append(agent_stats)
            persona_baseline_results.append(baseline_stats)

        agent_agg = aggregate(persona_agent_results)
        baseline_agg = aggregate(persona_baseline_results)

        print(f"\n  {persona.name}:")
        print(f"    Agent:    avg_regret={agent_agg['avg_realized_regret']:.4f}, "
              f"purchase_rate={agent_agg['purchase_rate']*100:.0f}%, "
              f"exceedance={agent_agg['exceedance_rate']*100:.1f}%, "
              f"avg_queries={agent_agg['avg_queries']:.1f}")
        print(f"    Baseline: avg_regret={baseline_agg['avg_realized_regret']:.4f}, "
              f"exceedance={baseline_agg['exceedance_rate']*100:.1f}%")

        all_persona_results[persona.name] = {
            'agent': {'aggregate': agent_agg},
            'baseline': {'aggregate': baseline_agg},
        }

    return all_persona_results


# ── Experiment 3: Ablation studies ──────────────────────────────────────────

def experiment_ablations(num_episodes=100, d=8, eps_reg=0.3):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 3: Ablation Studies ({num_episodes} episodes each)")
    print(f"{'='*60}")

    ablations = {
        'full_model': {
            'engine_config': EngineConfig(eps_reg=eps_reg, eps_var=0.8, tau_util=0.0),
            'env_config': EnvConfig(data_path=DATA_PATH),
        },
        'no_epistemic_gate': {
            'engine_config': EngineConfig(eps_reg=eps_reg, eps_var=1e6, tau_util=0.0),  # effectively disabled
            'env_config': EnvConfig(data_path=DATA_PATH),
        },
        'no_regret_gate': {
            'engine_config': EngineConfig(eps_reg=1e6, eps_var=0.8, tau_util=0.0),  # effectively disabled
            'env_config': EnvConfig(data_path=DATA_PATH),
        },
        'no_market_dynamics': {
            'engine_config': EngineConfig(eps_reg=eps_reg, eps_var=0.8, tau_util=0.0),
            'env_config': EnvConfig(data_path=DATA_PATH, alpha=0.0, price_fluctuation=0.0),
        },
    }

    ablation_results = {}

    for name, cfg in ablations.items():
        results = []
        for seed in range(num_episodes):
            rng = np.random.RandomState(seed)
            true_theta = rng.randn(d)
            true_theta /= np.linalg.norm(true_theta)
            prior_m0 = true_theta + rng.normal(0, 0.4, d)
            prior_S0 = np.eye(d) * 0.5

            stats = run_episode(true_theta, DATA_PATH, prior_m0, prior_S0,
                                engine_config=cfg['engine_config'],
                                env_config=cfg['env_config'], max_steps=30)
            results.append(stats)

        agg = aggregate(results)
        print(f"\n  {name}:")
        print(f"    avg_regret={agg['avg_realized_regret']:.4f}, "
              f"purchase_rate={agg['purchase_rate']*100:.0f}%, "
              f"exceedance={agg['exceedance_rate']*100:.1f}%, "
              f"avg_queries={agg['avg_queries']:.1f}")

        ablation_results[name] = {'aggregate': agg}

    return ablation_results


# ── Experiment 4: Multi-seed robustness ─────────────────────────────────────

def experiment_multi_seed_robustness(num_outer_seeds=5, num_episodes=50, d=8, eps_reg=0.3):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 4: Multi-Seed Robustness ({num_outer_seeds} seeds x {num_episodes} episodes)")
    print(f"{'='*60}")

    seed_results = []

    for outer_seed in range(num_outer_seeds):
        np.random.seed(outer_seed * 1000)
        agent_results = []
        baseline_results = []

        for ep in range(num_episodes):
            rng = np.random.RandomState(outer_seed * 1000 + ep)
            true_theta = rng.randn(d)
            true_theta /= np.linalg.norm(true_theta)
            prior_m0 = true_theta + rng.normal(0, 0.4, d)
            prior_S0 = np.eye(d) * 0.5

            engine_config = EngineConfig(eps_reg=eps_reg, eps_var=0.8, tau_util=0.0)

            agent_stats = run_episode(true_theta, DATA_PATH, prior_m0, prior_S0,
                                      engine_config=engine_config, max_steps=30)
            baseline_stats = run_baseline_episode(true_theta, DATA_PATH, prior_m0, prior_S0,
                                                  eps_reg=eps_reg)
            agent_results.append(agent_stats)
            baseline_results.append(baseline_stats)

        agent_agg = aggregate(agent_results)
        baseline_agg = aggregate(baseline_results)

        print(f"\n  Seed {outer_seed}: agent_regret={agent_agg['avg_realized_regret']:.4f}, "
              f"baseline_regret={baseline_agg['avg_realized_regret']:.4f}, "
              f"agent_purchase_rate={agent_agg['purchase_rate']*100:.0f}%")

        seed_results.append({
            'outer_seed': outer_seed,
            'agent': agent_agg,
            'baseline': baseline_agg,
        })

    # Cross-seed summary
    agent_regrets = [s['agent']['avg_realized_regret'] for s in seed_results]
    baseline_regrets = [s['baseline']['avg_realized_regret'] for s in seed_results]
    print(f"\n  Cross-seed agent regret: {np.mean(agent_regrets):.4f} ± {np.std(agent_regrets):.4f}")
    print(f"  Cross-seed baseline regret: {np.mean(baseline_regrets):.4f} ± {np.std(baseline_regrets):.4f}")

    return seed_results


# ── Main ────────────────────────────────────────────────────────────────────

def strip_step_data(results_dict):
    """Remove step_data from results for JSON serialization (keep file sizes small)."""
    if isinstance(results_dict, dict):
        out = {}
        for k, v in results_dict.items():
            if k == 'results' and isinstance(v, list):
                out[k] = [{kk: vv for kk, vv in r.items() if kk != 'step_data'} for r in v]
            else:
                out[k] = strip_step_data(v)
        return out
    elif isinstance(results_dict, list):
        return [strip_step_data(item) for item in results_dict]
    return results_dict


def main():
    all_results = {}

    # Experiment 1
    exp1 = experiment_agent_vs_baseline(num_episodes=100, eps_reg=0.8)
    all_results['agent_vs_baseline'] = strip_step_data(exp1)

    # Experiment 2
    exp2 = experiment_personas(num_episodes_per_persona=100, eps_reg=0.8)
    all_results['personas'] = exp2

    # Experiment 3
    exp3 = experiment_ablations(num_episodes=100, eps_reg=0.8)
    all_results['ablations'] = exp3

    # Experiment 4
    exp4 = experiment_multi_seed_robustness(num_outer_seeds=5, num_episodes=50, eps_reg=0.8)
    all_results['multi_seed_robustness'] = exp4

    # Save all results
    output_path = os.path.join(RESULTS_DIR, 'full_experiment_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {output_path}")


if __name__ == '__main__':
    main()
