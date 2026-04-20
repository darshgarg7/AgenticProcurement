"""
Visualization script for test and experiment results.

Generates plots mapped to Proposal Section 7 evaluation criteria:
  1. Agent vs Baseline: average realized regret comparison
  2. Regret distribution (histogram) across episodes
  3. Epistemic uncertainty decay within a single episode
  4. Action distribution (Purchase / QueryUser / Search / Wait)
  5. Regret exceedance rate comparison
  6. Purchase rate across varying epistemic thresholds (sensitivity)

All figures saved to results/figures/.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

from environment.simulator import StochasticMarket
from decision.delegation_engine import DelegationEngine
from models.bayesian_user import BayesianPreferenceModel
from evaluation.metrics import MetricsTracker
from core.interfaces import Purchase, QueryUser, Wait, Search
from config.settings import EngineConfig, EnvConfig, ModelConfig

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'products.csv')

os.makedirs(FIG_DIR, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def run_detailed_episode(true_theta, data_path, prior_m0, prior_S0, max_steps=20):
    """Run one episode and return per-step telemetry."""
    d = len(true_theta)
    env_config = EnvConfig(data_path=data_path)
    engine_config = EngineConfig(eps_reg=1.0, eps_var=0.8, tau_util=0.0)
    model_config = ModelConfig(sigma2=0.05)

    model = BayesianPreferenceModel(d=d, m0=prior_m0, S0=prior_S0, config=model_config)
    engine = DelegationEngine(model, config=engine_config)
    env = StochasticMarket(config=env_config)
    tracker = MetricsTracker()

    history = {
        'epistemic_uncertainty': [],
        'expected_utility': [],
        'actions': [],
        'regret': None,
        'purchased': False,
        'steps': 0,
        'queries': 0,
    }

    for step in range(max_steps):
        obs = env.observe()
        if obs.features.shape[0] == 0:
            history['actions'].append('Search')
            env.step()
            continue

        epi_unc = model.epistemic_uncertainty(obs.features)
        exp_util = model.expected_utility(obs.features)
        best_idx = int(np.argmax(exp_util))

        history['epistemic_uncertainty'].append(float(epi_unc[best_idx]))
        history['expected_utility'].append(float(exp_util[best_idx]))

        action = engine.decide(obs)

        if isinstance(action, Purchase):
            idx = obs.item_ids.index(action.item_id)
            utils = obs.features @ true_theta
            best_u = float(np.max(utils))
            chosen_u = float(utils[idx])
            history['regret'] = best_u - chosen_u
            history['purchased'] = True
            history['steps'] = step + 1
            history['queries'] = tracker.queries
            history['actions'].append('Purchase')
            break
        elif isinstance(action, QueryUser):
            idx = obs.item_ids.index(action.item_id)
            x = obs.features[idx]
            y = float(true_theta @ x) + np.random.normal(0, np.sqrt(model_config.sigma2))
            model.update(x, y)
            tracker.record_query()
            history['actions'].append('QueryUser')
            history['queries'] = tracker.queries
        elif isinstance(action, Wait):
            tracker.record_delay()
            env.step()
            history['actions'].append('Wait')
        else:
            tracker.record_delay()
            env.step()
            history['actions'].append('Search')

    history['steps'] = history['steps'] or max_steps
    return history


def run_baseline(true_theta, data_path, prior_m0, prior_S0):
    d = len(true_theta)
    model = BayesianPreferenceModel(d=d, m0=prior_m0, S0=prior_S0)
    env = StochasticMarket(config=EnvConfig(data_path=data_path))
    obs = env.observe()
    if obs.features.shape[0] == 0:
        return {'regret': 0.0, 'purchased': False}
    exp_utils = model.expected_utility(obs.features)
    best_idx = int(np.argmax(exp_utils))
    utils = obs.features @ true_theta
    best_u = float(np.max(utils))
    chosen_u = float(utils[best_idx])
    return {'regret': best_u - chosen_u, 'purchased': True}


# ── Data Collection ──────────────────────────────────────────────────────────

def collect_experiment_data(num_episodes=50, d=8):
    """Run episodes across seeds and collect all telemetry."""
    agent_results = []
    baseline_results = []

    for seed in range(num_episodes):
        rng = np.random.RandomState(seed)
        true_theta = rng.randn(d)
        true_theta /= np.linalg.norm(true_theta)
        prior_m0 = true_theta + rng.normal(0, 0.4, d)
        prior_S0 = np.eye(d) * 0.5

        agent_hist = run_detailed_episode(true_theta, DATA_PATH, prior_m0, prior_S0)
        baseline_stat = run_baseline(true_theta, DATA_PATH, prior_m0, prior_S0)

        agent_results.append(agent_hist)
        baseline_results.append(baseline_stat)

    return agent_results, baseline_results


# ── Plot Functions ───────────────────────────────────────────────────────────

def plot_regret_comparison(agent_results, baseline_results):
    """Fig 1: Bar chart — average realized regret, agent vs baseline. [Proposal §7]"""
    agent_regrets = [r['regret'] for r in agent_results if r['purchased'] and r['regret'] is not None]
    baseline_regrets = [r['regret'] for r in baseline_results if r['purchased']]

    means = [np.mean(baseline_regrets), np.mean(agent_regrets)]
    stds = [np.std(baseline_regrets), np.std(agent_regrets)]
    labels = ['Baseline\n(Greedy)', 'Our Agent\n(Minimax Regret)']
    colors = ['#d9534f', '#5cb85c']

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Average Realized Regret', fontsize=12)
    ax.set_title('Agent vs Baseline: Realized Regret', fontsize=13, fontweight='bold')
    ax.set_ylim(bottom=0)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{m:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '1_regret_comparison.png'), dpi=150)
    plt.close()
    print("  Saved 1_regret_comparison.png")


def plot_regret_distribution(agent_results, baseline_results):
    """Fig 2: Histogram — regret distribution for agent vs baseline. [Proposal §7]"""
    agent_regrets = [r['regret'] for r in agent_results if r['purchased'] and r['regret'] is not None]
    baseline_regrets = [r['regret'] for r in baseline_results if r['purchased']]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(max(agent_regrets, default=0), max(baseline_regrets, default=0)) + 0.05, 25)
    ax.hist(baseline_regrets, bins=bins, alpha=0.6, label='Baseline (Greedy)', color='#d9534f', edgecolor='black')
    ax.hist(agent_regrets, bins=bins, alpha=0.6, label='Our Agent', color='#5cb85c', edgecolor='black')
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, label='ε_reg threshold (1.0)')
    ax.set_xlabel('Realized Regret', fontsize=12)
    ax.set_ylabel('Number of Episodes', fontsize=12)
    ax.set_title('Regret Distribution Across Episodes', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '2_regret_distribution.png'), dpi=150)
    plt.close()
    print("  Saved 2_regret_distribution.png")


def plot_epistemic_uncertainty_decay(agent_results):
    """Fig 3: Line plot — epistemic uncertainty decreasing over steps. [Proposal §4.1, §4.4]"""
    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot a few representative episodes
    plotted = 0
    for i, r in enumerate(agent_results):
        if len(r['epistemic_uncertainty']) >= 3:
            steps = list(range(1, len(r['epistemic_uncertainty']) + 1))
            ax.plot(steps, r['epistemic_uncertainty'], alpha=0.3, color='steelblue', linewidth=1)
            plotted += 1
        if plotted >= 30:
            break

    # Compute and plot the average trajectory
    max_len = max(len(r['epistemic_uncertainty']) for r in agent_results if r['epistemic_uncertainty'])
    avg_unc = []
    for step_idx in range(max_len):
        vals = [r['epistemic_uncertainty'][step_idx]
                for r in agent_results
                if len(r['epistemic_uncertainty']) > step_idx]
        if vals:
            avg_unc.append(np.mean(vals))
    if avg_unc:
        ax.plot(range(1, len(avg_unc) + 1), avg_unc, color='darkblue', linewidth=2.5,
                label='Mean across episodes', zorder=5)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Epistemic Uncertainty (x⊤Sx)', fontsize=12)
    ax.set_title('Epistemic Uncertainty Reduction Over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '3_epistemic_uncertainty_decay.png'), dpi=150)
    plt.close()
    print("  Saved 3_epistemic_uncertainty_decay.png")


def plot_action_distribution(agent_results):
    """Fig 4: Pie chart — distribution of actions taken. [Proposal Algorithm 1]"""
    all_actions = []
    for r in agent_results:
        all_actions.extend(r['actions'])

    counts = {}
    for a in all_actions:
        counts[a] = counts.get(a, 0) + 1

    labels = list(counts.keys())
    sizes = list(counts.values())
    color_map = {'Purchase': '#5cb85c', 'QueryUser': '#5bc0de', 'Search': '#f0ad4e', 'Wait': '#d9534f'}
    colors = [color_map.get(l, '#999999') for l in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 11})
    for t in autotexts:
        t.set_fontweight('bold')
    ax.set_title('Action Distribution Across All Episodes', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '4_action_distribution.png'), dpi=150)
    plt.close()
    print("  Saved 4_action_distribution.png")


def plot_exceedance_comparison(agent_results, baseline_results, threshold=1.0):
    """Fig 5: Bar chart — regret exceedance rate. [Proposal §7]"""
    agent_purchased = [r for r in agent_results if r['purchased'] and r['regret'] is not None]
    baseline_purchased = [r for r in baseline_results if r['purchased']]

    agent_exc = np.mean([1 if r['regret'] > threshold else 0 for r in agent_purchased]) * 100 if agent_purchased else 0
    baseline_exc = np.mean([1 if r['regret'] > threshold else 0 for r in baseline_purchased]) * 100 if baseline_purchased else 0

    agent_purchase_rate = len(agent_purchased) / len(agent_results) * 100
    baseline_purchase_rate = len(baseline_purchased) / len(baseline_results) * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Exceedance rate
    bars1 = axes[0].bar(['Baseline', 'Our Agent'], [baseline_exc, agent_exc],
                         color=['#d9534f', '#5cb85c'], edgecolor='black')
    axes[0].set_ylabel('Exceedance Rate (%)', fontsize=11)
    axes[0].set_title(f'Regret Exceedance (threshold={threshold})', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, max(baseline_exc, agent_exc, 10) + 5)
    for bar, val in zip(bars1, [baseline_exc, agent_exc]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # Purchase rate
    bars2 = axes[1].bar(['Baseline', 'Our Agent'], [baseline_purchase_rate, agent_purchase_rate],
                         color=['#d9534f', '#5cb85c'], edgecolor='black')
    axes[1].set_ylabel('Purchase Rate (%)', fontsize=11)
    axes[1].set_title('Purchase Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 110)
    for bar, val in zip(bars2, [baseline_purchase_rate, agent_purchase_rate]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.0f}%', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '5_exceedance_and_purchase_rate.png'), dpi=150)
    plt.close()
    print("  Saved 5_exceedance_and_purchase_rate.png")


def plot_threshold_sensitivity(d=8, num_episodes=30):
    """Fig 6: Line plot — purchase rate & avg regret vs ε_epi threshold. [Proposal §4.4]"""
    eps_epi_values = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    purchase_rates = []
    avg_regrets = []

    for eps_epi in eps_epi_values:
        purchased_count = 0
        regrets = []
        for seed in range(num_episodes):
            rng = np.random.RandomState(seed)
            true_theta = rng.randn(d)
            true_theta /= np.linalg.norm(true_theta)
            prior_m0 = true_theta + rng.normal(0, 0.4, d)
            prior_S0 = np.eye(d) * 0.5

            env_config = EnvConfig(data_path=DATA_PATH)
            engine_config = EngineConfig(eps_reg=1.0, eps_var=eps_epi, tau_util=0.0)
            model_config = ModelConfig(sigma2=0.05)

            model = BayesianPreferenceModel(d=d, m0=prior_m0, S0=prior_S0, config=model_config)
            engine = DelegationEngine(model, config=engine_config)
            env = StochasticMarket(config=env_config)

            for step in range(20):
                obs = env.observe()
                if obs.features.shape[0] == 0:
                    env.step()
                    continue
                action = engine.decide(obs)
                if isinstance(action, Purchase):
                    idx = obs.item_ids.index(action.item_id)
                    utils = obs.features @ true_theta
                    regrets.append(float(np.max(utils) - utils[idx]))
                    purchased_count += 1
                    break
                elif isinstance(action, QueryUser):
                    idx = obs.item_ids.index(action.item_id)
                    x = obs.features[idx]
                    y = float(true_theta @ x) + np.random.normal(0, np.sqrt(model_config.sigma2))
                    model.update(x, y)
                else:
                    env.step()

        purchase_rates.append(purchased_count / num_episodes * 100)
        avg_regrets.append(np.mean(regrets) if regrets else 0.0)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    color1 = '#5cb85c'
    color2 = '#5bc0de'

    ax1.set_xlabel('ε_epi (Epistemic Uncertainty Threshold)', fontsize=12)
    ax1.set_ylabel('Purchase Rate (%)', color=color1, fontsize=12)
    ax1.plot(eps_epi_values, purchase_rates, 'o-', color=color1, linewidth=2, markersize=6, label='Purchase Rate')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 110)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg Realized Regret', color=color2, fontsize=12)
    ax2.plot(eps_epi_values, avg_regrets, 's--', color=color2, linewidth=2, markersize=6, label='Avg Regret')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

    ax1.set_title('Sensitivity: Purchase Rate & Regret vs ε_epi Threshold', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '6_threshold_sensitivity.png'), dpi=150)
    plt.close()
    print("  Saved 6_threshold_sensitivity.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Collecting experiment data (50 episodes)...")
    agent_results, baseline_results = collect_experiment_data(num_episodes=50)

    # Save raw results as JSON
    def serialize(results):
        out = []
        for r in results:
            entry = dict(r)
            entry['epistemic_uncertainty'] = list(entry.get('epistemic_uncertainty', []))
            entry['expected_utility'] = list(entry.get('expected_utility', []))
            out.append(entry)
        return out

    with open(os.path.join(RESULTS_DIR, 'agent_results.json'), 'w') as f:
        json.dump(serialize(agent_results), f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'baseline_results.json'), 'w') as f:
        json.dump(serialize(baseline_results), f, indent=2)
    print(f"Raw results saved to {RESULTS_DIR}/\n")

    print("Generating figures...")
    plot_regret_comparison(agent_results, baseline_results)
    plot_regret_distribution(agent_results, baseline_results)
    plot_epistemic_uncertainty_decay(agent_results)
    plot_action_distribution(agent_results)
    plot_exceedance_comparison(agent_results, baseline_results)

    print("\nRunning threshold sensitivity analysis (this takes a moment)...")
    plot_threshold_sensitivity()

    print(f"\nAll figures saved to {FIG_DIR}/")

    # Print summary for the record
    agent_purchased = [r for r in agent_results if r['purchased']]
    baseline_purchased = [r for r in baseline_results if r['purchased']]
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Agent  — purchase rate: {len(agent_purchased)/len(agent_results)*100:.0f}%, "
          f"avg regret: {np.mean([r['regret'] for r in agent_purchased]):.4f}, "
          f"avg queries: {np.mean([r['queries'] for r in agent_results]):.1f}")
    print(f"Baseline — purchase rate: {len(baseline_purchased)/len(baseline_results)*100:.0f}%, "
          f"avg regret: {np.mean([r['regret'] for r in baseline_purchased]):.4f}")


if __name__ == '__main__':
    main()
