"""
Updated visualization script that generates figures from full experiment results.

Reads results/full_experiment_results.json and generates:
  1. Agent vs Baseline regret comparison (bar chart)
  2. Regret distribution histogram
  3. Epistemic uncertainty decay
  4. Action distribution pie chart
  5. Exceedance & purchase rate comparison
  6. Threshold sensitivity sweep
  7. Persona comparison (grouped bar chart)           [NEW]
  8. Ablation study results (grouped bar chart)        [NEW]
  9. Multi-seed robustness (bar chart)                 [NEW]

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
from experiments.run_full_experiments import run_episode as run_agent_episode, run_baseline_episode

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'products.csv')

os.makedirs(FIG_DIR, exist_ok=True)


# ── Load experiment results ─────────────────────────────────────────────────

def load_results():
    path = os.path.join(RESULTS_DIR, 'full_experiment_results.json')
    with open(path, 'r') as f:
        return json.load(f)


# ── Helpers ──────────────────────────────────────────────────────────────────

def run_detailed_episode(true_theta, data_path, prior_m0, prior_S0, max_steps=30):
    """Run one episode and return per-step telemetry."""
    d = len(true_theta)
    env_config = EnvConfig(data_path=data_path)
    engine_config = EngineConfig(eps_reg=0.8, eps_var=0.8, tau_util=0.0)
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


def collect_detailed_data(num_episodes=50, d=8):
    """Run episodes with per-step telemetry for uncertainty/action plots."""
    agent_results = []

    for seed in range(num_episodes):
        rng = np.random.RandomState(seed)
        true_theta = rng.randn(d)
        true_theta /= np.linalg.norm(true_theta)
        prior_m0 = true_theta + rng.normal(0, 0.4, d)
        prior_S0 = np.eye(d) * 0.5

        agent_hist = run_detailed_episode(true_theta, DATA_PATH, prior_m0, prior_S0)
        agent_results.append(agent_hist)

    return agent_results


# ── Plot Functions ───────────────────────────────────────────────────────────

def plot_regret_comparison(exp_data):
    """Fig 1: Bar chart — average realized regret, agent vs baseline. [Proposal §7]"""
    agent_agg = exp_data['agent_vs_baseline']['agent']['aggregate']
    baseline_agg = exp_data['agent_vs_baseline']['baseline']['aggregate']

    means = [baseline_agg['avg_realized_regret'], agent_agg['avg_realized_regret']]
    stds = [baseline_agg['std_realized_regret'], agent_agg['std_realized_regret']]
    labels = ['Baseline\n(Greedy)', 'Our Agent\n(Minimax Regret)']
    colors = ['#d9534f', '#5cb85c']

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Average Realized Regret', fontsize=12)
    ax.set_title('Agent vs Baseline: Realized Regret (ε_reg=0.8)', fontsize=13, fontweight='bold')
    ax.set_ylim(bottom=0)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{m:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '1_regret_comparison.png'), dpi=150)
    plt.close()
    print("  Saved 1_regret_comparison.png")


def plot_regret_distribution(exp_data):
    """Fig 2: Histogram — regret distribution for agent vs baseline. [Proposal §7]"""
    agent_results = exp_data['agent_vs_baseline']['agent']['results']
    baseline_results = exp_data['agent_vs_baseline']['baseline']['results']

    agent_regrets = [r['realized_regret'] for r in agent_results if r['purchased']]
    baseline_regrets = [r['realized_regret'] for r in baseline_results if r['purchased']]

    fig, ax = plt.subplots(figsize=(7, 4))
    max_val = max(max(agent_regrets, default=0), max(baseline_regrets, default=0)) + 0.05
    bins = np.linspace(0, max_val, 25)
    ax.hist(baseline_regrets, bins=bins, alpha=0.6, label='Baseline (Greedy)', color='#d9534f', edgecolor='black')
    ax.hist(agent_regrets, bins=bins, alpha=0.6, label='Our Agent', color='#5cb85c', edgecolor='black')
    ax.axvline(x=0.8, color='black', linestyle='--', linewidth=1, label='ε_reg threshold (0.8)')
    ax.set_xlabel('Realized Regret', fontsize=12)
    ax.set_ylabel('Number of Episodes', fontsize=12)
    ax.set_title('Regret Distribution Across Episodes', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '2_regret_distribution.png'), dpi=150)
    plt.close()
    print("  Saved 2_regret_distribution.png")


def plot_epistemic_uncertainty_decay(agent_detailed):
    """Fig 3: Line plot — epistemic uncertainty decreasing over steps. [Proposal §4.1, §4.4]"""
    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot a few representative episodes
    plotted = 0
    for i, r in enumerate(agent_detailed):
        if len(r['epistemic_uncertainty']) >= 3:
            steps = list(range(1, len(r['epistemic_uncertainty']) + 1))
            ax.plot(steps, r['epistemic_uncertainty'], alpha=0.3, color='steelblue', linewidth=1)
            plotted += 1
        if plotted >= 30:
            break

    # Compute and plot the average trajectory
    max_len = max(len(r['epistemic_uncertainty']) for r in agent_detailed if r['epistemic_uncertainty'])
    avg_unc = []
    for step_idx in range(max_len):
        vals = [r['epistemic_uncertainty'][step_idx]
                for r in agent_detailed
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


def plot_action_distribution(agent_detailed):
    """Fig 4: Pie chart — distribution of actions taken. [Proposal Algorithm 1]"""
    all_actions = []
    for r in agent_detailed:
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


def plot_exceedance_comparison(exp_data):
    """Fig 5: Bar chart — regret exceedance rate & purchase rate. [Proposal §7]"""
    agent_agg = exp_data['agent_vs_baseline']['agent']['aggregate']
    baseline_agg = exp_data['agent_vs_baseline']['baseline']['aggregate']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    agent_exc = agent_agg['exceedance_rate'] * 100
    baseline_exc = baseline_agg['exceedance_rate'] * 100

    # Exceedance rate
    bars1 = axes[0].bar(['Baseline', 'Our Agent'], [baseline_exc, agent_exc],
                         color=['#d9534f', '#5cb85c'], edgecolor='black')
    axes[0].set_ylabel('Exceedance Rate (%)', fontsize=11)
    axes[0].set_title('Regret Exceedance (ε_reg=0.8)', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, max(baseline_exc, agent_exc, 10) + 5)
    for bar, val in zip(bars1, [baseline_exc, agent_exc]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # Purchase rate
    agent_pr = agent_agg['purchase_rate'] * 100
    baseline_pr = baseline_agg['purchase_rate'] * 100

    bars2 = axes[1].bar(['Baseline', 'Our Agent'], [baseline_pr, agent_pr],
                         color=['#d9534f', '#5cb85c'], edgecolor='black')
    axes[1].set_ylabel('Purchase Rate (%)', fontsize=11)
    axes[1].set_title('Purchase Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 110)
    for bar, val in zip(bars2, [baseline_pr, agent_pr]):
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
            engine_config = EngineConfig(eps_reg=0.8, eps_var=eps_epi, tau_util=0.0)
            model_config = ModelConfig(sigma2=0.05)

            model = BayesianPreferenceModel(d=d, m0=prior_m0, S0=prior_S0, config=model_config)
            engine = DelegationEngine(model, config=engine_config)
            env = StochasticMarket(config=env_config)

            for step in range(30):
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

    ax1.set_title('Sensitivity: Purchase Rate & Regret vs ε_epi', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '6_threshold_sensitivity.png'), dpi=150)
    plt.close()
    print("  Saved 6_threshold_sensitivity.png")


# ── NEW Plots from Full Experiments ─────────────────────────────────────────

def plot_persona_comparison(exp_data):
    """Fig 7: Grouped bar chart — regret & purchase rate by persona. [Proposal §6]"""
    personas = exp_data['personas']
    names = list(personas.keys())

    agent_regrets = [personas[n]['agent']['aggregate']['avg_realized_regret'] for n in names]
    baseline_regrets = [personas[n]['baseline']['aggregate']['avg_realized_regret'] for n in names]
    agent_pr = [personas[n]['agent']['aggregate']['purchase_rate'] * 100 for n in names]
    baseline_pr = [personas[n]['baseline']['aggregate']['purchase_rate'] * 100 for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(names))
    width = 0.35

    # Regret comparison
    bars1 = axes[0].bar(x - width/2, baseline_regrets, width, label='Baseline', color='#d9534f', edgecolor='black')
    bars2 = axes[0].bar(x + width/2, agent_regrets, width, label='Agent', color='#5cb85c', edgecolor='black')
    axes[0].set_ylabel('Avg Realized Regret', fontsize=12)
    axes[0].set_title('Regret by Persona', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace('_', '\n') for n in names], fontsize=10)
    axes[0].legend()
    axes[0].set_ylim(bottom=0)
    for bar, val in zip(bars1, baseline_regrets):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{val:.3f}', ha='center', fontsize=9)
    for bar, val in zip(bars2, agent_regrets):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{val:.3f}', ha='center', fontsize=9)

    # Purchase rate comparison
    bars3 = axes[1].bar(x - width/2, baseline_pr, width, label='Baseline', color='#d9534f', edgecolor='black')
    bars4 = axes[1].bar(x + width/2, agent_pr, width, label='Agent', color='#5cb85c', edgecolor='black')
    axes[1].set_ylabel('Purchase Rate (%)', fontsize=12)
    axes[1].set_title('Purchase Rate by Persona', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.replace('_', '\n') for n in names], fontsize=10)
    axes[1].legend()
    axes[1].set_ylim(0, 110)
    for bar, val in zip(bars3, baseline_pr):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.0f}%', ha='center', fontsize=9)
    for bar, val in zip(bars4, agent_pr):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.0f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '7_persona_comparison.png'), dpi=150)
    plt.close()
    print("  Saved 7_persona_comparison.png")


def plot_ablation_results(exp_data):
    """Fig 8: Grouped bar chart — ablation study. [Proposal §7]"""
    ablations = exp_data['ablations']
    names = list(ablations.keys())
    display_names = ['Full Model', 'No Epistemic\nGate', 'No Regret\nGate', 'No Market\nDynamics']

    regrets = [ablations[n]['aggregate']['avg_realized_regret'] for n in names]
    purchase_rates = [ablations[n]['aggregate']['purchase_rate'] * 100 for n in names]
    queries = [ablations[n]['aggregate']['avg_queries'] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#5cb85c', '#5bc0de', '#f0ad4e', '#d9534f']

    # Regret
    bars = axes[0].bar(display_names, regrets, color=colors, edgecolor='black')
    axes[0].set_ylabel('Avg Realized Regret', fontsize=11)
    axes[0].set_title('Ablation: Regret', fontsize=12, fontweight='bold')
    axes[0].set_ylim(bottom=0)
    for bar, val in zip(bars, regrets):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    # Purchase rate
    bars = axes[1].bar(display_names, purchase_rates, color=colors, edgecolor='black')
    axes[1].set_ylabel('Purchase Rate (%)', fontsize=11)
    axes[1].set_title('Ablation: Purchase Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 110)
    for bar, val in zip(bars, purchase_rates):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')

    # Queries
    bars = axes[2].bar(display_names, queries, color=colors, edgecolor='black')
    axes[2].set_ylabel('Avg Queries per Episode', fontsize=11)
    axes[2].set_title('Ablation: Query Rate', fontsize=12, fontweight='bold')
    axes[2].set_ylim(bottom=0)
    for bar, val in zip(bars, queries):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '8_ablation_results.png'), dpi=150)
    plt.close()
    print("  Saved 8_ablation_results.png")


def plot_multi_seed_robustness(exp_data):
    """Fig 9: Bar chart — multi-seed robustness. [Proposal §7]"""
    seeds = exp_data['multi_seed_robustness']
    agent_regrets = [s['agent']['avg_realized_regret'] for s in seeds]
    baseline_regrets = [s['baseline']['avg_realized_regret'] for s in seeds]
    seed_labels = [f"Seed {s['outer_seed']}" for s in seeds]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(seed_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_regrets, width, label='Baseline', color='#d9534f',
                   edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, agent_regrets, width, label='Agent', color='#5cb85c',
                   edgecolor='black', alpha=0.8)

    ax.axhline(y=np.mean(baseline_regrets), color='#d9534f', linestyle='--', linewidth=1.5,
               label=f'Baseline mean ({np.mean(baseline_regrets):.3f})')
    ax.axhline(y=np.mean(agent_regrets), color='#5cb85c', linestyle='--', linewidth=1.5,
               label=f'Agent mean ({np.mean(agent_regrets):.3f})')

    ax.set_ylabel('Avg Realized Regret', fontsize=12)
    ax.set_title('Multi-Seed Robustness: Agent vs Baseline', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seed_labels, fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, '9_multi_seed_robustness.png'), dpi=150)
    plt.close()
    print("  Saved 9_multi_seed_robustness.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading experiment results from JSON...")
    exp_data = load_results()

    print("Collecting detailed episode data for per-step plots (50 episodes)...")
    agent_detailed = collect_detailed_data(num_episodes=50)

    print("\nGenerating figures...")
    plot_regret_comparison(exp_data)
    plot_regret_distribution(exp_data)
    plot_epistemic_uncertainty_decay(agent_detailed)
    plot_action_distribution(agent_detailed)
    plot_exceedance_comparison(exp_data)

    print("\nRunning threshold sensitivity analysis (this takes a moment)...")
    plot_threshold_sensitivity()

    print("\nGenerating new experiment figures...")
    plot_persona_comparison(exp_data)
    plot_ablation_results(exp_data)
    plot_multi_seed_robustness(exp_data)

    print(f"\nAll 9 figures saved to {FIG_DIR}/")

    # Print summary from JSON data
    agent_agg = exp_data['agent_vs_baseline']['agent']['aggregate']
    baseline_agg = exp_data['agent_vs_baseline']['baseline']['aggregate']
    print("\n" + "="*50)
    print("SUMMARY (from full experiment results)")
    print("="*50)
    print(f"Agent  — purchase rate: {agent_agg['purchase_rate']*100:.0f}%, "
          f"avg regret: {agent_agg['avg_realized_regret']:.4f}, "
          f"avg queries: {agent_agg['avg_queries']:.1f}")
    print(f"Baseline — purchase rate: {baseline_agg['purchase_rate']*100:.0f}%, "
          f"avg regret: {baseline_agg['avg_realized_regret']:.4f}")


if __name__ == '__main__':
    main()
