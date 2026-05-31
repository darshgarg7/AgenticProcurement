"""
Streamlit Web Demo — Active Inference-Inspired Delegated Procurement

Tab 1: Live Shopping Agent — step through a procurement episode interactively
Tab 2: Research Dashboard — view experiment results and figures
"""

import json
import os
from dataclasses import replace

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config.settings import EngineConfig, EnvConfig, ModelConfig, PersonaConfig
from core.interfaces import Purchase, QueryUser, Wait
from decision.delegation_engine import DelegationEngine
from environment.simulator import StochasticMarket
from models.bayesian_user import BayesianPreferenceModel

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'products.csv')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results', 'full_experiment_results.json')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'results', 'figures')

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Delegated Procurement Agent",
    page_icon="🛒",
    layout="wide",
)

# ── Tab layout ──────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["🛒 Live Shopping Agent", "📊 Research Dashboard"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Live Shopping Agent
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Live Shopping Agent Simulation")
    st.markdown(
        "Watch the Bayesian procurement agent decide in real time: "
        "**Purchase**, **Query User**, **Wait**, or **Search**."
    )

    # ── Sidebar-like controls in columns ────────────────────────────────
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

    with col_ctrl1:
        persona_choice = st.selectbox(
            "User Persona",
            ["budget_shopper", "quality_maximizer", "balanced", "random"],
            index=3,
        )
    with col_ctrl2:
        eps_reg = st.slider("ε_reg (regret threshold)", 0.1, 2.0, 0.8, 0.1)
        eps_var = st.slider("ε_var (uncertainty threshold)", 0.1, 2.0, 0.8, 0.1)
    with col_ctrl3:
        max_steps = st.slider("Max steps", 5, 50, 30)
        seed = st.number_input("Random seed", 0, 9999, 42)

    run_btn = st.button("▶ Run Episode", type="primary", use_container_width=True)

    if run_btn:
        d = 8
        rng = np.random.RandomState(seed)

        # Build persona
        if persona_choice == "random":
            true_theta = rng.randn(d)
            true_theta /= np.linalg.norm(true_theta)
            prior_m0 = true_theta + rng.normal(0, 0.4, d)
            prior_S0 = np.eye(d) * 0.5
            persona_name = "random"
        else:
            factory = getattr(PersonaConfig, persona_choice)
            persona = factory(d=d, seed=seed)
            true_theta = persona.true_theta
            prior_m0 = persona.prior_mean
            prior_S0 = persona.prior_cov
            persona_name = persona.name

        # Initialize
        engine_config = EngineConfig(eps_reg=eps_reg, eps_var=eps_var, tau_util=0.0)
        env_config = EnvConfig(data_path=DATA_PATH)
        engine_config = replace(
            engine_config,
            wait_stockout_alpha=env_config.alpha,
            wait_price_fluctuation=env_config.price_fluctuation,
        )
        model_config = ModelConfig(sigma2=0.05)
        model = BayesianPreferenceModel(d=d, m0=prior_m0, S0=prior_S0, config=model_config)
        engine = DelegationEngine(model, config=engine_config, rng=rng)
        env = StochasticMarket(config=env_config, rng=rng)

        # Run episode
        step_log = []
        epi_unc_history = []
        action_counts = {"Purchase": 0, "QueryUser": 0, "Wait": 0, "Search": 0}
        final_outcome = None

        progress = st.progress(0, text="Running episode...")

        for step in range(max_steps):
            progress.progress((step + 1) / max_steps, text=f"Step {step + 1}/{max_steps}")
            obs = env.observe()

            if obs.features.shape[0] == 0:
                step_log.append({"Step": step + 1, "Action": "Search", "Reason": "No items available",
                                 "Epistemic Unc.": "-", "Best Exp. Utility": "-"})
                action_counts["Search"] += 1
                env.step()
                continue

            epi_unc = model.epistemic_uncertainty(obs.features)
            exp_util = model.expected_utility(obs.features)
            best_idx = int(np.argmax(exp_util))
            epi_unc_history.append(float(epi_unc[best_idx]))

            action = engine.decide(obs)

            if isinstance(action, Purchase):
                idx = obs.item_ids.index(action.item_id)
                utils = obs.features @ true_theta
                best_u = float(np.max(utils))
                chosen_u = float(utils[idx])
                realized_regret = best_u - chosen_u

                step_log.append({
                    "Step": step + 1, "Action": "✅ Purchase",
                    "Reason": f"Item {action.item_id} (regret={realized_regret:.4f})",
                    "Epistemic Unc.": f"{epi_unc[best_idx]:.4f}",
                    "Best Exp. Utility": f"{exp_util[best_idx]:.4f}",
                })
                action_counts["Purchase"] += 1
                final_outcome = {
                    "purchased": True, "item_id": action.item_id,
                    "realized_regret": realized_regret, "steps": step + 1,
                }
                break

            elif isinstance(action, QueryUser):
                idx = obs.item_ids.index(action.item_id)
                x = obs.features[idx]
                y = float(true_theta @ x) + rng.normal(0, np.sqrt(model_config.sigma2))
                model.update(x, y)
                step_log.append({
                    "Step": step + 1, "Action": "❓ QueryUser",
                    "Reason": f"Ask about item {action.item_id}",
                    "Epistemic Unc.": f"{epi_unc[best_idx]:.4f}",
                    "Best Exp. Utility": f"{exp_util[best_idx]:.4f}",
                })
                action_counts["QueryUser"] += 1

            elif isinstance(action, Wait):
                env.step()
                step_log.append({
                    "Step": step + 1, "Action": "⏳ Wait",
                    "Reason": "MC rollout: future may be better",
                    "Epistemic Unc.": f"{epi_unc[best_idx]:.4f}",
                    "Best Exp. Utility": f"{exp_util[best_idx]:.4f}",
                })
                action_counts["Wait"] += 1

            else:
                env.step()
                step_log.append({
                    "Step": step + 1, "Action": "🔍 Search",
                    "Reason": "Explore more options",
                    "Epistemic Unc.": f"{epi_unc[best_idx]:.4f}",
                    "Best Exp. Utility": f"{exp_util[best_idx]:.4f}",
                })
                action_counts["Search"] += 1

        progress.empty()

        if final_outcome is None:
            final_outcome = {"purchased": False, "steps": max_steps}

        # ── Results display ─────────────────────────────────────────────
        st.divider()

        if final_outcome["purchased"]:
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("Outcome", "✅ Purchased")
            col_r2.metric("Item", f"#{final_outcome['item_id']}")
            col_r3.metric("Realized Regret", f"{final_outcome['realized_regret']:.4f}")
            col_r4.metric("Steps Taken", final_outcome['steps'])
        else:
            st.warning(f"Agent did not purchase within {max_steps} steps (too cautious at current thresholds).")

        # Step log table
        st.subheader("Step-by-Step Log")
        st.dataframe(pd.DataFrame(step_log), use_container_width=True, hide_index=True)

        # Charts side by side
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.subheader("Epistemic Uncertainty Decay")
            if epi_unc_history:
                fig1, ax1 = plt.subplots(figsize=(6, 3.5))
                ax1.plot(range(1, len(epi_unc_history) + 1), epi_unc_history,
                         'o-', color='steelblue', linewidth=2, markersize=5)
                ax1.axhline(y=eps_var, color='red', linestyle='--', label=f'ε_var={eps_var}')
                ax1.set_xlabel('Step')
                ax1.set_ylabel('Epistemic Uncertainty')
                ax1.legend()
                ax1.set_title('Uncertainty Reduction')
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close()

        with col_c2:
            st.subheader("Action Distribution")
            used_actions = {k: v for k, v in action_counts.items() if v > 0}
            if used_actions:
                color_map = {'Purchase': '#5cb85c', 'QueryUser': '#5bc0de',
                             'Search': '#f0ad4e', 'Wait': '#d9534f'}
                fig2, ax2 = plt.subplots(figsize=(6, 3.5))
                colors = [color_map.get(k, '#999') for k in used_actions]
                ax2.bar(used_actions.keys(), used_actions.values(), color=colors, edgecolor='black')
                ax2.set_ylabel('Count')
                ax2.set_title('Actions Taken This Episode')
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Research Dashboard
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Research Dashboard")
    st.markdown(
        "Results from the full experiment suite: **Agent vs Baseline**, "
        "**Persona Comparison**, **Ablation Studies**, and **Multi-Seed Robustness**."
    )

    # ── Load results ────────────────────────────────────────────────────
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            exp_data = json.load(f)

        # ── Key metrics ────────────────────────────────────────────────
        st.subheader("Key Results Summary")
        agent_agg = exp_data['agent_vs_baseline']['agent']['aggregate']
        baseline_agg = exp_data['agent_vs_baseline']['baseline']['aggregate']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Agent Avg Regret", f"{agent_agg['avg_realized_regret']:.4f}",
                     delta=f"{agent_agg['avg_realized_regret'] - baseline_agg['avg_realized_regret']:.4f}",
                     delta_color="inverse")
        col2.metric("Baseline Avg Regret", f"{baseline_agg['avg_realized_regret']:.4f}")
        col3.metric("Agent Purchase Rate", f"{agent_agg['purchase_rate']*100:.0f}%")
        col4.metric("Agent Exceedance", f"{agent_agg['exceedance_rate']*100:.1f}%")

        st.divider()

        # ── Figures ─────────────────────────────────────────────────────
        st.subheader("Experiment Figures")

        figures = [
            ("1_regret_comparison.png", "Agent vs Baseline Regret"),
            ("2_regret_distribution.png", "Regret Distribution"),
            ("3_epistemic_uncertainty_decay.png", "Epistemic Uncertainty Decay"),
            ("4_action_distribution.png", "Action Distribution"),
            ("5_exceedance_and_purchase_rate.png", "Exceedance & Purchase Rate"),
            ("6_threshold_sensitivity.png", "Threshold Sensitivity"),
            ("7_persona_comparison.png", "Persona Comparison"),
            ("8_ablation_results.png", "Ablation Study"),
            ("9_multi_seed_robustness.png", "Multi-Seed Robustness"),
        ]

        # Display in a 3-column grid
        for row_start in range(0, len(figures), 3):
            cols = st.columns(3)
            for i, col in enumerate(cols):
                idx = row_start + i
                if idx < len(figures):
                    fname, caption = figures[idx]
                    fpath = os.path.join(FIG_DIR, fname)
                    if os.path.exists(fpath):
                        col.image(fpath, caption=caption, use_container_width=True)

        st.divider()

        # ── Persona details ─────────────────────────────────────────────
        st.subheader("Persona Experiment Details")
        if 'personas' in exp_data:
            persona_rows = []
            for name, data in exp_data['personas'].items():
                a = data['agent']['aggregate']
                b = data['baseline']['aggregate']
                persona_rows.append({
                    "Persona": name,
                    "Agent Regret": f"{a['avg_realized_regret']:.4f}",
                    "Baseline Regret": f"{b['avg_realized_regret']:.4f}",
                    "Agent Purchase %": f"{a['purchase_rate']*100:.0f}%",
                    "Agent Queries": f"{a['avg_queries']:.1f}",
                    "Agent Exceedance": f"{a['exceedance_rate']*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(persona_rows), use_container_width=True, hide_index=True)

        # ── Ablation details ────────────────────────────────────────────
        st.subheader("Ablation Study Details")
        if 'ablations' in exp_data:
            ablation_rows = []
            for name, data in exp_data['ablations'].items():
                a = data['aggregate']
                ablation_rows.append({
                    "Config": name,
                    "Avg Regret": f"{a['avg_realized_regret']:.4f}",
                    "Purchase Rate": f"{a['purchase_rate']*100:.0f}%",
                    "Avg Queries": f"{a['avg_queries']:.1f}",
                    "Exceedance": f"{a['exceedance_rate']*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(ablation_rows), use_container_width=True, hide_index=True)

        # ── Multi-seed details ──────────────────────────────────────────
        st.subheader("Multi-Seed Robustness")
        if 'multi_seed_robustness' in exp_data:
            seed_rows = []
            for s in exp_data['multi_seed_robustness']:
                seed_rows.append({
                    "Seed": s['outer_seed'],
                    "Agent Regret": f"{s['agent']['avg_realized_regret']:.4f}",
                    "Baseline Regret": f"{s['baseline']['avg_realized_regret']:.4f}",
                    "Agent Purchase %": f"{s['agent']['purchase_rate']*100:.0f}%",
                })
            st.dataframe(pd.DataFrame(seed_rows), use_container_width=True, hide_index=True)

            agent_regs = [s['agent']['avg_realized_regret'] for s in exp_data['multi_seed_robustness']]
            baseline_regs = [s['baseline']['avg_realized_regret'] for s in exp_data['multi_seed_robustness']]
            st.info(
                f"**Cross-seed summary:** Agent {np.mean(agent_regs):.4f} ± {np.std(agent_regs):.4f} "
                f"vs Baseline {np.mean(baseline_regs):.4f} ± {np.std(baseline_regs):.4f}"
            )

    else:
        st.error(
            "Experiment results not found. Run `python experiments/run_full_experiments.py` first."
        )

# ── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption("Active Inference-Inspired Delegated Procurement • CSCI 5512 • University of Minnesota")
