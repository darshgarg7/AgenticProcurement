# Erina Karati — Changelog

## April 20, 2026

### Codebase Audit & Gap Analysis
- Reviewed full codebase against project proposal (Sections 2–7)
- Identified what is implemented: Bayesian user model, stochastic simulator, delegation engine, metrics tracker, experiment runner, greedy baseline comparison
- Identified gaps: no unit/integration tests, no persona experiments, no ablation studies, no multi-seed robustness evaluation, no saved results, incomplete comparison output, Wait action uses coin flip instead of Monte Carlo rollouts

### Test Suite — Initial Build
- Created `tests/` directory with unit and integration tests for all core components
- **test_bayesian_user.py**: posterior convergence, epistemic uncertainty reduction, expected utility correctness, sample_theta distribution validity
- **test_simulator.py**: stock-out rate consistency with alpha parameter, price fluctuation bounds, feature extraction and normalization, empty market edge case
- **test_delegation_engine.py**: purchase when safety gate satisfied, defer when thresholds violated, regret computation correctness, QueryUser information gain logic
- **test_metrics.py**: counter increments, exceedance flag accuracy, default state
- **test_integration.py**: episode terminates within horizon, baseline episode produces valid output, agent outperforms baseline over multiple seeds
- All tests mapped to proposal evaluation criteria (Section 7)

### Visualization Script — `experiments/visualize_results.py`
- Created visualization pipeline that generates 6 figures mapped to Proposal §7
- **Fig 1** (`1_regret_comparison.png`): Agent vs baseline average realized regret bar chart with std error bars
- **Fig 2** (`2_regret_distribution.png`): Histogram of regret distribution across episodes, with ε_reg threshold line
- **Fig 3** (`3_epistemic_uncertainty_decay.png`): Per-step epistemic uncertainty (x⊤Sx) decay showing Bayesian updates working — individual episodes + mean trajectory
- **Fig 4** (`4_action_distribution.png`): Pie chart of action distribution (QueryUser 45%, Search 22%, Wait 24%, Purchase 9%)
- **Fig 5** (`5_exceedance_and_purchase_rate.png`): Side-by-side exceedance rate and purchase rate comparison
- **Fig 6** (`6_threshold_sensitivity.png`): Dual-axis line plot showing purchase rate and avg regret vs ε_epi threshold (sensitivity analysis)
- Raw results saved as JSON (`results/agent_results.json`, `results/baseline_results.json`)
- Summary: Agent avg regret 0.094 vs Baseline 0.184, agent purchase rate 74%, 0% exceedance

### Phase 1 — Core Enhancements

#### Monte Carlo Wait Rollouts (`decision/delegation_engine.py`)
- Replaced coin-flip Wait/Search logic with Monte Carlo rollout estimation per Proposal Algorithm 1
- Added `estimate_wait_value()` method: simulates future market states with stock-out and price noise, estimates expected improvement over current best
- Wait action now selected only when `wait_advantage > 0` (MC value exceeds current best utility)
- Added `import copy` for deep-copying environment state during simulations
- Added `mc_rollouts=20` and `discount_factor=0.95` to `EngineConfig`

#### Persona Experiments (`config/settings.py`)
- Added `PersonaConfig` dataclass with `true_theta`, `prior_mean`, `prior_cov`, `name` fields
- Implemented 3 static factory methods:
  - `budget_shopper(d, seed)`: negative price weight, emphasizes low cost
  - `quality_maximizer(d, seed)`: high rating/quality weights
  - `balanced(d, seed)`: moderate weights across all features

#### Ablation Studies (`experiments/run_full_experiments.py`)
- Built full experiment suite (4 experiments, ~450 episodes total):
  1. **Agent vs Baseline** (100 episodes): Agent regret=0.071, purchase=50%, exceedance=0% vs Baseline regret=0.170, exceedance=1%
  2. **Persona experiments** (3×100 episodes): budget_shopper purchase=27%/regret=0.057, quality_maximizer purchase=22%/regret=0.065, balanced purchase=24%/regret=0.065
  3. **Ablation studies** (4×100 episodes): No Regret Gate → 98% purchase rate with 2× higher regret (0.167), validating its necessity
  4. **Multi-seed robustness** (5×50 episodes): Agent regret 0.072±0.018 vs Baseline 0.180±0.030
- Results saved to `results/full_experiment_results.json`

#### Phase 1 Tests (`tests/test_phase1.py`)
- 10 unit tests + 1 smoke test covering all Phase 1 additions
- Tests for each persona factory, MC wait rollouts, ablation configs, full experiment pipeline
- All 55 tests passing across the full test suite

### Updated Visualizations — `experiments/visualize_results.py`
- Rewrote visualization script to read from `results/full_experiment_results.json`
- Updated existing plots 1–6 to use tuned parameters (ε_reg=0.8, max_steps=30)
- Added 3 new experiment figures:
  - **Fig 7** (`7_persona_comparison.png`): Grouped bar chart — regret & purchase rate by persona (budget/quality/balanced)
  - **Fig 8** (`8_ablation_results.png`): Triple bar chart — regret, purchase rate, query rate per ablation condition
  - **Fig 9** (`9_multi_seed_robustness.png`): Per-seed agent vs baseline regret with mean lines
- Total: 9 figures, all proposal-aligned
