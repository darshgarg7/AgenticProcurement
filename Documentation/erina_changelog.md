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
