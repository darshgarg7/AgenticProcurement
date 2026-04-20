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
