# Erina Karati — Changelog

---

## May 31, 2026

### Commits `1bfb1e2`, `d6b420f`, `24cebd4` — Web Demo UI Improvements

**`1bfb1e2` — Add cart nav dropdown: click cart icon to view purchased items**
- Added a floating dropdown anchored to the cart icon in the top nav bar that opens when the icon is clicked
- Dropdown displays the purchased item's emoji, full name, category, star rating, quality percentage, a green "✓ Purchased" badge, and price
- A footer row shows the running total
- Closes on the ✕ button, a click outside the wrapper, or automatically when a new episode run starts (`resetUI()`)
- Changes spanned all three frontend files:
  - `index.html`: wrapped cart icon in `.cart-wrapper`; added `.cart-nav-dropdown` with header, `.cnd-items` list, and `.cnd-footer` total
  - `style.css`: added `.cart-wrapper` (position: relative), `.cart-nav-dropdown` (absolute-positioned, 300 px wide, `cardAppear` slide-in animation), and all `.cnd-*` child styles
  - `app.js`: added DOM refs, toggle/close event handlers with click-outside detection, and updated `addToCart()` and `resetUI()` to populate and clear the dropdown

**`d6b420f` — Add Pause/Resume and Stop controls to episode animation**
- Added two new control buttons that appear in the hero bar only during an active run:
  - **⏸ Pause** (blue) — freezes the animation loop between steps; the button label toggles to **▶ Resume** (green) when paused
  - **■ Stop** (red) — immediately breaks out of the step loop and re-enables the Start button
- Rewrote the `wait(ms)` helper to be pause/stop-aware: returns instantly when `isStopped` is true; suspends execution in a held Promise when `isPaused` is true and only resolves when `_resumeResolve()` is called on resume
- Added `isPaused`, `isStopped`, and `_resumeResolve` state variables; all three are reset at the start of each new episode
- Added a `if (isStopped) break` guard at the top of the step loop so a stop mid-animation cleanly exits without processing further steps
- Button visibility is managed entirely by `runEpisode()`: shown at run start, hidden on natural completion or stop

**`24cebd4` — Fix mismatched product emojis: water bottle, jump rope, belt, blender, bike light**
- Audited all 60 entries in `web_demo/product_emojis.py` against their product noun labels
- Corrected 5 clearly wrong mappings:
  - `Water Bottle`: `🧴` (lotion pump bottle) → `🍶` (neutral bottle shape)
  - `Jump Rope`: `⏭️` (media next-track button) → `🪢` (rope/knot)
  - `Belt`: `👔` (necktie) → `🎗️` (ribbon/strap)
  - `Blender`: `🫙` (mason jar) → `🥤` (cup with straw, representing a blended drink)
  - `Bike Light`: `🚲` (whole bicycle) → `🔦` (flashlight/torch)

---

## May 11, 2026

### Commit `95a2753` — Fix Page Layout Gaps and Recompile PDF

- Audited the final report LaTeX source (`Documentation/final_report.tex`) for visual layout issues introduced by the prior round of edits
- Corrected spacing and alignment gaps between sections (notably around the figures and the results tables) that were rendering incorrectly in the compiled PDF
- Recompiled the full document and pushed the corrected PDF to the `erina` branch

---

## May 7, 2026

### Commits `b6c7857`, `765b1bb`, `d9ffe91` — Final Report Polish

**`b6c7857` — Final report touch up**
- Reviewed the full compiled PDF and identified remaining issues in phrasing, spacing, and figure captions
- Tightened the prose in the Method and Experiments sections to match the narrative structure of the experimental findings
- Ensured all figure cross-references (e.g., Fig. 7–9) were correctly linked to their captions

**`765b1bb` — Final report fix**
- Resolved a compilation error in the LaTeX source caused by a mismatched environment block
- Fixed an incorrectly formatted equation in the delegation decision rule section

**`d9ffe91` — Fix formatting in README.md citation section**
- Corrected Markdown formatting in the citation/references section of `README.md`, which had broken rendering due to stray backticks and inconsistent list syntax

---

## May 5, 2026

### Commits `b2ce139`, `bed05f5` → Merged via PR #5 (`e9b5533`) — Web Demo

**`b2ce139` — Add web demo and `app.py` entry point**

Built a full interactive web application (`web_demo/`) that visually demonstrates the Bayesian delegation agent shopping in real-time. The demo is an Amazon-style e-commerce frontend backed by a Flask API.

**Backend — `web_demo/server.py`**
- Flask app with 5 endpoints:
  - `GET /` — serves `index.html`
  - `GET /api/search?q=` — live product search over the CSV catalog with fuzzy name/category matching, returns up to 12 results
  - `GET /api/results` — serves `full_experiment_results.json` for the results dashboard
  - `GET /api/figures/<name>` — serves experiment figure PNGs from `results/figures/` with path-traversal protection via `os.path.basename`
  - `POST /api/run-episode` — core endpoint that instantiates a real `BayesianPreferenceModel`, `DelegationEngine`, and `StochasticMarket` and runs a full live episode, returning step-by-step telemetry including per-step action, reason, epistemic uncertainty, expected utility, and top-5 products with display metadata
- `get_product_info()` helper: maps normalized CSV features (`price_norm`, `rating_norm`, `quality_norm`) back to human-readable display values — prices in $9.99–$249.99 range, ratings 1.0–5.0, quality as integer percentage
- `items_from_obs()` helper: resolves `Observation.item_ids` back to full display product objects with category, emoji, name, and feature values
- Product name generation from 5 categories × 8 adjectives × 12 nouns = 480 distinct product names
- Input clamping on all `/api/run-episode` parameters (`eps_reg` ∈ [0.1, 5.0], `max_steps` ∈ [1, 100], `seed` ∈ [0, 99999]) to prevent abuse

**`web_demo/product_emojis.py`**
- 70-entry dictionary mapping all product noun types (Headphones, Dutch Oven, Yoga Mat, Leather Jacket, etc.) to their closest emoji — used to give each card a distinct icon instead of a generic category fallback

**Frontend — `web_demo/static/index.html`**
- Sticky top nav with brand logo, centered search bar with live dropdown, and cart icon with badge counter
- Hero section with full agent control panel: persona selector (Balanced / Budget Shopper / Quality Maximizer), ε regret threshold slider (0.1–2.0), speed selector (Slow / Normal / Fast / Instant), max steps input, and random seed input
- Status strip that updates live each step: current step number, action type (color-coded), human-readable reason string, and epistemic uncertainty reading
- Product grid that re-renders each step with animated product cards
- Shopping cart sidebar with item list, running total, and animated Checkout button
- Agent log sidebar with scrollable per-step log entries, color-coded by action type
- Purchase result overlay with full outcome details (item name, price, rating, quality, steps taken, realized regret, action breakdown)
- Post-simulation analysis section with: narrative paragraph, 5 metric boxes (steps, regret, queries, waits, searches), epistemic uncertainty line chart, action distribution donut chart, and a step-by-step timeline

**Frontend — `web_demo/static/app.js`**
- Step-by-step animation loop: fetches episode data from `/api/run-episode` once, then replays it at the user-selected speed using `setTimeout`-based `wait()` promises
- Per-action card rendering: Search refreshes the grid; QueryUser highlights the queried card in blue while dimming others; Wait dims all cards at 60% opacity; Purchase highlights the chosen card in green, dims others, then triggers fly-to-cart animation
- Fly-to-cart animation: clones the purchased card DOM element, positions it absolutely over the original card's bounding rect, then CSS-transitions it to the cart icon with scale-down and fade-out
- Cart shake animation on item arrival via CSS keyframes
- `renderStars(rating)` utility: converts a float rating to filled/half/empty star characters
- Canvas-drawn epistemic uncertainty line chart with hi-DPI scaling, area fill gradient, axis labels, colored tick marks per action type along the x-axis
- Canvas-drawn action donut chart with per-slice color coding and center total label
- Auto-generated narrative paragraph explaining what happened during the episode, including uncertainty reduction percentage, threshold calibration assessment, and natural-language description of each action type
- Live product search with 250ms debounce, dropdown rendering with category icons and price display, click-to-dismiss on Escape or outside click

**Frontend — `web_demo/static/style.css`**
- Amazon-inspired design system: `#131921` nav bar, `#ff9900` accent throughout, `#232f3e` hero/analysis sections, `#eaeded` page background
- Fully responsive layout: two-column shop area (280px sidebar + flex grid) collapses to single column on mobile; controls row wraps vertically; metric boxes collapse to 2-per-row; charts stack vertically
- Product card animations: `cardAppear` (translateY + scale on entry), `purchasePulse` (scale + green glow on selection), `shimmer` loading state
- 5 category color themes for card backgrounds and search result icons (Electronics/blue, Home/orange, Sports/green, Fashion/purple, Books/yellow)
- Overlay backdrop with `backdrop-filter: blur(4px)` for purchase result modal

**`app.py`**
- Entry point script that launches the Flask server from the project root with the correct import paths

**`bed05f5` — Add web demo setup instructions to README**
- Added a "Web Demo" section to `README.md` with installation steps, how to start the server, and a description of the demo's features

---

## May 4, 2026

### Commits `5347beb`, `168356b`, `d85ae58` → Merged via PR #4 (`4049d8e`) — Final Report

**`5347beb` — Add final report LaTeX, PDF, BibTeX references, and web demo screenshots**
- Authored the full LaTeX report (`Documentation/final_report.tex`) documenting the project end-to-end: abstract, introduction, related work, method, experiments, and conclusion
- Wrote `Documentation/references.bib` with all BibTeX citations
- Included web demo screenshots as figures in the report

**`168356b` — Add Figures 5 & 6 to report; all 12 figures now referenced**
- Added Fig. 5 (exceedance & purchase rate comparison) and Fig. 6 (threshold sensitivity sweep) as embedded figures with captions in the LaTeX source
- Ensured all 12 experiment figures (Figs. 1–9 from `results/figures/`) were correctly cross-referenced in the text

**`d85ae58` — Fix title page**
- Added all team member names, Group 14 designation, and instructor name to the title page
- Removed auto-generated table of contents and manual page breaks that were causing layout issues in the compiled output

---

## April 20, 2026

### Commits `c5f8dc5`, `512918d`, `6a4d41f` → Merged via PRs #2 & #3 (`163c0ef`, `b8e5337`)

---

### Codebase Audit & Gap Analysis

Performed a systematic review of the existing codebase against the project proposal (Sections 2–7) to identify what was working and what was missing before doing any new development.

**What was already implemented:**
- `BayesianPreferenceModel` — Bayesian linear regression over user preference vector θ ∈ ℝᵈ with Gaussian prior and closed-form posterior updates
- `StochasticMarket` — market environment that reads from `data/products.csv`, returns L2-normalized feature vectors, and applies stochastic stock-out (α=0.02) and price fluctuation (σ=0.05) each step
- `DelegationEngine` — decision engine with a three-condition safety gate (utility threshold τ, worst-case regret bound ε_reg, epistemic uncertainty bound ε_var) and a basic QueryUser/Wait/Search fallback
- `MetricsTracker` — counters for purchases, queries, delays, realized regret, and exceedance flags
- `run_experiment.py` — basic single-episode runner and greedy baseline comparison

**Gaps identified:**
- No unit or integration tests — correctness of core algorithms was entirely unverified
- Wait action used a random coin flip rather than principled Monte Carlo rollout estimation as specified in Proposal Algorithm 1
- No persona experiments — the system only ran with a single generic random θ
- No ablation studies — no way to validate that individual components (regret gate, uncertainty gate) were actually contributing
- No multi-seed robustness evaluation
- Results were not persisted — every run discarded its output
- No figures or visualizations

---

### `c5f8dc5` — Test Suite

Built a comprehensive test suite covering all core components from scratch. All tests are explicitly mapped to the proposal's evaluation criteria (Section 7).

**`tests/test_bayesian_user.py`**
- `test_posterior_convergence`: verifies that `model.m` converges toward `true_theta` after 50 noisy query observations via `model.update(x, y)`, checking that the L2 error after updates is less than half the prior error
- `test_epistemic_uncertainty_decreases`: confirms that `model.epistemic_uncertainty(X)` strictly decreases after each `model.update()` call — validates the information-gain property of Bayesian updates
- `test_expected_utility_shape`: checks that `model.expected_utility(X)` returns shape `(M,)` for an `(M, d)` feature matrix
- `test_sample_theta_distribution`: draws 5000 samples from `model.sample_theta(5000)` and checks that the sample mean is within 2σ of `model.m` and the sample covariance is within 10% Frobenius norm of `model.S`

**`tests/test_simulator.py`**
- `test_stock_out_rate`: runs the market for 200 steps and verifies the empirical stock-out rate is close to α=0.02 (within 2×)
- `test_price_fluctuation_bounds`: verifies that after 100 steps of `env.step()`, `price_norm` values remain in [0.01, ∞) (the enforced lower bound)
- `test_feature_normalization`: checks that `observe().features` rows all have L2 norm ≈ 1.0
- `test_empty_market_edge_case`: depletes the entire catalog and confirms `observe()` returns an `Observation` with `features.shape[0] == 0` without raising

**`tests/test_delegation_engine.py`**
- `test_purchase_when_gate_satisfied`: constructs a tight posterior (S = 0.001·I) centered on a known θ with permissive thresholds; asserts that `engine.decide()` returns a `Purchase` action — validates Proposal Eq. 16
- `test_purchase_selects_highest_utility`: confirms the purchased `item_id` corresponds to the item with highest `expected_utility` — not just any item
- `test_defer_when_uncertainty_high`: uses a very broad prior (S = 100·I) with a tight ε_var threshold; asserts the engine does not return `Purchase`
- `test_defer_when_utility_below_tau`: sets τ_util = 100.0 (impossible to satisfy); asserts deferral
- `test_regret_shape`: checks that `compute_worst_case_regret(X)` returns shape `(M,)` for M items
- `test_regret_non_negative`: with N=200 samples, verifies all regret values ≥ 0 (up to floating-point tolerance)
- `test_query_user_when_ig_high`: constructs a scenario with high epistemic uncertainty on the best item and forces `ig_query > ig_search`; asserts `QueryUser` is returned

**`tests/test_metrics.py`**
- `test_counter_increments`: verifies `record_purchase`, `record_query`, `record_delay` all increment their respective counters correctly
- `test_exceedance_flag`: `record_purchase(realized_regret=0.5, threshold=0.3)` → `exceeded_regret = True`; `record_purchase(realized_regret=0.2, threshold=0.3)` → `exceeded_regret = False`
- `test_default_state`: fresh `MetricsTracker` has all counters at zero and `purchased = False`

**`tests/test_integration.py`**
- `test_episode_terminates_within_horizon`: runs a full episode with `max_steps=50` and asserts it completes (either purchase or horizon exhaustion) without error
- `test_baseline_episode_valid`: runs `run_baseline_episode()` and checks all expected keys are present in the output dict
- `test_agent_outperforms_baseline_avg_regret`: runs 20 episodes for both agent and baseline under the same seeds; asserts agent mean regret < baseline mean regret

---

### `512918d` — Initial Visualization Pipeline

Created `experiments/visualize_results.py` with 6 figures mapped to Proposal §7:

- **Fig 1** (`1_regret_comparison.png`): Side-by-side bar chart comparing agent vs. greedy baseline average realized regret, with standard error bars. Agent: 0.094, Baseline: 0.184
- **Fig 2** (`2_regret_distribution.png`): Histogram of per-episode realized regret across all agent runs, with a vertical dashed line at ε_reg = 0.8 showing the safety threshold. Shows the distribution is heavily left-skewed toward near-zero regret
- **Fig 3** (`3_epistemic_uncertainty_decay.png`): Per-step epistemic uncertainty x⊤Sx plotted for each episode as a faint line, with the mean trajectory overlaid in bold. Visually confirms that Bayesian updates are reducing uncertainty over the episode
- **Fig 4** (`4_action_distribution.png`): Pie chart of the aggregate action distribution — QueryUser 45%, Wait 24%, Search 22%, Purchase 9% — showing the agent spends most of its budget gathering information before committing
- **Fig 5** (`5_exceedance_and_purchase_rate.png`): Side-by-side bars for exceedance rate and purchase rate, agent vs. baseline. Agent: 0% exceedance, 74% purchase rate. Baseline: 1% exceedance, 100% purchase rate
- **Fig 6** (`6_threshold_sensitivity.png`): Dual-axis line plot sweeping ε_epi from 0.1 to 2.0. Shows how tightening the epistemic threshold reduces purchase rate while also lowering average regret

---

### `6a4d41f` — Phase 1: Core Enhancements

#### Monte Carlo Wait Rollouts — `decision/delegation_engine.py`

Replaced the original coin-flip Wait/Search fallback with the principled Monte Carlo rollout algorithm specified in Proposal Algorithm 1.

**`estimate_wait_value(obs)` method:**
- Takes the current `Observation` and returns a scalar estimate of the expected best utility available after one wait step
- Runs `config.mc_rollouts = 20` independent simulations of the market transition:
  - Each item survives with probability 1 − α ≈ 0.98 (matching the simulator's stock-out rate)
  - Surviving items receive multiplicative price noise drawn from N(1.0, 0.05)
  - Re-normalizes feature vectors row-wise after perturbation to maintain the unit-norm invariant
  - Computes `model.expected_utility()` on the simulated next-state features and records the best value
- Returns `discount_factor × mean(future_values)` where `discount_factor = 0.95`
- Handles the empty-market edge case by returning `current_best × 0.5` as a penalty

**Updated `decide()` logic:**
- Now computes three competing values: `ig_query` (from Proposal Eq. 17: `0.5 × log(1 + var*/σ²)`), `ig_search` (fixed constant `base_search_ig = 0.1`), and `wait_advantage = wait_value − current_best_utility`
- Selects QueryUser when `ig_query > ig_search` and `ig_query > max(wait_advantage, 0)`
- Selects Wait when `wait_advantage > ig_search`
- Falls back to Search otherwise
- Decision is now fully deterministic given the current model state and market observation

**`EngineConfig` additions:**
- `mc_rollouts: int = 20`
- `discount_factor: float = 0.95`

#### Persona Experiments — `config/settings.py`

Added `PersonaConfig` dataclass to represent distinct synthetic user types, enabling controlled experiments with known ground-truth preferences.

**`PersonaConfig` fields:** `name`, `true_theta` (unit-norm preference vector), `prior_mean` (slightly perturbed from true_theta to simulate imperfect prior knowledge), `prior_cov` (isotropic Gaussian, σ² = 0.5)

**Three factory methods (all produce unit-norm `true_theta` with seed-controlled prior noise):**
- `budget_shopper(d, seed)`: `true_theta[0] = −0.8` (price_norm, strong price aversion), `true_theta[1] = 0.3` (rating), `true_theta[2] = 0.2` (quality) — normalized to unit norm. Models a user who primarily wants the cheapest option
- `quality_maximizer(d, seed)`: `true_theta[0] = 0.1` (price-indifferent), `true_theta[1] = 0.7` (rating), `true_theta[2] = 0.6` (quality) — models a user who ignores price and maximizes perceived quality
- `balanced(d, seed)`: `true_theta[0] = −0.4`, `true_theta[1] = 0.5`, `true_theta[2] = 0.5` — models a user who trades off all three dimensions

#### Full Experiment Suite — `experiments/run_full_experiments.py`

Built a self-contained experiment runner executing ~450 total episodes across 4 structured experiments. All results serialized to `results/full_experiment_results.json`.

**`run_episode()` core runner:**
- Accepts `true_theta`, `prior_m0/S0`, and optional override configs for engine, environment, and model
- Runs a full episode loop: observe → `engine.decide()` → execute action (with `model.update(x, y)` for QueryUser) → track metrics
- Records per-step telemetry: action type, epistemic uncertainty, expected utility at best item
- Returns a stats dict including `purchased`, `realized_regret`, `queries`, `delays`, `steps_taken`, and the full `step_data` list

**`run_baseline_episode()` greedy baseline:**
- Takes a single market snapshot, picks the item with highest expected utility under the prior, and purchases immediately — no exploration
- Returns the same stats format for direct comparison

**Experiment 1 — Agent vs. Baseline (100 episodes):**
- Seeds 0–99, random `true_theta` each seed (unit-norm), prior mean perturbed by N(0, 0.4)
- Agent: ε_reg=0.8, ε_var=0.8, τ=0.0, max_steps=30
- Results: Agent regret=0.071 ± 0.05, purchase=50%, exceedance=0% vs. Baseline regret=0.170 ± 0.08, exceedance=1%

**Experiment 2 — Persona Experiments (3 × 100 episodes):**
- Each persona run under the same seed range with persona-specific `true_theta` and prior
- `budget_shopper`: purchase=27%, avg regret=0.057
- `quality_maximizer`: purchase=22%, avg regret=0.065
- `balanced`: purchase=24%, avg regret=0.065
- Lower purchase rates reflect the personalized priors being harder to satisfy under the fixed safety gate thresholds

**Experiment 3 — Ablation Studies (4 × 100 episodes):**
- `full_agent`: baseline configuration (ε_reg=0.8, ε_var=0.8)
- `no_regret_gate`: ε_reg=999 (gate disabled) → purchase=98%, avg regret=0.167 — confirms the regret gate is the primary safety mechanism
- `no_uncertainty_gate`: ε_var=999 (gate disabled) → purchase rate increases, regret slightly higher
- `no_dynamics`: EnvConfig(alpha=0, price_fluctuation=0) → market is static; removes Wait's advantage, making Search/QueryUser dominant

**Experiment 4 — Multi-Seed Robustness (5 × 50 episodes):**
- 5 independent seed blocks; within each block runs 50 episodes and computes aggregate stats
- Agent: regret 0.072 ± 0.018 across seed blocks vs. Baseline: 0.180 ± 0.030
- Low variance across seeds confirms the agent's performance is stable and not artifact of lucky seeds

#### Phase 1 Tests — `tests/test_phase1.py`

11 new tests covering all Phase 1 additions:

- `test_budget_shopper_prefers_low_price`: asserts `true_theta[0] < 0` (negative price weight)
- `test_quality_maximizer_prefers_quality`: asserts `|true_theta[1]|, |true_theta[2]| > |true_theta[0]|`
- `test_balanced_persona`: basic shape checks
- `test_theta_is_unit_norm`: all three factories produce unit-norm `true_theta` (checked to 5 decimal places)
- `test_different_seeds_give_different_priors`: same true_theta, different prior means for different seeds
- `test_wait_value_is_finite`: `estimate_wait_value()` returns a finite float for a valid observation
- `test_wait_value_empty_obs`: returns `-inf` for empty market
- `test_no_more_coin_flip`: with impossible purchase thresholds, calling `engine.decide()` 20 times returns the same action each time (deterministic, not random)
- `test_ablation_no_regret_gate`: ε_reg=999 does not raise; produces higher purchase rate than default
- `test_full_experiment_smoke`: `run_full_experiments.main()` completes without error and writes JSON to disk
- Total test suite after Phase 1: **55 tests, all passing**

#### Updated Visualizations — `experiments/visualize_results.py`

Rewrote the visualization script to read from `results/full_experiment_results.json` rather than re-running episodes inline. Added 3 new figures:

- **Fig 7** (`7_persona_comparison.png`): Grouped bar chart with two metric groups (avg regret, purchase rate) × 3 persona bars each. Visually shows that `budget_shopper` achieves the lowest regret because its preference structure aligns well with the price-dominant catalog
- **Fig 8** (`8_ablation_results.png`): Triple bar chart (regret / purchase rate / query rate) per ablation condition. The `no_regret_gate` bar most dramatically demonstrates the safety gate's role — near-100% purchase rate but 2× higher regret
- **Fig 9** (`9_multi_seed_robustness.png`): Per-seed agent vs. baseline regret as paired bars, with horizontal mean lines overlaid. The consistent gap across all 5 seed blocks establishes statistical robustness

Total: **9 figures**, all directly mapped to Proposal §7 evaluation criteria.

---

## April 6, 2026

### Commits `329aa0f`, `b239ea7`, `1474c1d`, `1cac361`, `67a95d2` → Merged via PR #1 (`d661338`)

**`329aa0f` — Complete first working version**
- Assembled all components into a runnable end-to-end system for the first time
- Verified that a full episode (observe → decide → update → observe loop) completes without error

**`b239ea7` — Update `experiments/run_experiment.py`**
- Wired up the episode loop and greedy baseline comparison into a single runnable script
- Added basic console output of per-episode results

**`1474c1d` — Update `models/bayesian_user.py`**
- Implemented the closed-form Bayesian linear regression posterior update:
  - `update(x, y)`: `S⁻¹_new = S⁻¹ + xxᵀ/σ²`, `m_new = S_new(S⁻¹m + xy/σ²)`
- `expected_utility(X)` → `Xm` (vectorized inner product)
- `epistemic_uncertainty(X)` → `diag(XSXᵀ)` computed efficiently as `sum(X * (X @ S), axis=1)`
- `sample_theta(n)` → draws from `N(m, S)` using `np.random.multivariate_normal`

**`1cac361` — Update `decision/delegation_engine.py`**
- Implemented `compute_worst_case_regret(X)`: draws N=100 θ samples from the posterior, computes the full regret matrix `R[n,i] = max_j(θₙᵀxⱼ) − θₙᵀxᵢ`, and returns the 95th-percentile column-wise (worst-case over sampled θ values for each item)
- Implemented the three-condition safety gate: purchase iff `μ* ≥ τ_util AND R* ≤ ε_reg AND σ²* ≤ ε_var`
- Implemented `QueryUser` selection based on information gain: `IG = 0.5 × log(1 + σ²*/σ²_noise)`

**`67a95d2` — Update `environment/simulator.py`**
- Implemented `_extract_features(df)`: selects all non-id/non-utility columns and row-normalizes to unit L2 norm
- Implemented `observe()`: returns an `Observation` with item IDs and normalized feature matrix
- Implemented `step()`: applies stochastic stock-out (removes each item independently with probability α=0.02) and multiplicative price noise drawn from N(1.0, 0.05), with a floor of 0.01 on `price_norm`

---

## March 29, 2026

### Commits `8e1acfd`, `8628506`, `3adee39` — Project Initialization

**`8e1acfd` — Initial commit**
- Created the repository and established the project directory structure:
  - `core/interfaces.py` — abstract base classes `IPreferenceModel`, `IMarketEnvironment`, `IDecisionEngine` and dataclasses `Observation`, `Purchase`, `QueryUser`, `Wait`, `Search`
  - `config/settings.py` — `EngineConfig`, `EnvConfig`, `ModelConfig` dataclasses
  - `data/products.csv` — product catalog with `item_id`, `price_norm`, `rating_norm`, `quality_norm`, and one-hot `cat_0`–`cat_4` columns
  - Stub files for `models/`, `decision/`, `environment/`, `evaluation/`, `experiments/`

**`8628506` — Add README.tex**
- Drafted initial project README in LaTeX format describing the problem setting, proposed approach, and team members

**`3adee39` — Rename README.tex → README.md**
- Converted the README to Markdown for GitHub rendering compatibility
