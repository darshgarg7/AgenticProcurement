# Introduction and Motivation

In an emerging agent-mediated economy, autonomous buyers may eventually
negotiate and execute high-value transactions, for example purchasing a
\$1,000 laptop or enterprise subscription, on behalf of human users.
While large language models (LLMs) and web-scale recommenders can rank
items and generate fluent rationales, they are fundamentally black-box
function approximators and do not naturally expose calibrated
uncertainty over future user satisfaction or market
outcomes.[@charpentier2022disentangling] As a result, they are not by
themselves sufficient for deciding when it is appropriate to commit to a
purchase, particularly in volatile, partially observable markets.

We address the problem of *autonomous high-stakes procurement* under
uncertainty. Our central design principle is that a buyer agent must (i)
be model-based, with explicit probabilistic representations of user
preferences and environment dynamics; (ii) quantify its own epistemic
uncertainty over the consequences of a purchase; and (iii) optimize a
*minimax regret* objective rather than a single point estimate of
expected value.[@rigter2020minimax; @regretDecision1982] When the
worst-case regret is small and epistemic uncertainty is low, the agent
may autonomously execute a transaction. When regret or epistemic
uncertainty is large, the agent should instead defer to the user, either
by asking a clarifying question or by abstaining from purchase if no
safe decision emerges by the episode horizon. This formulation directly
ties to course themes in Bayesian inference, decision making under
uncertainty, and stochastic control.[@charpentier2022disentangling]

Our proposal is also motivated by the observation that autonomous
procurement is not simply a ranking problem. The core challenge is
deciding *when an agent should act at all* under incomplete knowledge of
both user preferences and market dynamics. In this sense, the project is
fundamentally about uncertainty-aware delegation: the agent must trade
off immediate reward, information gathering, and the cost of a
potentially regrettable purchase.

# Problem Statement

We formalize autonomous procurement as a stochastic control problem with
partial observability. Time is discrete, $t = 0,1,\dots,T$. At each step
$t$, the agent observes an information state $o_t$ derived from an
underlying environment state $s_t \in \mathcal{S}$, selects an action
$a_t$ from a finite action set $\mathcal{A}$, and transitions to
$s_{t+1}$ according to unknown stochastic dynamics. The episode
terminates when the agent executes a [Purchase]{.smallcaps} action or
the horizon $T$ is reached. In the latter case, abstention is modeled
implicitly as ending the episode without purchase when no action
satisfies the safety criterion.

## Action Space and Environment

The action set is
$$\mathcal{A} = \{\textsc{Search},\, \textsc{Purchase},\, \textsc{QueryUser},\, \textsc{Wait}\}.$$

The environment includes a catalog of items $i \in \mathcal{I}$, each
with a feature vector $x_i \in \mathbb{R}^d$ capturing price, rating,
category, and other attributes. Item availability and effective price
evolve stochastically due to stock-out events, discounts, and competing
buyers.

We assume a parametric environment model
$$P(s_{t+1} \mid s_t, a_t, \theta_E),$$ where $\theta_E$ parameterizes
price and availability dynamics. The true environment parameters are
unknown; the agent operates with an approximate model that captures key
stochastic structure, such as the probability of stock-out as a function
of demand and rating.

The [Wait]{.smallcaps} action allows the agent to defer execution when
the expected temporal value of waiting is positive, meaning the agent
anticipates that the price or availability model may transition into a
more favorable state, such as after a predicted discount or restock,
before committing to purchase. Concretely, the agent must decide when to
purchase, when to ask a clarifying question, when to continue gathering
information, and when to defer action because uncertainty remains too
high.

## User Utility and Minimax Regret

User preferences over item features are represented by a latent utility
function $$U_{\text{user}}(x; \theta_U),$$ where $\theta_U$ is a vector
of unknown preference parameters, for example trade-offs between price,
quality, and brand. The agent does not observe $\theta_U$ directly and
must infer it from sparse signals, such as explicit ratings, binary
feedback, or answers to targeted queries.

For a given environment state and unknown $\theta_U$, an optimal
clairvoyant decision-maker over the currently available set could choose
the item
$$i_t^\star(\theta_U) = \arg\max_{i \in \mathcal{I}_t} U_{\text{user}}(x_i; \theta_U).$$
If the agent instead selects item $i$, the *regret* under preference
parameters $\theta_U$ is
$$R(i; \theta_U) = U_{\text{user}}(x_{i_t^\star(\theta_U)}; \theta_U) - U_{\text{user}}(x_i; \theta_U).$$

Because $\theta_U$ is unknown, the agent maintains a belief distribution
$b_t(\theta_U)$ and evaluates *minimax* or worst-case regret over a
credible set of parameters $\Theta_t$ induced by this belief, for
example a high-probability ellipsoid around the posterior
mean.[@rigter2020minimax] The decision task is to choose a policy that,
when it does purchase, keeps worst-case regret below an acceptable
threshold.[@regretDecision1982]

# Background and Related Work

## Web Agents and Delegated Shopping

Recent work on language-grounded web agents has introduced environments
where agents navigate product listings and web pages to fulfill natural
language shopping instructions.[@yao2022webshop] These systems often
leverage LLMs to generate actions and rationales over semi-structured
web content.[@yao2022webshop] While they demonstrate strong performance
on benchmark tasks, they typically optimize success rates or rewards
without explicit uncertainty quantification or regret guarantees. As a
result, they do not provide principled mechanisms for deciding when to
defer to a human in high-stakes scenarios.

## Uncertainty-Aware Recommendation and Epistemic Uncertainty

Uncertainty-aware recommendation systems model user-item interactions
probabilistically, estimating predictive distributions rather than point
predictions. Bayesian and ensemble-based approaches distinguish between
aleatoric (inherent noise) and epistemic (model) uncertainty, using the
latter to improve robustness and to avoid overconfident recommendations
on sparse or out-of-distribution data.[@bdecf2025] In this work, we
adopt a Bayesian formulation of user preferences and treat epistemic
variance as a first-class quantity that drives both minimax regret and a
purchase safety gate.[@charpentier2022disentangling]

## Minimax Regret and Robust Decision Making

Minimax regret is a classical criterion in robust decision theory:
instead of maximizing expected utility under a single believed model, a
decision-maker chooses an action that minimizes the maximum regret over
a set of plausible models.[@regretDecision1982] This yields policies
that are conservative with respect to model misspecification and
uncertainty about underlying parameters. In stochastic environments,
minimax regret can be combined with Bayesian belief updates to balance
robustness and data efficiency.[@rigter2020minimax] Our agent uses
minimax regret over a belief-defined confidence set to decide when
autonomous purchasing is acceptable.

## Active Inference and Information-Seeking Action

Active inference provides a complementary perspective in which an agent
maintains a generative model over hidden states, updates beliefs as
evidence arrives, and selects actions that jointly pursue preferred
outcomes and reduce uncertainty.[@bogacz2017tutorial; @smith2022active]
This framework is particularly relevant to delegated procurement because
actions such as [QueryUser]{.smallcaps} and [Search]{.smallcaps} are not
merely fallback behaviors; they are *epistemic actions* that improve the
agent's knowledge of user preferences or market conditions before
committing to a purchase. This makes active inference a useful
conceptual lens for interpreting uncertainty-aware delegation.

## Positioning

Our contribution lies at the intersection of delegated web agents,
uncertainty-aware recommendation, robust decision making, and active
inference.[@yao2022webshop; @bdecf2025; @rigter2020minimax; @bogacz2017tutorial]
Unlike standard LLM-based shopping agents, our model-based delegation
engine maintains explicit probabilistic beliefs over user preferences
and environment dynamics. Unlike typical recommenders that output ranked
lists, our agent must decide *whether* to act autonomously at all, using
minimax regret and epistemic variance as safety metrics. This
combination yields an uncertainty-aware decision layer tailored to
delegated purchasing under stochasticity.

# Proposed Methodology

We now describe our modeling choices and decision rule in more detail.
Throughout, stochasticity and probabilistic reasoning are central.

While classical active inference is often formulated through variational
free energy minimization, our implementation is better understood as
*active inference-inspired*: the agent uses Bayesian belief updates and
information-seeking actions, especially [QueryUser]{.smallcaps}, to
reduce epistemic uncertainty before purchase.

## User Preference Model and Belief State

We represent each item by a feature vector $x \in \mathbb{R}^d$, for
example normalized price, quality score, and brand indicator. The user's
latent utility is modeled as
$$U_{\text{user}}(x; \theta_U) = \theta_U^\top x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_\epsilon^2),$$
with a Gaussian prior over preferences,
$$\theta_U \sim \mathcal{N}(m_0, S_0).$$

As the agent observes user signals, such as "I care more about
durability than style" or ratings on probed items, it updates a
posterior $p(\theta_U \mid \mathcal{D}_t)$, which remains Gaussian under
linear-Gaussian assumptions. The belief state is summarized by
$(m_t, S_t)$.

For a candidate item $x$, the predictive distribution over utility is
$$U_{\text{user}}(x) \mid \mathcal{D}_t \sim \mathcal{N}\big(m_t^\top x,\; x^\top S_t x + \sigma_\epsilon^2\big),$$
so the agent has both an expected utility and an uncertainty estimate
for each item.

We distinguish *epistemic uncertainty*, given by $x^\top S_t x$, from
total predictive uncertainty, given by
$x^\top S_t x + \sigma_\epsilon^2$. Since epistemic uncertainty is
reducible through additional evidence, it is the primary uncertainty
term used in the purchase safety gate.

## Environment Stochasticity and Digital Twin

To model market volatility, we define a simplified stochastic process
over availability and effective price. Let $z_{i,t} \in \{0,1\}$
indicate whether item $i$ is in stock at time $t$. We specify a
stock-out probability
$$\mathbb{P}(z_{i,t+1} = 0 \mid z_{i,t} = 1) = \alpha_i,$$ where
$\alpha_i$ is a function of item attributes such as demand, rating, and
category. Similarly, we may model stochastic discounts as a random
process on price.

In our implementation, this environment model serves as a stochastic
simulator: before committing to a final purchase decision, the agent
runs Monte Carlo rollouts to estimate distributions over future outcomes
under candidate policies. This explicitly incorporates environmental
stochasticity into the decision rule.

## Minimax Regret Objective

Given a belief $p(\theta_U \mid \mathcal{D}_t)$, we define a
high-probability confidence set
$$\Theta_t = \{\theta_U : (\theta_U - m_t)^\top S_t^{-1} (\theta_U - m_t) \leq c_t\},$$
for some radius $c_t$ determined by a chosen confidence level. For any
item $i$, the worst-case regret over $\Theta_t$ is
$$\bar{R}_t(i) = \sup_{\theta_U \in \Theta_t} \Big[ \max_{j \in \mathcal{I}_t} U_{\text{user}}(x_j; \theta_U) - U_{\text{user}}(x_i; \theta_U) \Big],$$
where $\mathcal{I}_t$ is the set of currently available items. The agent
seeks items with small $\bar{R}_t(i)$; in practice we approximate
$\bar{R}_t(i)$ using linear bounds or sampling over $\theta_U$ from the
confidence set.[@rigter2020minimax]

## Safety Gate and Delegation Decision

At a decision point, for each candidate item $i$ the agent computes
$$\begin{aligned}
\mu_t(i) &= m_t^\top x_i,\\
\sigma_{\text{epi},t}^2(i) &= x_i^\top S_t x_i,\\
\sigma_{\text{pred},t}^2(i) &= x_i^\top S_t x_i + \sigma_\epsilon^2,\\
\bar{R}_t(i) &\approx \text{approximate worst-case regret over } \Theta_t.
\end{aligned}$$

We introduce two thresholds:

-   $\epsilon_{\text{reg}}$: maximum tolerable worst-case regret.

-   $\epsilon_{\text{epi}}$: maximum tolerable epistemic uncertainty.

The *autonomous purchase condition* is
$$\bar{R}_t(i) \leq \epsilon_{\text{reg}} \quad \text{and} \quad \sigma_{\text{epi},t}^2(i) \leq \epsilon_{\text{epi}}.$$

If there exists an item satisfying this condition and with sufficiently
high mean utility, the agent may execute [Purchase]{.smallcaps}.
Otherwise, the agent gathers more information by choosing
[QueryUser]{.smallcaps}, [Search]{.smallcaps}, or [Wait]{.smallcaps}. We
still track predictive uncertainty $\sigma_{\text{pred},t}^2(i)$ for
analysis, but the safety gate is driven primarily by epistemic
uncertainty because it is reducible through additional evidence.

## Connection to Active Inference

Our formulation is also compatible with an active inference perspective,
in which the agent maintains a generative model over hidden user
preferences and market dynamics, updates beliefs as new evidence
arrives, and selects actions that reduce uncertainty while moving toward
preferred outcomes.[@bogacz2017tutorial; @smith2022active] In the MVP,
the maintained belief state is primarily over latent user preference
parameters $\theta_U$; market dynamics are stochastic but treated as a
fixed simulator model rather than jointly inferred hidden variables. In
this framing, [QueryUser]{.smallcaps} and [Search]{.smallcaps} are not
merely fallback actions; they are epistemic actions that reduce
uncertainty in the agent's belief state. Operationally,
[QueryUser]{.smallcaps} acts as a targeted preference-elicitation step
over the latent utility model. This aligns with active inference
accounts in which action selection trades off goal-directed behavior
with information gain under a probabilistic generative
model.[@smith2022active]

In our setting, this connection appears operationally through the
expected information gain term
$$IG(a) = \mathcal{H}(b_t) - \mathbb{E}[\mathcal{H}(b_{t+1})],$$ which
measures how much an action is expected to reduce uncertainty over the
user-preference belief state. Thus, although our implementation is
presented in minimax-regret form, it can also be interpreted as an
active inference-inspired delegation agent that prefers autonomous
purchase only when both expected utility is high and epistemic
uncertainty is sufficiently low.

## Pseudocode for the Delegation Engine

Below we sketch the core decision logic.

**Algorithm 1: Model-Based Delegation Engine**

1.  Input: belief $(m_t, S_t)$, available items $\mathcal{I}_t$,
    thresholds $\epsilon_{\text{reg}}, \epsilon_{\text{epi}}$, utility
    threshold $\tau_{\text{util}}$.

2.  For each $i \in \mathcal{I}_t$:

    1.  Compute $\mu_t(i)$, $\sigma_{\text{epi},t}^2(i)$, and
        approximate worst-case regret $\bar{R}_t(i)$ over $\Theta_t$.

3.  Define the safe candidate set
    $$\mathcal{C}_t = \left\{ i \in \mathcal{I}_t \,:\, \mu_t(i) \geq \tau_{\text{util}},\ \bar{R}_t(i) \leq \epsilon_{\text{reg}},\ \sigma_{\text{epi},t}^2(i) \leq \epsilon_{\text{epi}} \right\}.$$

4.  If $\mathcal{C}_t \neq \emptyset$:

    -   Return [Purchase]{.smallcaps}$(i^\star)$ where
        $i^\star = \arg\max_{i \in \mathcal{C}_t} \mu_t(i)$.

5.  Else:

    -   Estimate $IG(\textsc{QueryUser})$ as the expected reduction in
        entropy of the posterior over $\theta_U$ after a user answer.

    -   Estimate the one-step value of [Wait]{.smallcaps} using Monte
        Carlo rollouts under stochastic price and availability dynamics.

    -   If $IG(\textsc{QueryUser})$ is largest, return
        [QueryUser]{.smallcaps}.

    -   Else if the expected value of [Wait]{.smallcaps} exceeds the
        expected value of continued search, return [Wait]{.smallcaps}.

    -   Else return [Search]{.smallcaps}.

This algorithm is inherently stochastic: it relies on probabilistic
beliefs over $\theta_U$, predictive distributions over utilities,
randomized market transitions, Monte Carlo rollouts in the simulator,
and a confidence set $\Theta_t$ that evolves under Bayesian updates.

# Data and Environment

To keep the project feasible while still demonstrating the full
stochastic decision pipeline, we adopt a two-level environment plan.

#### Static CSV Simulator (Core MVP).

The primary environment is a static catalog stored as a CSV with
approximately 1,000 products. Each row contains curated attributes:
price, rating, category, and derived quality metrics. We simulate
availability and price transitions using the stochastic stock-out model
and, optionally, random discount processes. The agent interacts with
this environment entirely in simulation, enabling rapid experimentation.
The catalog will be derived from a curated public dataset or a
controlled synthetic dataset, depending on data availability and
preprocessing time.

#### Stochastic Simulator Interface.

In the static simulator, the planning model and the environment
coincide. If time permits, we will connect the same delegation engine to
a more realistic web environment, where the simulator is used for Monte
Carlo planning prior to issuing real actions.

Synthetic user personas, for example a "budget shopper" or "quality
maximizer," are defined by different priors $(m_0, S_0)$ over
$\theta_U$, providing hidden preference structure while avoiding privacy
issues.

#### Feasibility and Scope Control.

To ensure the project remains tractable within the course timeline, we
intentionally begin with a controlled simulator rather than full web
deployment. This allows us to isolate the core scientific question,
namely whether explicit uncertainty quantification and minimax-regret
planning improve delegated purchasing decisions under stochasticity,
before introducing additional engineering complexity. If integration
with a live web environment proves too time-consuming, the static
simulator will still fully support the proposed stochastic modeling,
Bayesian updates, Monte Carlo evaluation, and baseline comparisons.

# Implementation Plan and Timeline

We assume a three-person team, with work being organized around modular
components.

## Roles

-   **Member 1**: Implement the Bayesian user preference model,
    posterior updates, and confidence set construction.

-   **Member 2**: Implement the stochastic stock-out and price dynamics,
    simulator logic, and feature processing from the CSV catalog.

-   **Member 3**: Implement the minimax-regret approximation, delegation
    engine, experiment pipeline, and evaluation scripts.

## Timeline

#### Weeks 1--2: Foundations.

Finalize the problem specification and metrics. Implement a prototype
Bayesian user model and generate synthetic personas. Load a toy CSV
dataset and basic feature extraction.

#### Week 3: Environment Stochasticity.

Implement the stock-out and price dynamics, environment step function,
and item sampling procedures. Verify that the simulator produces
realistic variability across episodes.

#### Week 4: Delegation Engine.

Implement the minimax-regret calculation using sampling or linear bounds
and the epistemic-uncertainty safety gate. Integrate thresholds and
decision logic into a clean API.

#### Week 5: Integration and Debugging.

Connect the user model, environment, and planner in a single loop. Run
smoke tests demonstrating Buy versus Query decisions under different
priors and thresholds.

#### Week 6: Evaluation and Analysis.

Run experiments across personas and stochastic seeds. Collect metrics,
analyze trade-offs between regret, number of queries, no-purchase rate,
and purchase rate, and produce plots and tables for the final report.

# Evaluation and Success Criteria

All evaluation will be conducted across multiple stochastic seeds,
sampled user preference realizations, and randomized market transitions
so that performance reflects robustness under uncertainty rather than
success on a single deterministic scenario.

-   **Average Regret**: Expected regret of the purchased item relative
    to the clairvoyant optimum, averaged over episodes.

-   **Worst-Case Regret**: Empirical approximation of worst-case regret
    over sampled $\theta_U$ from the confidence set, highlighting
    robustness.

-   **Delegation / No-Purchase Rate**: Frequency of
    [QueryUser]{.smallcaps} actions or episodes that end without
    purchase; we expect this to increase when priors are broad and
    thresholds are strict.

-   **Regret Exceedance Rate**: Frequency with which the realized regret
    of a purchased item exceeds the target tolerance under sampled true
    preferences, measuring whether the safety gate is well-calibrated
    under stochasticity.

-   **Persona Adaptation**: Differences in behavior across synthetic
    personas, demonstrating that changing priors alone changes the
    agent's strategy without retraining.

We will compare the model-based minimax-regret agent against a baseline
that selects items solely by posterior mean utility $\mu_t(i)$, without
explicit regret or uncertainty constraints. We hypothesize that the
baseline will either incur higher regret or require ad hoc heuristics to
avoid poor purchases, whereas our approach uses principled probabilistic
criteria.

We will also perform ablations that remove (i) posterior uncertainty
estimates, (ii) the minimax-regret term, and (iii) stochastic market
dynamics, in order to isolate which components are responsible for
improvements in safety and robustness.

We will consider the project successful if, relative to the
posterior-mean baseline, the proposed agent reduces empirical regret and
regret exceedance rate while maintaining a reasonable purchase rate
across stochastic seeds and user personas.

# Conclusion

We propose a model-based delegation engine for autonomous high-stakes
procurement in stochastic markets. By explicitly modeling latent user
preferences and stochastic market dynamics, our system treats
uncertainty, regret, and delegation as first-class decision variables
rather than after-the-fact heuristics. By quantifying epistemic
uncertainty and optimizing a minimax-regret objective, the agent
provides a principled uncertainty-aware decision layer for deciding when
to buy, when to ask, and when to
defer.[@charpentier2022disentangling; @rigter2020minimax] This directly
addresses the course requirement that the project incorporate
stochasticity into the core decision-making process.

::: thebibliography
9

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. WebShop:
Towards Scalable Real-World Web Interaction with Grounded Language
Agents. *Advances in Neural Information Processing Systems (NeurIPS)*,
2022.

Radin Cheraghi, Amir Mohammad Mahfoozi, Sepehr Zolfaghari,
Mohammadshayan Shabani, Maryam Ramezani, and Hamid R. Rabiee. Epistemic
Uncertainty-aware Recommendation Systems via Bayesian Deep Ensemble
Learning. *arXiv preprint arXiv:2504.10753*, 2025.

Bertrand Charpentier, Ransalu Senanayake, Mykel J. Kochenderfer, and
Stephan Günnemann. Disentangling Epistemic and Aleatoric Uncertainty in
Reinforcement Learning. *arXiv preprint arXiv:2206.01558*, 2022.

Marc Rigter, Bruno Lacerda, and Nick Hawes. Minimax Regret Optimisation
for Robust Planning in Uncertain Markov Decision Processes. In
*Proceedings of the Conference on Robot Learning (CoRL)*, 2020.

David E. Bell. Regret in Decision Making under Uncertainty. *Operations
Research*, 30(5):961--981, 1982.

Rafal Bogacz. A tutorial on the free-energy framework for modelling
perception and learning. *Journal of Mathematical Psychology*,
76:198--211, 2017.

Ryan Smith, Thomas Parr, and Karl J. Friston. A step-by-step tutorial on
active inference and its application to empirical data. *Journal of
Mathematical Psychology*, 107:102632, 2022.
:::
