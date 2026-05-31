"""Shared episode orchestration for experiments, demos, and tests.

This module owns the canonical agent-environment loop. Keeping episode
execution here prevents the batch experiments and web demo from drifting into
slightly different behaviors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from config.settings import EngineConfig, EnvConfig, ModelConfig
from core.interfaces import Observation, Purchase, QueryUser, Search, Wait
from decision.delegation_engine import DelegationEngine
from environment.simulator import StochasticMarket
from evaluation.metrics import MetricsTracker
from models.bayesian_user import BayesianPreferenceModel


@dataclass
class EpisodeContext:
    """Mutable runtime state for one procurement episode."""

    true_theta: npt.NDArray[np.float64]
    model_config: ModelConfig
    engine_config: EngineConfig
    rng: Any
    engine: DelegationEngine
    env: StochasticMarket
    tracker: MetricsTracker


@dataclass
class EpisodeStep:
    """Auditable record for one agent decision step."""

    step: int
    action: str
    item_id: Any | None
    target_index: int | None
    best_idx: int | None
    epi_unc: float | None
    exp_util: float | None
    estimated_wc_regret: float | None
    realized_regret: float | None
    available_item_ids: list[Any]
    top_item_ids: list[Any]
    wait_value: float | None = None
    wait_advantage: float | None = None
    query_information_gain: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


def make_episode_context(
    true_theta: npt.NDArray[np.float64],
    data_path: str,
    prior_m0: npt.NDArray[np.float64],
    prior_S0: npt.NDArray[np.float64],
    engine_config: EngineConfig | None = None,
    env_config: EnvConfig | None = None,
    model_config: ModelConfig | None = None,
    rng: Any | None = None,
) -> EpisodeContext:
    """Build the model, engine, market, and tracker for one episode.

    The engine wait-rollout settings are synchronized with the environment
    dynamics so offline experiments and live demos use the same stochastic
    assumptions.
    """
    theta = np.asarray(true_theta, dtype=np.float64)
    if theta.ndim != 1:
        raise ValueError("true_theta must be a 1D vector")

    env_cfg = env_config if env_config is not None else EnvConfig(data_path=data_path)
    engine_cfg = engine_config if engine_config is not None else EngineConfig(
        eps_reg=0.8, eps_var=0.8, tau_util=0.0
    )
    engine_cfg = replace(
        engine_cfg,
        wait_stockout_alpha=env_cfg.alpha,
        wait_price_fluctuation=env_cfg.price_fluctuation,
    )
    model_cfg = model_config if model_config is not None else ModelConfig(sigma2=0.05)
    random = np.random if rng is None else rng

    model = BayesianPreferenceModel(
        d=len(theta),
        m0=prior_m0,
        S0=prior_S0,
        config=model_cfg,
    )
    engine = DelegationEngine(model, config=engine_cfg, rng=random)
    env = StochasticMarket(config=env_cfg, rng=random)
    return EpisodeContext(
        true_theta=theta,
        model_config=model_cfg,
        engine_config=engine_cfg,
        rng=random,
        engine=engine,
        env=env,
        tracker=MetricsTracker(),
    )


def step_agent_episode(context: EpisodeContext, step_index: int, top_k: int = 5) -> EpisodeStep:
    """Advance a procurement episode by one agent action."""
    obs = context.env.observe()
    step_number = step_index + 1

    if obs.features.shape[0] == 0:
        context.tracker.record_delay()
        context.env.step()
        return EpisodeStep(
            step=step_number,
            action="Search",
            item_id=None,
            target_index=None,
            best_idx=None,
            epi_unc=None,
            exp_util=None,
            estimated_wc_regret=None,
            realized_regret=None,
            available_item_ids=[],
            top_item_ids=[],
        )

    action, diagnostics = context.engine.decide_with_diagnostics(obs)
    best_idx = diagnostics.best_idx
    if best_idx is None:
        raise RuntimeError("non-empty observation produced no best item")

    top_indices = np.argsort(-diagnostics.expected_utils)[:top_k]
    available_item_ids = list(obs.item_ids)
    top_item_ids = [obs.item_ids[i] for i in top_indices]

    common = {
        "step": step_number,
        "best_idx": best_idx,
        "epi_unc": float(diagnostics.epistemic_uncertainties[best_idx]),
        "exp_util": float(diagnostics.expected_utils[best_idx]),
        "estimated_wc_regret": float(diagnostics.worst_case_regrets[best_idx]),
        "realized_regret": None,
        "available_item_ids": available_item_ids,
        "top_item_ids": top_item_ids,
        "wait_value": diagnostics.wait_value,
        "wait_advantage": diagnostics.wait_advantage,
        "query_information_gain": diagnostics.ig_query,
    }

    if isinstance(action, Purchase):
        idx = obs.item_ids.index(action.item_id)
        realized_regret = compute_realized_regret(obs, context.true_theta, idx)
        estimated_wc_regret = float(diagnostics.worst_case_regrets[idx])
        context.tracker.record_purchase(
            realized_regret=realized_regret,
            estimated_wc_regret=estimated_wc_regret,
            threshold=context.engine_config.eps_reg,
        )
        purchase_common = {
            **common,
            "estimated_wc_regret": estimated_wc_regret,
            "realized_regret": realized_regret,
        }
        return EpisodeStep(
            **purchase_common,
            action="Purchase",
            item_id=action.item_id,
            target_index=idx,
        )

    if isinstance(action, QueryUser):
        idx = obs.item_ids.index(action.item_id)
        x = obs.features[idx]
        y = float(context.true_theta @ x) + context.rng.normal(
            0, np.sqrt(context.model_config.sigma2)
        )
        context.engine.model.update(x, y)
        context.tracker.record_query()
        return EpisodeStep(
            **common,
            action="QueryUser",
            item_id=action.item_id,
            target_index=idx,
        )

    if isinstance(action, Wait):
        context.tracker.record_delay()
        context.env.step()
        return EpisodeStep(
            **common,
            action="Wait",
            item_id=None,
            target_index=None,
        )

    if isinstance(action, Search):
        context.tracker.record_delay()
        context.env.step()
        return EpisodeStep(
            **common,
            action="Search",
            item_id=None,
            target_index=None,
        )

    raise TypeError(f"Unsupported action type: {type(action)!r}")


def run_agent_episode(
    true_theta: npt.NDArray[np.float64],
    data_path: str,
    prior_m0: npt.NDArray[np.float64],
    prior_S0: npt.NDArray[np.float64],
    engine_config: EngineConfig | None = None,
    env_config: EnvConfig | None = None,
    model_config: ModelConfig | None = None,
    max_steps: int = 20,
    rng: Any | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run the uncertainty-aware agent until purchase or horizon."""
    context = make_episode_context(
        true_theta=true_theta,
        data_path=data_path,
        prior_m0=prior_m0,
        prior_S0=prior_S0,
        engine_config=engine_config,
        env_config=env_config,
        model_config=model_config,
        rng=rng,
    )
    steps: list[EpisodeStep] = []

    for step_index in range(max_steps):
        step = step_agent_episode(context, step_index, top_k=top_k)
        steps.append(step)
        if step.action == "Purchase":
            break

    stats = context.tracker.get_stats()
    stats["steps_taken"] = len(steps)
    stats["step_data"] = [step.to_dict() for step in steps]
    return stats


def run_baseline_episode(
    true_theta: npt.NDArray[np.float64],
    data_path: str,
    prior_m0: npt.NDArray[np.float64],
    prior_S0: npt.NDArray[np.float64],
    eps_reg: float = 1.0,
) -> dict[str, Any]:
    """Run the greedy first-step baseline used in the experiments."""
    theta = np.asarray(true_theta, dtype=np.float64)
    model = BayesianPreferenceModel(d=len(theta), m0=prior_m0, S0=prior_S0)
    env = StochasticMarket(config=EnvConfig(data_path=data_path))
    obs = env.observe()

    if obs.features.shape[0] == 0:
        return {
            "queries": 0,
            "delays": 0,
            "purchased": False,
            "realized_regret": 0.0,
            "estimated_worst_case_regret": 0.0,
            "exceeded_regret": False,
        }

    expected_utils = model.expected_utility(obs.features)
    best_idx = int(np.argmax(expected_utils))
    realized_regret = compute_realized_regret(obs, theta, best_idx)
    return {
        "queries": 0,
        "delays": 0,
        "purchased": True,
        "realized_regret": realized_regret,
        "estimated_worst_case_regret": 0.0,
        "exceeded_regret": realized_regret > eps_reg,
    }


def compute_realized_regret(
    obs: Observation,
    true_theta: npt.NDArray[np.float64],
    chosen_idx_in_obs: int,
) -> float:
    """Compute regret relative to the best currently available item."""
    utilities = obs.features @ true_theta
    return float(np.max(utilities) - utilities[chosen_idx_in_obs])
