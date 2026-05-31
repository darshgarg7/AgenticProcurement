"""Offline evaluation on logged procurement candidate sets."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from config.settings import EngineConfig
from core.interfaces import IPreferenceModel, Observation, Purchase
from decision.delegation_engine import DelegationEngine


@dataclass(frozen=True)
class ProcurementSnapshot:
    """One real or logged procurement decision context."""

    request_id: str
    item_ids: list[str]
    features: npt.NDArray[np.float64]
    realized_utilities: npt.NDArray[np.float64]
    feature_columns: list[str]
    logged_choice_item_id: str | None = None

    def __post_init__(self) -> None:
        """Validate snapshot dimensions and utility coverage."""
        if self.features.ndim != 2:
            raise ValueError("features must be a 2D array")
        if self.realized_utilities.ndim != 1:
            raise ValueError("realized_utilities must be a 1D array")
        if len(self.item_ids) != self.features.shape[0]:
            raise ValueError("item_ids length must match feature rows")
        if len(self.item_ids) != self.realized_utilities.shape[0]:
            raise ValueError("item_ids length must match realized utilities")
        if len(self.feature_columns) != self.features.shape[1]:
            raise ValueError("feature_columns length must match feature dimension")

    def observation(self) -> Observation:
        """Return the policy-facing observation for this snapshot."""
        return Observation(item_ids=list(self.item_ids), features=self.features.copy())

    def regret_for_item(self, item_id: str) -> float:
        """Compute realized regret for a selected item in this snapshot."""
        if item_id not in self.item_ids:
            raise ValueError(f"item_id {item_id!r} is not present in request {self.request_id!r}")
        selected_idx = self.item_ids.index(item_id)
        return float(np.max(self.realized_utilities) - self.realized_utilities[selected_idx])

    def logged_choice_regret(self) -> float | None:
        """Return regret of the logged choice when the log identifies one."""
        if self.logged_choice_item_id is None:
            return None
        return self.regret_for_item(self.logged_choice_item_id)


@dataclass(frozen=True)
class LoggedProcurementDataset:
    """Collection of logged procurement snapshots loaded from external data."""

    snapshots: list[ProcurementSnapshot]
    feature_columns: list[str]
    source_path: str

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        feature_columns: list[str] | None = None,
        request_column: str = "request_id",
        item_column: str = "item_id",
        utility_column: str = "realized_utility",
        choice_column: str = "chosen_item_id",
        chosen_flag_column: str = "was_chosen",
    ) -> LoggedProcurementDataset:
        """Load a logged decision dataset from a row-oriented CSV file.

        Expected minimum schema:
        ``request_id,item_id,realized_utility,<feature columns...>``.
        A logged choice may be represented either by a repeated
        ``chosen_item_id`` value or by one row with ``was_chosen`` set to true.
        """
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path} is missing a CSV header")
            rows = list(reader)

        if not rows:
            raise ValueError("logged procurement dataset must contain at least one row")

        fieldnames = list(reader.fieldnames)
        required = {request_column, item_column, utility_column}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"logged procurement dataset missing columns: {sorted(missing)}")

        reserved = {
            request_column,
            item_column,
            utility_column,
            choice_column,
            chosen_flag_column,
        }
        features = feature_columns or _infer_feature_columns(rows, fieldnames, reserved)
        if not features:
            raise ValueError("logged procurement dataset must include numeric feature columns")

        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row[request_column]].append(row)

        snapshots = [
            _snapshot_from_rows(
                request_id=request_id,
                rows=request_rows,
                feature_columns=features,
                item_column=item_column,
                utility_column=utility_column,
                choice_column=choice_column,
                chosen_flag_column=chosen_flag_column,
            )
            for request_id, request_rows in grouped.items()
        ]
        return cls(snapshots=snapshots, feature_columns=features, source_path=str(path))


@dataclass(frozen=True)
class OfflinePolicyEvaluation:
    """Aggregated replay metrics for a policy on logged snapshots."""

    num_snapshots: int
    purchases: int
    abstentions: int
    avg_regret: float
    max_regret: float
    regret_exceedance_rate: float
    action_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metrics dictionary."""
        return asdict(self)


def evaluate_logged_choices(dataset: LoggedProcurementDataset) -> OfflinePolicyEvaluation:
    """Evaluate the regret of choices recorded in the procurement log."""
    regrets = [
        regret
        for snapshot in dataset.snapshots
        if (regret := snapshot.logged_choice_regret()) is not None
    ]
    abstentions = len(dataset.snapshots) - len(regrets)
    return _aggregate_regrets(
        num_snapshots=len(dataset.snapshots),
        regrets=regrets,
        abstentions=abstentions,
        action_counts={"LoggedChoice": len(regrets), "MissingChoice": abstentions},
        threshold=None,
    )


def evaluate_agent_on_snapshots(
    dataset: LoggedProcurementDataset,
    model: IPreferenceModel,
    engine_config: EngineConfig | None = None,
    rng: Any | None = None,
) -> OfflinePolicyEvaluation:
    """Replay the current agent policy against logged procurement snapshots."""
    engine = DelegationEngine(model, config=engine_config, rng=rng)
    regrets: list[float] = []
    action_counts = {"Purchase": 0, "QueryUser": 0, "Wait": 0, "Search": 0}

    for snapshot in dataset.snapshots:
        action = engine.decide(snapshot.observation())
        action_name = action.__class__.__name__
        action_counts[action_name] = action_counts.get(action_name, 0) + 1
        if isinstance(action, Purchase):
            regrets.append(snapshot.regret_for_item(str(action.item_id)))

    return _aggregate_regrets(
        num_snapshots=len(dataset.snapshots),
        regrets=regrets,
        abstentions=len(dataset.snapshots) - len(regrets),
        action_counts=action_counts,
        threshold=engine.config.eps_reg,
    )


def _snapshot_from_rows(
    request_id: str,
    rows: list[dict[str, str]],
    feature_columns: list[str],
    item_column: str,
    utility_column: str,
    choice_column: str,
    chosen_flag_column: str,
) -> ProcurementSnapshot:
    item_ids = [row[item_column] for row in rows]
    features = np.array(
        [[float(row[column]) for column in feature_columns] for row in rows],
        dtype=np.float64,
    )
    utilities = np.array([float(row[utility_column]) for row in rows], dtype=np.float64)
    logged_choice = _logged_choice_from_rows(rows, item_column, choice_column, chosen_flag_column)
    return ProcurementSnapshot(
        request_id=request_id,
        item_ids=item_ids,
        features=features,
        realized_utilities=utilities,
        feature_columns=feature_columns,
        logged_choice_item_id=logged_choice,
    )


def _logged_choice_from_rows(
    rows: list[dict[str, str]],
    item_column: str,
    choice_column: str,
    chosen_flag_column: str,
) -> str | None:
    for row in rows:
        if choice_column in row and row[choice_column]:
            return row[choice_column]

    for row in rows:
        if chosen_flag_column in row and row[chosen_flag_column].strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }:
            return row[item_column]
    return None


def _infer_feature_columns(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    reserved: set[str],
) -> list[str]:
    feature_columns = []
    for column in fieldnames:
        if column in reserved:
            continue
        try:
            for row in rows:
                float(row[column])
        except (TypeError, ValueError):
            continue
        feature_columns.append(column)
    return feature_columns


def _aggregate_regrets(
    num_snapshots: int,
    regrets: list[float],
    abstentions: int,
    action_counts: dict[str, int],
    threshold: float | None,
) -> OfflinePolicyEvaluation:
    exceedances = [
        regret > threshold
        for regret in regrets
    ] if threshold is not None else []
    return OfflinePolicyEvaluation(
        num_snapshots=num_snapshots,
        purchases=len(regrets),
        abstentions=abstentions,
        avg_regret=float(np.mean(regrets)) if regrets else 0.0,
        max_regret=float(np.max(regrets)) if regrets else 0.0,
        regret_exceedance_rate=float(np.mean(exceedances)) if exceedances else 0.0,
        action_counts=action_counts,
    )
