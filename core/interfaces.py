"""Core dataclasses and abstract interfaces for the procurement system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class Observation:
    """Agent-visible market snapshot."""

    item_ids: list[Any]
    features: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("Observation.features must be a 2D array")
        if len(self.item_ids) != self.features.shape[0]:
            raise ValueError(
                "Observation.item_ids length must match the number of feature rows"
            )

@dataclass
class Purchase:
    """Terminal action that buys one item."""

    item_id: Any

@dataclass
class QueryUser:
    """Epistemic action that asks the user about one item."""

    item_id: Any

@dataclass
class Search:
    """Action that advances the market while browsing for options."""

    pass

@dataclass
class Wait:
    """Action that advances the market without issuing a user query."""

    pass

Action = Purchase | QueryUser | Search | Wait

@dataclass
class DecisionDiagnostics:
    """Numerical decision signals computed for an observation."""

    expected_utils: npt.NDArray[np.float64]
    epistemic_uncertainties: npt.NDArray[np.float64]
    worst_case_regrets: npt.NDArray[np.float64]
    best_idx: int | None
    ig_query: float | None = None
    wait_value: float | None = None
    wait_advantage: float | None = None

class IPreferenceModel(ABC):
    """Abstract preference model used by the delegation engine."""

    @abstractmethod
    def update(self, x: npt.NDArray[np.float64], y: float) -> None:
        """Update the model with one observed utility signal."""
        pass
        
    @abstractmethod
    def expected_utility(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return expected utility for each candidate feature row."""
        pass
        
    @abstractmethod
    def epistemic_uncertainty(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return reducible uncertainty for each candidate feature row."""
        pass
        
    @abstractmethod
    def sample_theta(self, n: int = 1, rng: Any | None = None) -> npt.NDArray[np.float64]:
        """Draw preference-parameter samples."""
        pass

class IMarketEnvironment(ABC):
    """Abstract market environment exposed to the agent."""

    @abstractmethod
    def observe(self) -> Observation:
        """Return the current observation."""
        pass
        
    @abstractmethod
    def step(self) -> None:
        """Advance the environment dynamics by one step."""
        pass

class IDecisionEngine(ABC):
    """Abstract policy that maps observations to actions."""

    @abstractmethod
    def decide(self, obs: Observation, verbose: bool = False) -> Action:
        """Choose the next agent action."""
        pass
