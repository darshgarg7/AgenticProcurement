from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any
import numpy as np
import numpy.typing as npt

@dataclass
class Observation:
    item_ids: List[Any]
    features: npt.NDArray[np.float64]

@dataclass
class Purchase:
    item_id: Any

@dataclass
class QueryUser:
    item_id: Any

@dataclass
class Search:
    pass

@dataclass
class Wait:
    pass

Action = Purchase | QueryUser | Search | Wait

class IPreferenceModel(ABC):
    @abstractmethod
    def update(self, x: npt.NDArray[np.float64], y: float) -> None:
        pass
        
    @abstractmethod
    def expected_utility(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass
        
    @abstractmethod
    def epistemic_uncertainty(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass
        
    @abstractmethod
    def sample_theta(self, n: int = 1) -> npt.NDArray[np.float64]:
        pass

class IMarketEnvironment(ABC):
    @abstractmethod
    def observe(self) -> Observation:
        pass
        
    @abstractmethod
    def step(self) -> None:
        pass

class IDecisionEngine(ABC):
    @abstractmethod
    def decide(self, obs: Observation, verbose: bool = False) -> Action:
        pass
