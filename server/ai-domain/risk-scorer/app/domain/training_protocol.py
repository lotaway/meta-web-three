from typing import Protocol, TypeVar, Any, Dict
from abc import abstractmethod

T_Data = TypeVar("T_Data")
T_Model = TypeVar("T_Model")

class TrainingData(Protocol):
    @abstractmethod
    def extract_features(self) -> Any:
        ...
    
    @abstractmethod
    def extract_target(self) -> Any:
        ...

class RiskModel(Protocol):
    @abstractmethod
    def optimize_parameters(self, features: Any, target: Any) -> None:
        ...

    @abstractmethod
    def export_artifact(self) -> Dict[str, Any]:
        ...

class PerformanceValidator(Protocol):
    @abstractmethod
    def calculate_discrimination_index(self, model: RiskModel, data: TrainingData) -> Dict[str, float]:
        ...
