from typing import Dict, Any
import pandas as pd
from ..domain.training_protocol import RiskModel, TrainingData
from ..infrastructure.deep_learning.torch_risk_model import TorchDeepRiskModel
from ..infrastructure.model_store_joblib import JoblibModelStore
from ..config import target_label

class HistoricalCreditDataset:
    def __init__(self, raw_data: pd.DataFrame, target_name: str):
        self._processed_data = self._clean_raw_data(raw_data)
        self._target_name = target_name

    def _clean_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        cleaned = data.fillna(data.median())
        return cleaned.drop(columns=["Unnamed: 0"], errors="ignore")

    def split_features_and_target(self):
        x = self._processed_data.drop(columns=[self._target_name])
        y = self._processed_data[self._target_name]
        return x, y

def execute_neural_risk_training_workflow(
    raw_historical_records: pd.DataFrame, 
    defaulter_indicator: str = None
) -> Dict[str, Any]:
    indicator = defaulter_indicator or target_label()
    dataset = HistoricalCreditDataset(raw_historical_records, indicator)
    
    features, target = dataset.split_features_and_target()
    
    trained_model = _optimize_deep_model(features, target)
    
    return _persist_training_payload(trained_model)

def _optimize_deep_model(features, target) -> RiskModel:
    import torch
    x_tensor = torch.FloatTensor(features.values)
    y_tensor = torch.FloatTensor(target.values).reshape(-1, 1)
    
    model = TorchDeepRiskModel(feature_count=features.shape[1])
    model.optimize_parameters(x_tensor, y_tensor)
    return model

def _persist_training_payload(model: RiskModel) -> Dict[str, Any]:
    artifact = model.export_artifact()
    JoblibModelStore().save(artifact)
    return artifact
