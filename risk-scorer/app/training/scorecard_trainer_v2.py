from typing import Dict, Any, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from ..infrastructure.binning_scorecardpy import ScorecardBinning
from ..infrastructure.model_store_joblib import JoblibModelStore
from ..infrastructure.iv_filter import select_by_iv
from ..infrastructure.evaluation_scorecardpy import evaluate_auc_ks
from ..infrastructure.scorecard_mapping import build_scorecard
from ..config import target_label, scoring_params

class CreditRiskTrainingPipeline:
    def __init__(self, iv_exclusion_threshold: float = 0.02):
        self._iv_threshold = iv_exclusion_threshold
        self._binning_service = ScorecardBinning()

    def execute_training_workflow(
        self, 
        historical_records: pd.DataFrame, 
        target_field: str = None
    ) -> Dict[str, Any]:
        indicator = target_field or target_label()
        cleaned_data = self._preprocess_raw_records(historical_records, indicator)
        
        character_bins = self._binning_service.generate(cleaned_data, indicator)
        woe_encoded_data = self._binning_service.apply(cleaned_data, character_bins)
        active_features = select_by_iv(character_bins, self._iv_threshold)
        
        return self._build_predictive_model(woe_encoded_data, indicator, active_features, character_bins)

    def _preprocess_raw_records(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        data = data.drop(columns=["Unnamed: 0"], errors="ignore")
        data = data.fillna(data.median())
        data[target] = pd.to_numeric(data[target])
        return data

    def _build_predictive_model(
        self, 
        encoded_data: pd.DataFrame, 
        target: str, 
        features: List[str],
        bins: Any
    ) -> Dict[str, Any]:
        x_train, x_test, y_train, y_test = self._split_performance_data(encoded_data, target, features)
        
        regression_optimizer = LogisticRegression(max_iter=1000, C=0.1)
        regression_optimizer.fit(x_train, y_train)
        
        card = self._generate_scorecard_mapping(bins, regression_optimizer, features)
        metrics = self._validate_model_performance(regression_optimizer, x_train, y_train, x_test, y_test)
        
        return self._assemble_training_payload(regression_optimizer, bins, card, features, metrics, target)

    def _split_performance_data(self, data: pd.DataFrame, target: str, features: List[str]):
        return train_test_split(data[features], data[target], test_size=0.2, random_state=42)

    def _generate_scorecard_mapping(self, bins, model, features):
        params = scoring_params()
        return build_scorecard(bins, model, features, pdo=params["pdo"], base_score=params["base_score"])

    def _validate_model_performance(self, model, x_tr, y_tr, x_te, y_te):
        return {
            "train": evaluate_auc_ks(model, x_tr, y_tr),
            "test": evaluate_auc_ks(model, x_te, y_te)
        }

    def _assemble_training_payload(self, model, bins, card, features, metrics, target):
        return {
            "model": model,
            "bins": bins,
            "scorecard": card,
            "selected_features": features,
            "performance_metrics": metrics,
            "target_variable": target
        }

def save_training_artifact(payload: Dict[str, Any]) -> None:
    JoblibModelStore().save(payload)
