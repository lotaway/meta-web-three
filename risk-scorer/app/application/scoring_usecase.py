from typing import Dict, Any, Tuple
from .interfaces import BinningService, ModelStore, PredictionModel
from ..domain.scoring_policy import pd_to_score, decision_by_score
from ..domain.exceptions import (
    ModelNotFoundError,
    PredictionFailedError,
    InvalidInputError,
)


class RiskScoringUseCase:
    def __init__(
        self,
        binning: BinningService,
        model_store: ModelStore,
        base_score: int,
        pdo: int,
        approve_threshold: int,
        review_threshold: int,
    ):
        self.binning = binning
        self.model_store = model_store
        self.base_score = base_score
        self.pdo = pdo
        self.approve_threshold = approve_threshold
        self.review_threshold = review_threshold

    def score(self, df: Any, target: str) -> Tuple[int, str]:
        if df is None or target is None or target == "":
            raise InvalidInputError("missing_input")
        payload = self.model_store.load()
        if "model" not in payload or "bins" not in payload:
            raise ModelNotFoundError("model_payload_invalid")
        model = payload["model"]
        bins = payload["bins"]
        df_woe = self.binning.apply(df, bins) if bins is not None else df
        X = df_woe.drop(columns=[target]) if target in df_woe.columns else df_woe
        pd = self._predict_pd(model, X)
        score = pd_to_score(pd, self.base_score, self.pdo)
        decision = decision_by_score(
            score, self.approve_threshold, self.review_threshold
        )
        return score, decision

    def _predict_pd(self, model: PredictionModel, X: Any) -> float:
        try:
            proba = model.predict_proba(X)
            return float(proba[:, 1][0])
        except Exception as e:
            raise PredictionFailedError(str(e))
