import pandas as pd
import numpy as np
from app.application.scoring_usecase import RiskScoringUseCase
from app.application.interfaces import BinningService, ModelStore

class DummyBinning(BinningService):
    def generate(self, df, target: str):
        return {}
    def apply(self, df, bins):
        return df

class DummyModel:
    def predict_proba(self, X):
        return np.array([[0.8, 0.2]])

class DummyStore(ModelStore):
    def load(self):
        return {"model": DummyModel(), "bins": {}}
    def save(self, payload):
        pass

def test_usecase_score():
    df = pd.DataFrame([{"age":30,"income":10000}])
    uc = RiskScoringUseCase(
        binning=DummyBinning(),
        model_store=DummyStore(),
        base_score=600,
        pdo=50,
        approve_threshold=700,
        review_threshold=600,
    )
    score, decision = uc.score(df, "default_flag")
    assert isinstance(score, int)
    assert decision in {"approve","review","reject"}

