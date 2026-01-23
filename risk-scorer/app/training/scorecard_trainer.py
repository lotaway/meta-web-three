import pandas as pd
from sklearn.linear_model import LogisticRegression
from ..infrastructure.binning_scorecardpy import ScorecardBinning
from ..infrastructure.model_store_joblib import JoblibModelStore
from ..infrastructure.iv_filter import select_by_iv
from ..infrastructure.evaluation_scorecardpy import evaluate_auc_ks
from ..infrastructure.scorecard_mapping import build_scorecard
from ..config import target_label


def train_from_dataframe(
    df: pd.DataFrame, target: str = None, iv_threshold: float = 0.02
):
    t = target if target else target_label()
    bins = ScorecardBinning().generate(df, t)
    df_woe = ScorecardBinning().apply(df, bins)
    keep_vars = select_by_iv(bins, iv_threshold)
    cols = [c for c in df_woe.columns if c in keep_vars and c != t]
    X = (
        df_woe[cols]
        if len(cols) > 0
        else (df_woe.drop(columns=[t]) if t in df_woe.columns else df_woe)
    )
    y = df[t] if t in df.columns else df_woe[t]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    metrics = evaluate_auc_ks(model, X, y)
    return {"model": model, "bins": bins, "metrics": metrics}


def save_payload(payload):
    JoblibModelStore().save(payload)
