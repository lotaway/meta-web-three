import scorecardpy as sc
import pandas as pd

def build_scorecard(bins, model, pdo: int, base_score: int):
    card = sc.scorecard(bins, model, pdo=pdo, basepoints=base_score)
    return card

def to_yaml_mapping(scorecard_df: pd.DataFrame):
    out = []
    for _, row in scorecard_df.iterrows():
        out.append({
            "variable": row["variable"],
            "bin": row["bin"],
            "woe": float(row["woe"]),
            "score": int(row["points"]),
        })
    return out

