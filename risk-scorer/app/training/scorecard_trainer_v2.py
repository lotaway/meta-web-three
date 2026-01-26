import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ..infrastructure.binning_scorecardpy import ScorecardBinning
from ..infrastructure.model_store_joblib import JoblibModelStore
from ..infrastructure.iv_filter import select_by_iv
from ..infrastructure.evaluation_scorecardpy import evaluate_auc_ks
from ..infrastructure.scorecard_mapping import build_scorecard
from ..config import target_label, scoring_params


def clean_dataset(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Clean the dataset:
    - Drop column 'Unnamed: 0' if it exists (index column in Kaggle datasets)
    - Fill missing values with median
    - Ensure target is numeric
    """
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    # Fill missing values: MonthlyIncome and NumberOfDependents are often missing in GMSC dataset
    df = df.fillna(df.median())
    
    if target in df.columns:
        df[target] = pd.to_numeric(df[target])
        
    return df


def train_v2(
    df: pd.DataFrame, 
    target: str = None, 
    iv_threshold: float = 0.02,
    test_size: float = 0.2
):
    """
    Advanced training pipeline:
    1. Preprocessing and cleaning
    2. Binning (WoE)
    3. Feature selection by IV
    4. Data splitting
    5. Logistic Regression Modeling
    6. Scorecard Mapping (Points conversion)
    7. Evaluation
    """
    t = target if target else target_label()
    if t not in df.columns and "SeriousDlqin2yrs" in df.columns:
        t = "SeriousDlqin2yrs" # Auto-detect for GMSC dataset
        
    df = clean_dataset(df, t)
    
    # Generate Bins
    binning_svc = ScorecardBinning()
    bins = binning_svc.generate(df, t)
    
    # Apply WoE
    df_woe = binning_svc.apply(df, bins)
    
    # Filter variables by IV
    keep_vars = select_by_iv(bins, iv_threshold)
    
    # Prepare X and y
    cols = [c for c in df_woe.columns if c in keep_vars and c != t]
    if not cols:
        # Fallback to all if IV filter is too aggressive
        cols = [c for c in df_woe.columns if c != t]
        
    X = df_woe[cols]
    y = df_woe[t]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train Model
    model = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs')
    model.fit(X_train, y_train)
    
    # Build Scorecard (Mapping WoE to Points)
    params = scoring_params()
    scorecard = build_scorecard(
        bins, 
        model, 
        cols,
        pdo=params.get("pdo", 50), 
        base_score=params.get("base_score", 600)
    )
    
    # Evaluate
    train_metrics = evaluate_auc_ks(model, X_train, y_train)
    test_metrics = evaluate_auc_ks(model, X_test, y_test)
    
    payload = {
        "model": model,
        "bins": bins,
        "scorecard": scorecard,
        "features": cols,
        "metrics": {
            "train": train_metrics,
            "test": test_metrics
        },
        "target": t
    }
    
    return payload


def save_payload(payload):
    JoblibModelStore().save(payload)
