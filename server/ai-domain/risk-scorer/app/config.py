import os
from pathlib import Path

def model_store_path() -> str:
    p = os.getenv("RISK_MODEL_PATH")
    if p and p.strip():
        return p
    return str(Path(__file__).parent.parent / "models" / "scorecard.pkl")

def target_label() -> str:
    v = os.getenv("RISK_TARGET_LABEL")
    if v and v.strip():
        return v
    return "default_flag"

def scoring_params() -> dict:
    approve = int(os.getenv("RISK_APPROVE_THRESHOLD", "700"))
    review = int(os.getenv("RISK_REVIEW_THRESHOLD", "600"))
    base = int(os.getenv("RISK_BASE_SCORE", "600"))
    pdo = int(os.getenv("RISK_PDO", "50"))
    return {
        "approve_threshold": approve,
        "review_threshold": review,
        "base_score": base,
        "pdo": pdo,
    }

