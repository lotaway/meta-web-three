from typing import Tuple

def pd_to_score(pd: float, base_score: int, pdo: int) -> int:
    if pd <= 0 or pd >= 1:
        raise ValueError("pd_out_of_range")
    odds = (1 - pd) / pd
    points = pdo * (log_odds(odds))
    return int(round(base_score + points))

def log_odds(odds: float) -> float:
    if odds <= 0:
        raise ValueError("odds_out_of_range")
    import math
    return math.log2(odds)

def decision_by_score(score: int, approve_threshold: int, review_threshold: int) -> str:
    if score >= approve_threshold:
        return "approve"
    if score >= review_threshold:
        return "review"
    return "reject"

