from app.domain.scoring_policy import pd_to_score, decision_by_score
import math

def test_pd_to_score_basic():
    s = pd_to_score(0.1, 600, 50)
    assert isinstance(s, int)
    assert s > 600

def test_pd_to_score_bounds():
    try:
        pd_to_score(0.0, 600, 50)
        assert False
    except ValueError:
        assert True
    try:
        pd_to_score(1.0, 600, 50)
        assert False
    except ValueError:
        assert True

def test_decision_by_score():
    d1 = decision_by_score(750, 700, 600)
    d2 = decision_by_score(650, 700, 600)
    d3 = decision_by_score(550, 700, 600)
    assert d1 == "approve"
    assert d2 == "review"
    assert d3 == "reject"

