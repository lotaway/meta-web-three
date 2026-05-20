from RiskScorerService_pb2_grpc import RiskScorerServiceServicer
from RiskScorerService_pb2 import TestRequest, TestResponse, ScoreRequest, ScoreResponse
from .application.scoring_usecase import RiskScoringUseCase
from .application.preprocess import to_dataframe
from .infrastructure.binning_scorecardpy import ScorecardBinning
from .infrastructure.model_store_joblib import JoblibModelStore
from .infrastructure.model_store_factory import ModelStoreFactory
from .config import scoring_params, target_label
import grpc


def build_usecase():
    params = scoring_params()
    return RiskScoringUseCase(
        binning=ScorecardBinning(),
        model_store=ModelStoreFactory(),
        base_score=params["base_score"],
        pdo=params["pdo"],
        approve_threshold=params["approve_threshold"],
        review_threshold=params["review_threshold"],
    )


class RiskScorerGrpcService(RiskScorerServiceServicer):
    def __init__(self):
        self.usecase = build_usecase()

    def test(self, request: TestRequest, context):
        return TestResponse(result=100)

    def score(self, request: ScoreRequest, context):
        try:
            features = self._extract_features(request)
            df = to_dataframe(features)
            score, decision = self.usecase.score(df, target_label())
            return ScoreResponse(score=score, decision=decision)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ScoreResponse()

    def _extract_features(self, request: ScoreRequest):
        m = {}
        for k, f in request.features.items():
            if f.HasField("age"):
                m["age"] = f.age
            elif f.HasField("external_debt_ratio"):
                m["external_debt_ratio"] = f.external_debt_ratio
            elif f.HasField("first_order"):
                m["first_order"] = f.first_order
            elif f.HasField("gps_stability"):
                m["gps_stability"] = f.gps_stability
            elif f.HasField("device_shared_degree"):
                m["device_shared_degree"] = f.device_shared_degree
        return m
