import os
from ..application.interfaces import ModelStore
from .model_store_joblib import JoblibModelStore
from .predictor_onnx import OnnxPredictionModel

class ModelStoreFactory(ModelStore):
    def load(self):
        t = os.getenv("RISK_MODEL_TYPE", "joblib").lower()
        if t == "onnx":
            path = os.getenv("RISK_ONNX_PATH", "risk_model.onnx")
            model = OnnxPredictionModel(path)
            return {"model": model, "bins": None}
        return JoblibModelStore().load()
    def save(self, payload):
        JoblibModelStore().save(payload)

