import numpy as np
import onnxruntime as ort
from ..application.interfaces import PredictionModel

class OnnxPredictionModel(PredictionModel):
    def __init__(self, path: str):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.order = ["age", "external_debt_ratio", "first_order", "gps_stability", "device_shared_degree"]

    def predict_proba(self, X):
        vectors = []
        for _, row in X.iterrows():
            v = []
            for key in self.order:
                val = row.get(key, 0)
                if key == "first_order":
                    v.append(1.0 if bool(val) else 0.0)
                else:
                    v.append(float(val))
            vectors.append(v)
        arr = np.array(vectors, dtype=np.float32)
        out = self.sess.run([self.output_name], {self.input_name: arr})[0]
        probs = np.array(out)
        if probs.ndim == 3:
            probs = probs[:, 0, :]
        return probs

