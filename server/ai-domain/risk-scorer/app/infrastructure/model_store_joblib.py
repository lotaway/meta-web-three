import joblib
from ..domain.exceptions import ModelNotFoundError
from ..config import model_store_path

class JoblibModelStore:
    def load(self):
        path = model_store_path()
        try:
            return joblib.load(path)
        except Exception as e:
            raise ModelNotFoundError(str(e))
    def save(self, payload):
        path = model_store_path()
        joblib.dump(payload, path)

