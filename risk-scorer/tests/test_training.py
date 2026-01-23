import pandas as pd
from app.training.scorecard_trainer import train_from_dataframe, save_payload
from app.infrastructure.model_store_joblib import JoblibModelStore
from app.config import model_store_path
import os

def test_train_and_save_payload(tmp_path):
    os.environ["RISK_MODEL_PATH"] = str(tmp_path / "scorecard.pkl")
    df = pd.DataFrame([
        {"age":25,"external_debt_ratio":0.2,"first_order":True,"gps_stability":0.8,"device_shared_degree":1,"default_flag":1},
        {"age":40,"external_debt_ratio":0.0,"first_order":False,"gps_stability":0.5,"device_shared_degree":2,"default_flag":0},
        {"age":30,"external_debt_ratio":0.1,"first_order":True,"gps_stability":0.6,"device_shared_degree":1,"default_flag":0},
        {"age":22,"external_debt_ratio":0.3,"first_order":True,"gps_stability":0.7,"device_shared_degree":3,"default_flag":1},
    ])
    payload = train_from_dataframe(df, "default_flag")
    save_payload(payload)
    loaded = JoblibModelStore().load()
    assert "model" in loaded and "bins" in loaded
    assert "metrics" in payload and "auc" in payload["metrics"] and "ks" in payload["metrics"]
