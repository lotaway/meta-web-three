import pandas as pd
from app.training.scorecard_trainer_v2 import CreditRiskTrainingPipeline, save_training_artifact
from app.infrastructure.model_store_joblib import JoblibModelStore
import os

def test_train_and_save_payload(tmp_path):
    os.environ["RISK_MODEL_PATH"] = str(tmp_path / "scorecard.pkl")
    df = pd.DataFrame([
        {"age":25,"external_debt_ratio":0.2,"first_order":True,"gps_stability":0.8,"device_shared_degree":1,"default_flag":1},
        {"age":40,"external_debt_ratio":0.0,"first_order":False,"gps_stability":0.5,"device_shared_degree":2,"default_flag":0},
        {"age":30,"external_debt_ratio":0.1,"first_order":True,"gps_stability":0.6,"device_shared_degree":1,"default_flag":0},
        {"age":22,"external_debt_ratio":0.3,"first_order":True,"gps_stability":0.7,"device_shared_degree":3,"default_flag":1},
        {"age":35,"external_debt_ratio":0.15,"first_order":False,"gps_stability":0.9,"device_shared_degree":1,"default_flag":0},
        {"age":28,"external_debt_ratio":0.05,"first_order":True,"gps_stability":0.4,"device_shared_degree":2,"default_flag":1},
        {"age":45,"external_debt_ratio":0.12,"first_order":False,"gps_stability":0.85,"device_shared_degree":1,"default_flag":0},
        {"age":20,"external_debt_ratio":0.25,"first_order":True,"gps_stability":0.75,"device_shared_degree":3,"default_flag":1},
    ])
    pipeline = CreditRiskTrainingPipeline()
    payload = pipeline.execute_training_workflow(df, "default_flag")
    save_training_artifact(payload)
    
    loaded = JoblibModelStore().load()
    assert "model" in loaded and "bins" in loaded
    assert "performance_metrics" in payload 
    assert "train" in payload["performance_metrics"] and "test" in payload["performance_metrics"]
    assert "auc" in payload["performance_metrics"]["train"]
