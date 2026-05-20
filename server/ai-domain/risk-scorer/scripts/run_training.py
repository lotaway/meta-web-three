import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.training.scorecard_trainer_v2 import CreditRiskTrainingPipeline, save_training_artifact

def run_production_training_session():
    dataset_path = PROJECT_ROOT / "docs" / "dataset" / "GiveMeSomeCredit" / "cs-training.csv"
    if not dataset_path.exists():
        return

    raw_historical_records = pd.read_csv(dataset_path)
    
    pipeline = CreditRiskTrainingPipeline()
    training_artifact = pipeline.execute_training_workflow(raw_historical_records)
    
    _display_performance_report(training_artifact)
    save_training_artifact(training_artifact)

def _display_performance_report(artifact):
    metrics = artifact['performance_metrics']
    print("\n--- 训练执行报告 ---")
    print(f"入模特征数: {len(artifact['selected_features'])}")
    print(f"验证集评估 (AUC): {metrics['test']['auc']:.4f}")
    print(f"验证集评估 (KS): {metrics['test']['ks']:.4f}")

if __name__ == "__main__":
    run_production_training_session()
