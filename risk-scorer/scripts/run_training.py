import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).parent.parent
sys.path.append(str(root))

from app.training.scorecard_trainer_v2 import train_v2, save_payload

def main():
    dataset_path = root / "docs" / "dataset" / "GiveMeSomeCredit" / "cs-training.csv"
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    print("Starting training v2...")
    payload = train_v2(df)
    
    print("\nTraining Metrics:")
    print(f"Target: {payload['target']}")
    print(f"Features: {payload['features']}")
    print(f"Train AUC: {payload['metrics']['train']['auc']:.4f}, KS: {payload['metrics']['train']['ks']:.4f}")
    print(f"Test AUC: {payload['metrics']['test']['auc']:.4f}, KS: {payload['metrics']['test']['ks']:.4f}")
    
    print("\nSaving model payload...")
    save_payload(payload)
    print("Done!")

if __name__ == "__main__":
    main()
