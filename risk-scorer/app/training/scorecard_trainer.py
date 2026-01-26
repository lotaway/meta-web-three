import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from ..infrastructure.model_store_joblib import JoblibModelStore
from ..config import target_label

# ---------------------------------------------------------
# "Big Model" Style Architecture (Deep Learning / NN)
# ---------------------------------------------------------

class RiskScorerNN(nn.Module):
    def __init__(self, input_dim):
        super(RiskScorerNN, self).__init__()
        # Using a Multi-Layer Perceptron (MLP) architecture
        self.layer1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.GELU()  # Modern activation function
        self.dropout1 = nn.Dropout(0.2)
        
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.GELU()
        
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return self.sigmoid(x)

def train_from_dataframe(
    df: pd.DataFrame, target: str = None, epochs: int = 50, batch_size: int = 1024
):
    """
    Neural Network Training Pipeline (Deep Learning Version)
    """
    t = target if target else target_label()
    
    # 1. Basic Cleaning for NN (NN doesn't handle missing values automatically)
    df = df.fillna(df.median())
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    X = df.drop(columns=[t])
    y = df[t].values.reshape(-1, 1)
    
    # 2. Scaling (Critical for Neural Networks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # 4. Initialize Model, Loss and Optimizer
    model = RiskScorerNN(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Training Loop
    model.train()
    for epoch in range(epochs):
        # Mini-batch training (simplified)
        permutation = torch.randperm(X_train_t.size()[0])
        for i in range(0, X_train_t.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_t[indices], y_train_t[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # 6. Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).numpy()
        auc = roc_auc_score(y_test, preds)
    
    return {
        "model": model,
        "scaler": scaler,
        "metrics": {"auc": float(auc)},
        "type": "neural_network"
    }

def save_payload(payload):
    JoblibModelStore().save(payload)
