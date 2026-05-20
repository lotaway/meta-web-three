import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from ...domain.training_protocol import RiskModel

class NeuralRiskArchitecture(nn.Module):
    def __init__(self, input_dimension: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, feature_tensor):
        latent_state = self.encoder(feature_tensor)
        return self.classifier(latent_state)

class TorchDeepRiskModel(RiskModel):
    def __init__(self, feature_count: int, iteration_limit: int = 50):
        self._network = NeuralRiskArchitecture(feature_count)
        self._iteration_limit = iteration_limit
        self._optimizer = optim.Adam(self._network.parameters(), lr=0.001)
        self._objective_function = nn.BCELoss()

    def _execute_optimization_step(self, x, y):
        self._optimizer.zero_grad()
        prediction = self._network(x)
        loss = self._objective_function(prediction, y)
        loss.backward()
        self._optimizer.step()

    def export_artifact(self) -> Dict[str, Any]:
        return {
            "state_dict": self._network.state_dict(),
            "architecture": "MLP-GELU-v1"
        }
