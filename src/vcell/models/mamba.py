import torch
import torch.nn as nn


class MLPDeltaPredictor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        n_genes: int,
        hidden_dim: int = 512,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_genes),
        )

    def forward(self, batch):
        return self.net(batch["target_features"])