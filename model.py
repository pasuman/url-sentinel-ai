import torch
import torch.nn as nn


class PhishingMLP(nn.Module):
    """
    Simple MLP for phishing URL detection on pre-extracted features.

    Architecture: Linear(input_dim, 128) -> ReLU -> Dropout
                  -> Linear(128, 64) -> ReLU -> Dropout
                  -> Linear(64, 1) -> Sigmoid
    """

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(1)
