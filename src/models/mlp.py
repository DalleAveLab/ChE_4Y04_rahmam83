"""Multi-Layer Perceptron baseline for TEP fault detection."""

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int,
                 output_dim: int, dropout_prob: float = 0.0):
        super().__init__()

        self.hidden_blocks = nn.ModuleList()
        in_features = input_dim
        for _ in range(hidden_layers):
            self.hidden_blocks.append(nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
            ))
            in_features = hidden_dim

        self.output_layer = nn.Linear(in_features, output_dim)

    def forward(self, x):
        for block in self.hidden_blocks:
            x = block(x)
        return self.output_layer(x)
