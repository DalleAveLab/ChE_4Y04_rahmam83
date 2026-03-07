"""1D CNN (conv over time) for TEP fault detection."""

import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int,
                 output_dim: int, num_filters: int = 32, kernel_size: int = 3,
                 dropout_prob: float = 0.0, seq_len: int = 5):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = input_dim // seq_len  # channels for Conv1d

        # One Conv1d layer: treat features as channels, time as spatial dim
        self.conv = nn.Conv1d(
            in_channels=self.n_features,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding='same',
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

        # FC layers after global average pooling
        self.fc_layers = nn.ModuleList()
        in_features = num_filters
        for _ in range(hidden_layers):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
            ))
            in_features = hidden_dim

        self.output_layer = nn.Linear(in_features, output_dim)

    def forward(self, x):
        # (batch, 195) → (batch, n_features, seq_len)
        x = x.view(x.size(0), self.n_features, self.seq_len)
        x = self.relu(self.conv(x))      # (batch, num_filters, seq_len)
        x = self.dropout(x)
        x = x.mean(dim=-1)               # global avg pool → (batch, num_filters)
        for fc in self.fc_layers:
            x = fc(x)
        return self.output_layer(x)
