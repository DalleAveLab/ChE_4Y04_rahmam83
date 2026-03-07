"""Vanilla RNN for TEP fault detection."""

import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int,
                 output_dim: int, dropout_prob: float = 0.0, seq_len: int = 5):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = input_dim // seq_len

        # PyTorch only applies inter-layer dropout; warns and ignores when num_layers=1
        rnn_dropout = dropout_prob if hidden_layers > 1 else 0.0
        self.rnn = nn.RNN(
            input_size=self.n_features,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), self.seq_len, self.n_features)
        out, _ = self.rnn(x)
        return self.output_layer(out[:, -1, :])
