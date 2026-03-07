"""Bayesian RNN (Pyro) for TEP fault detection.

Adapted from user-provided VariableBRNN code. Uses PyroSample weight priors
(Normal(0,1)) on all RNN and output layer parameters.

Training: use Pyro SVI with AutoNormal guide and TraceMeanField_ELBO.
See scripts/tune_bayesian.py and scripts/train_best_bayesian.py.
"""

import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


class BayesianRNN(PyroModule):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int,
                 output_dim: int, dropout_prob: float = 0.0, seq_len: int = 5):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = input_dim // seq_len
        self.output_dim = output_dim

        hidden_sizes = [hidden_dim] * hidden_layers
        self.dropout = nn.Dropout(p=dropout_prob)

        layer_list = []
        for i, h in enumerate(hidden_sizes):
            in_sz = self.n_features if i == 0 else hidden_sizes[i - 1]
            layer = PyroModule[nn.RNN](in_sz, h, batch_first=True)
            layer.weight_ih_l0 = PyroSample(
                dist.Normal(0., 1.).expand([h, in_sz]).to_event(2))
            layer.weight_hh_l0 = PyroSample(
                dist.Normal(0., 1.).expand([h, h]).to_event(2))
            layer.bias_ih_l0 = PyroSample(
                dist.Normal(0., 1.).expand([h]).to_event(1))
            layer.bias_hh_l0 = PyroSample(
                dist.Normal(0., 1.).expand([h]).to_event(1))
            layer_list.append(layer)

        self.rnn_layers = nn.ModuleList(layer_list)

        self.fc = PyroModule[nn.Linear](hidden_dim, output_dim)
        self.fc.weight = PyroSample(
            dist.Normal(0., 1.).expand([output_dim, hidden_dim]).to_event(2))
        self.fc.bias = PyroSample(
            dist.Normal(0., 1.).expand([output_dim]).to_event(1))

    def forward(self, x, y=None):
        # x: (batch, input_dim) → (batch, seq_len, n_features)
        out = x.view(x.size(0), self.seq_len, self.n_features)
        for rnn in self.rnn_layers:
            out, _ = rnn(out)
            out = self.dropout(out)

        last = out[:, -1, :]        # (batch, hidden_dim)
        logits = self.fc(last)      # (batch, output_dim)

        if y is not None:
            with pyro.plate("data", x.size(0)):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)

        return logits
