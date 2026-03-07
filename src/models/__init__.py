"""Model registry for TEP fault detection."""

from .efficient_kan import EfficientKAN
from .fourier_kan import FourierKAN
from .wavelet_kan import WaveletKAN
from .fast_kan import FastKAN
from .mlp import MLP
from .cnn import CNN
from .rnn import RNN
from .lstm import LSTM

MODEL_REGISTRY = {
    'efficient_kan': EfficientKAN,
    'fourier_kan':   FourierKAN,
    'wavelet_kan':   WaveletKAN,
    'fast_kan':      FastKAN,
    'mlp':           MLP,
    'cnn':           CNN,
    'rnn':           RNN,
    'lstm':          LSTM,
}