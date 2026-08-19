"""Neural network initialization and activation layer utilities."""

import numpy as np
import torch
import torch.nn as nn


def layer_init_orthogonal(layer, std=np.sqrt(2), bias_const=0.0):
    """Initializes linear or conv layer weights with orthogonal matrix and constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def activation_factory(activation):
    """Instantiates PyTorch activation layer module by name."""
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'rrelu':
        return nn.RReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'silu':
        return nn.SiLU()
    else:
        raise NotImplementedError
