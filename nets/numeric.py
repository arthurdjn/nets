"""
The ``functional`` modules defines basic functions to generate tensors and transform data.
"""

import numpy as np
import nets


def zeros(shape, *args, **kwargs):
    data = np.zeros(shape)
    return nets.Tensor(data, *args, **kwargs)


def zeros_like(t, *args, **kwargs):
    data = np.zeros_like(t.numpy())
    return nets.Tensor(data, *args, **kwargs)


def ones(shape, *args, **kwargs):
    data = np.ones(shape)
    return nets.Tensor(data, *args, **kwargs)


def ones_like(t, *args, **kwargs):
    data = np.ones_like(t.numpy())
    return nets.Tensor(data, *args, **kwargs)


def eye(size, *args, **kwargs):
    data = np.eye(size)
    return nets.Tensor(data, *args, **kwargs)


def identity(size, *args, **kwargs):
    data = np.identity(size)
    return nets.Tensor(data, *args, **kwargs)