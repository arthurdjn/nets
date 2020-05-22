"""
The ``functional`` modules defines basic functions to generate tensors and transform data.
"""

import numpy as np
import nets


def zeros(shape, *args, **kwargs):
    data = np.zeros(shape)
    return nets.Tensor(data, *args, **kwargs)


def zeros_like(t, *args, **kwargs):
    data = np.zeros_like(t.data)
    return nets.Tensor(data, t.requires_grad, *args, **kwargs)


def ones(shape, *args, **kwargs):
    data = np.ones(shape)
    return nets.Tensor(data, *args, **kwargs)


def ones_like(t, *args, **kwargs):
    data = np.ones_like(t.data)
    return nets.Tensor(data, t.requires_grad, *args, **kwargs)


def eye(size, *args, **kwargs):
    data = np.eye(size)
    return nets.Tensor(data, *args, **kwargs)


def identity(size, *args, **kwargs):
    data = np.identity(size)
    return nets.Tensor(data, *args, **kwargs)


def arange(*args, requires_grad=False):
    return nets.Tensor(np.arange(*args), requires_grad)


def astype(t, new_type):
    return nets.Tensor(t.data.astype(new_type))


# TODO: move to autograd
def concatenate(t1, t2):
    return nets.Tensor(np.concatenate(t1.data, t2.data))


# TODO: move to autograd
def append(t, value, axis=None):
    value = nets.to_array(value)
    data = np.append(t.data, value, axis=axis)
    return nets.Tensor(data)

# TODO: vstack, hstack etc. with autograd