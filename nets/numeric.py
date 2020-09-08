"""
The ``functional`` modules defines basic functions to generate tensors and transform data.
"""

# Basic imports
import logging
import numpy as np
try:
    import cupy as cp
except Exception as error:
    logging.error(f"Could not import module cupy. {error}")

# NETS Package
import nets


def zeros(shape, device='cpu', **kwargs):
    if device == 'cpu':
        data = np.zeros(shape)
    else:
        data = cp.zeros(shape)
    return nets.Tensor(data, device=device, **kwargs)


def zeros_like(tensor):
    if tensor.device == 'cpu':
        data = np.zeros_like(tensor.data)
    else:
        data = cp.zeros_like(tensor.data)
    return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)


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
