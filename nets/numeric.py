"""
The ``functional`` modules defines basic functions to generate tensors and transform data.
"""

# Basic imports
import numpy as ops

# NETS Package
import nets


def zeros(shape, *args, **kwargs):
    data = ops.zeros(shape)
    return nets.Tensor(data, *args, **kwargs)


def zeros_like(t, *args, **kwargs):
    data = ops.zeros_like(t.data)
    return nets.Tensor(data, t.requires_grad, *args, **kwargs)


def ones(shape, *args, **kwargs):
    data = ops.ones(shape)
    return nets.Tensor(data, *args, **kwargs)


def ones_like(t, *args, **kwargs):
    data = ops.ones_like(t.data)
    return nets.Tensor(data, t.requires_grad, *args, **kwargs)


def eye(size, *args, **kwargs):
    data = ops.eye(size)
    return nets.Tensor(data, *args, **kwargs)


def identity(size, *args, **kwargs):
    data = ops.identity(size)
    return nets.Tensor(data, *args, **kwargs)


def arange(*args, requires_grad=False):
    return nets.Tensor(ops.arange(*args), requires_grad)


def astype(t, new_type):
    return nets.Tensor(t.data.astype(new_type))


# TODO: move to autograd
def concatenate(t1, t2):
    return nets.Tensor(ops.concatenate(t1.data, t2.data))


# TODO: move to autograd
def append(t, value, axis=None):
    value = nets.to_array(value)
    data = ops.append(t.data, value, axis=axis)
    return nets.Tensor(data)

# TODO: vstack, hstack etc. with autograd
