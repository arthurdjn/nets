"""
This modules defines basic transformations on a ``Tensor`` like ``transpose`` or ``reshape``.
"""

# Basic imports
import numpy as ops
import logging
try:
    import cupy as cp
except Exception as error:
    logging.error(f"CuPy not imported. {error}")

# NETS Package
import nets
from .hook import Hook
from ._utils import numpy_unpad, inv_permutation


def _T(t):
    r"""Transpose a tensor object.

    .. math::

        T_{out} = (t_{i, j}^{[out]})_{i, j} \quad where \quad t_{i, j}^{[out]} = t_{j ,i}

    Args:
        t (Tensor): tensor to transpose

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    data = t.data.T
    requires_grad = t.requires_grad
    hooks = []
    if requires_grad:
        hooks.append(Hook(t, lambda grad: grad.T))

    return nets.Tensor(data, requires_grad=requires_grad, hooks=hooks)


def transpose(t, indices=None):
    r"""Permutation a tensor object.

    Args:
        t (Tensor):
        indices (tuple, optional): index to transpose.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    if indices is None:
        indices = tuple(range(t.ndim - 1, -1, -1))
    data = t.data.transpose(indices)
    requires_grad = t.requires_grad
    hooks = []
    if requires_grad:
        def grad_fn(grad):
            indices_inv = tuple(inv_permutation(indices))
            grad = grad.transpose(indices_inv)
            return grad

        hooks.append(Hook(t, grad_fn))

    return nets.Tensor(data, requires_grad=requires_grad, hooks=hooks)


def pad(t, padding, constant_values=0):
    r"""Reshape a ``Tensor`` to a bigger size and add a ``padding`` on the side, with a ``0`` constant value.

    Args:
        t (Tensor): tensor to transform
        padding (tuple): padding dimensions
        constant_values (scalar, optional): scalar affected in the padding

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    data = ops.pad(t.data, pad_width=padding, constant_values=constant_values)
    requires_grad = t.requires_grad
    hooks = []
    if requires_grad:
        hooks.append(Hook(t, lambda grad: numpy_unpad(grad, padding)))

    return nets.Tensor(data, requires_grad=requires_grad, hooks=hooks)


def max(t, axis=None):
    r"""Get the maximum from a ``Tensor``.

    Args:
        t (Tensor): tensor to transform
        axis (int, optional): index of the axis to search. Default is ``None``.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    data = ops.max(t.data, axis=axis)
    requires_grad = t.requires_grad
    hooks = []
    if requires_grad:
        def grad_fn(grad):
            bigger_grad = ops.zeros_like(t.data)
            if axis is None:
                # If there is no axis, the argmax is the location of he maximum single element
                max_indices = ops.unravel_index(ops.argmax(t.data), t.shape)
                bigger_grad[max_indices] = grad
            else:
                # If there is an axis, we reconstruct the bigger matrix by 'rolling' on this axis
                max_indices = ops.argmax(t.data, axis=axis)
                for i, roll in enumerate(ops.rollaxis(bigger_grad, axis)):
                    roll += (max_indices == i).astype(int) * grad

            return bigger_grad

        hooks.append(Hook(t, grad_fn))

    return nets.Tensor(data, requires_grad=requires_grad, hooks=hooks)


def argmax(t, axis=None):
    r"""Get the indices of maximum elements from a ``Tensor``.

    Args:
        t (Tensor): tensor get maximum indices from
        axis (int, optional): index of the axis. Default is ``None``.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    if axis is None:
        return nets.Tensor(ops.unravel_index(ops.argmax(t.data), t.shape))
    else:
        return nets.Tensor(ops.argmax(t.data, axis=axis))


def flatten(t):
    r"""Reshape in 1-dimensional a ``Tensor``.

    Args:
        t (Tensor): tensor get reshape.

    Returns:
        Tensor
    """
    return reshape(t, (t.size,))


ITERABLE = (list, tuple)


def concatenate(iterable):
    r"""Concatenate multiples ``Tensor`` from an iterable.

    .. note::

        The ``Tensor`` in ``iterable`` should and must have the same shape.

    Args:
        iterable (tuple, list): list containing ``Tensor`` to concatenate.

    Returns:
        Tensor: the concatenation of all ``Tensor``.
    """
    assert isinstance(iterable, ITERABLE), f'iterable type {type(iterable)} unsupported for `concatenate` function.' \
                                           f'Types currently supported are list, tuple.'
    requires_grad = False
    hooks = []
    data = ops.array([])
    for idx, t in enumerate(iterable):
        t = nets.to_tensor(t)
        requires_grad = t.requires_grad or requires_grad
        if data.size == 0:
            data = t.data
        else:
            data = ops.concatenate((data, t.data))
        if t.requires_grad:
            def grad_fn(grad):
                return grad[idx:idx+t.shape[0]]

            hooks.append(Hook(t, grad_fn))
    return nets.Tensor(data, requires_grad=requires_grad, hooks=hooks)


def append(t, value):
    r"""Append multiples ``Tensor`` from an iterable.

    .. note::

        The ``Tensor`` in ``iterable`` should and must have the same shape.

    Args:
        t (Tensor): list containing ``Tensor`` to concatenate.

    Returns:
        Tensor: the concatenation of all ``Tensor``.
    """
    t = nets.to_tensor(t)
    value = nets.to_tensor(value)
    requires_grad = False
    hooks = []
    requires_grad = t.requires_grad or value.requires_grad
    if t.size == 0:
        data = [value.data]
    elif value.size == 0:
        data = [t.data]
    else:
        data = t.data.tolist()
        data.append(value.data)

    if t.requires_grad:
        def grad_fn(grad):
            return grad[:-1]
        hooks.append(Hook(t, grad_fn))

    if value.requires_grad:
        def grad_fn(grad):
            return grad[-1]
        hooks.append(Hook(value, grad_fn))

    return nets.Tensor(data, requires_grad=requires_grad, hooks=hooks)
