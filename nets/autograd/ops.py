r"""
Defines basic operations between two tensors, like addition, subtraction, dot product etc.
"""

import numpy as np
import nets
from .hook import Hook


def sum(t, axis=None, keepdims=False):
    r"""Compute the sum of all elements in a tensor, and update the gradients and hooks.

    .. math::

        \text{sum} = \sum_{idx} t_{idx}

    Args:
        t (Tensor): tensor to get the sum from
        axis (int): axis to sum
        keepdims (bool): keep the same dimension in the resulting ``Tensor`` as the input if set to ``True``.
            Default is ``False``.

    Returns:
        Tensor: summed tensor
    """
    data = t.data.sum(axis=axis, keepdims=keepdims)
    requires_grad = t.requires_grad
    hooks = []
    # Update the hooks and gradients
    if requires_grad:
        def grad_fn(grad):
            r"""Update the gradient for the sum operation.

            Shape:
                - inputs (np.ndarray): upstream gradient
                - outputs (np.ndarray): gradient with shape the same shape as inputs data :math:`T`.
            """
            # We need to keep the information on which axis the sum was made (to be broadcasting compatible)
            # We always reshape the gradient in the same axis for back-propagation
            data_keepdims = t.data.sum(axis=axis, keepdims=True)
            return grad.reshape(data_keepdims.shape) + np.zeros_like(t.data)

        hooks.append(Hook(t, grad_fn))

    return nets.Tensor(data, requires_grad, hooks)


def add(t1, t2):
    r"""Add two tensor-like together.

    .. math::

        T_{out} = T_1 + T_2

    Args:
        t1 (Tensor like): tensor to add
        t2 (Tensor like): second tensor to add with

    Returns:
        Tensor: the sum of two Tensor-like object
    """
    t1 = nets.to_tensor(t1)
    t2 = nets.to_tensor(t2)
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    hooks = []
    # Update the hooks and gradients from t1
    if t1.requires_grad:
        def grad_fn1(grad):
            r"""Update the gradient for the addition.

            Shape:
                - inputs (np.ndarray): upstream gradient with shape the same shape as inputs data :math:`T`.
                - outputs (np.ndarray): downstream gradient with shape the same shape as inputs data :math:`T`.
            """
            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        hooks.append(Hook(t1, grad_fn1))

    # Update the hooks and gradients from t2
    if t2.requires_grad:
        def grad_fn2(grad):
            r"""Update the gradient for the addition.

            Shape:
                - inputs (np.ndarray): upstream gradient with shape the same shape as inputs data :math:`T`.
                - outputs (np.ndarray): downstream gradient with shape the same shape as inputs data :math:`T`.
            """
            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        hooks.append(Hook(t2, grad_fn2))

    return nets.Tensor(data, requires_grad, hooks)


def neg(t):
    r"""Oppose the values of a tensor.

    .. math::

        T_{out} = - T

    Args:
        t (Tensor): tensor  to oppose.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    data = -t.data
    requires_grad = t.requires_grad
    hooks = []

    if requires_grad:
        hooks.append(Hook(t, lambda grad: -grad))

    return nets.Tensor(data, requires_grad, hooks)


def sub(t1, t2):
    r"""Subtract two tensor-like object

    .. math::

        T_{out} = T_1 - T_2

    Args:
        t1 (Tensor like): tensor to subtract
        t2 (Tensor like): second tensor to subtract with

    Returns:
        Tensor
    """
    return add(t1, neg(t2))


def multiply(t1, t2):
    r"""Elementwise multiplication of two tensors-like object.

    .. math::

        T_{out} = T_1 \times T_2

    Args:
        t1 (Tensor like): tensor to multiply
        t2 (Tensor like): second tensor to multiply with

    Returns:
        Tensor
    """
    t1 = nets.to_tensor(t1)
    t2 = nets.to_tensor(t2)
    data = np.multiply(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    hooks = []

    if t1.requires_grad:
        def grad_fn1(grad):
            r"""Update the gradient from t1 for the the multiplication operation, :math:`grad = grad \times T_2`.

            Shape:
                - inputs (np.ndarray): upstream gradient with shape the same shape as inputs data :math:`T_1`.
                - outputs (np.ndarray): downstream gradient with shape the same shape as inputs data :math:`T_1`.
            """
            grad = grad * t2.data
            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        hooks.append(Hook(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad):
            r"""Update the gradient from t2 for the the multiplication operation, :math:`grad = grad \times T_1`.

            Shape:
                - inputs (np.ndarray): upstream gradient with shape the same shape as inputs data :math:`T_2`.
                - outputs (np.ndarray): downstream gradient with shape the same shape as inputs data :math:`T_2`.
            """
            grad = grad * t1.data
            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        hooks.append(Hook(t2, grad_fn2))

    return nets.Tensor(data, requires_grad, hooks)


def inverse(t):
    r"""Inverse a tensor-like object.

    .. math::

        T_{out} = \frac{1}{T}

    Args:
        t (Tensor like): tensor to inverse.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    requires_grad = t.requires_grad
    hooks = []

    if requires_grad:
        def grad_fn(grad):
            r"""Update the gradient for the inverse operation, :math:`grad = grad \times \frac{-1}{T^2}`.

            Shape:
                - inputs (np.ndarray): upstream gradient.
                - outputs (np.ndarray): downstream gradient.
            """
            return - 1 / (t.data ** 2) * grad

        hooks.append(Hook(t, grad_fn))

    return nets.Tensor(1 / t.data, requires_grad, hooks)


def div(t1, t2):
    r"""Divide two tensor-like object.

    .. math::

        T_{out} = T_1 \times \frac{1}{T_2}

    Args:
        t1 (Tensor like): tensor to multiply
        t2 (Tensor like): tensor to invert

    Returns:
        Tensor
    """
    t1 = nets.to_tensor(t1)
    t2 = nets.to_tensor(t2)
    return multiply(t1, inverse(t2))


def dot(t1, t2):
    r"""Dot product of two matrices.
    
    .. math::
    
        T_{out} = (t_{i, j}^{[out]})_{i, j} \quad where \quad t_{i, j}^{[out]} = \sum_{k=1}^{n} t_{i, k}^{[1]} \times
        t_{k, j}^{[2]}

    Args:
        t1 (Tensor like)
        t2 (Tensor like)

    Returns:
        Tensor
    """
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    hooks = []

    if t1.requires_grad:
        def grad_fn1(grad):
            return grad @ t2.data.T

        hooks.append(Hook(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad):
            return t1.data.T @ grad

        hooks.append(Hook(t2, grad_fn2))

    return nets.Tensor(data, requires_grad, hooks)


def slice(t, indices):
    r"""Slice a tensor from given indices.

    Args:
        t (Tensor): tensor to slice
        idxs (tuple, int, :): indices to extract data

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    if isinstance(indices, nets.Tensor):
        indices = indices.data
    data = t.data[indices]
    requires_grad = t.requires_grad
    hooks = []
    if requires_grad:
        def grad_fn(grad):
            bigger_grad = np.zeros_like(t.data)
            if grad.shape != bigger_grad.shape:
                bigger_grad[indices] = grad
            else:
                bigger_grad = grad
            return bigger_grad
        hooks.append(Hook(t, grad_fn))

    return nets.Tensor(data, requires_grad, hooks)


def gt(t, other):
    r"""Return a boolean tensor for *greater than* condition.

    .. math::

        condition = T > other

    Args:
        t (Tensor): tensor to compare
        other (Tensor like): object to compare the tensor

    Returns:
        Tensor
    """
    t = nets.to_array(t)
    other = nets.to_array(other)
    cond = t > other
    return nets.to_tensor(cond)


def set(t, key, value):
    if isinstance(key, nets.Tensor):
        key = key.data
    elif isinstance(key, tuple):
        keys = []
        for k in key:
            if isinstance(k, nets.Tensor):
                keys.append(k.data)
            else:
                keys.append(k)
        key = tuple(keys)
    t.data[key] = nets.to_tensor(value).data

    # Setting a tensor invalidate its gradient
    t.detach()

    return t


def ge(t, other):
    r"""Return a boolean tensor for *greater or equal* condition.

    .. math::

        condition = T \ge other

    Args:
        t (Tensor): tensor to compare
        other (Tensor like): object to compare the tensor

    Returns:
        Tensor
    """
    t = nets.to_array(t)
    other = nets.to_array(other)
    cond = t >= other
    return nets.to_tensor(cond)


def lt(t, other):
    r"""Return a boolean tensor for *lower than* condition.

    .. math::

        condition = T < other

    Args:
        t (Tensor): tensor to compare
        other (Tensor like): object to compare the tensor

    Returns:
        Tensor
    """
    t = nets.to_array(t)
    other = nets.to_array(other)
    cond = t < other
    return nets.to_tensor(cond)


def le(t, other):
    r"""Return a boolean tensor for *lower or equal* condition.

    .. math::

        condition = T \le other

    Args:
        t (Tensor): tensor to compare
        other (Tensor like): object to compare the tensor

    Returns:
        Tensor
    """
    t = nets.to_array(t)
    other = nets.to_array(other)
    cond = t <= other
    return nets.to_tensor(cond)


def eq(t, other):
    r"""Return a boolean tensor for *equal* condition.

    .. math::

        condition = T == other

    Args:
        t (Tensor): tensor to compare
        other (Tensor like): object to compare the tensor

    Returns:
        Tensor
    """
    t = nets.to_array(t)
    other = nets.to_array(other)
    cond = t == other
    return nets.to_tensor(cond)


def ne(t, other):
    r"""Return a boolean tensor for *not equal* condition.

    .. math::

        condition = T not other

    Args:
        t (Tensor): tensor to compare
        other (Tensor like): object to compare the tensor

    Returns:
        Tensor
    """
    t = nets.to_array(t)
    other = nets.to_array(other)
    cond = not t == other
    return nets.to_tensor(cond)
