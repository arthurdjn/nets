# File: ops.py
# Creation: Wednesday August 19th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


r"""
Defines basic operations between two tensors, like addition, subtraction, dot product etc.
"""

# Basic imports
from collections import OrderedDict
from abc import ABC, abstractmethod
from nets.cuda import numpy_or_cupy
import numpy as ops

# NETS package
import nets
from .hook import Hook


class Operation(ABC):
    def __init__(self, device='cpu'):
        super(Operation, self).__init__()
        self.device = device
        self.tensor_left = None
        self.tensor_right = None

    @abstractmethod
    def forward(self, tensor_left, tensor_right, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward_left(self, grad, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward_right(self, grad, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, tensor_left, tensor_right, *args, **kwargs):
        # Save the inputs in `cache`
        self.tensor_left =tensor_left
        self.tensor_right = tensor_right
        # Result of the operation
        out = self.forward(tensor_left, tensor_right, *args, **kwargs)
        # Save `hooks` for autograd
        if tensor_left.requires_grad:
            out.register_hook(Hook(tensor_left, self.backward_left))
        if tensor_right.requires_grad:
            out.register_hook(Hook(tensor_right, self.backward_right))
        return out

    def __repr__(self):
        return f'<Op: {self.__class__.__name__} on {self.device.upper()}>'


class Function(ABC):
    def __init__(self):
        super(Function, self).__init__()
        self.tensor = None

    @abstractmethod
    def forward(self, tensor, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, tensor, *args, **kwargs):
        self.tensor = tensor
        out = self.forward(tensor, *args, **kwargs)
        if tensor.requires_grad:
            out.register_hook(Hook(tensor, self.backward))
        return out

    def __repr__(self):
        return f'<Op: {self.__class__.__name__}>'


class Sum(Function):
    def __init__(self):
        super(Sum, self).__init__()
        self.tensor = None

    def forward(self, tensor, axis=None, keepdims=False):
        self.tensor = tensor
        data = tensor.data.sum(axis=axis, keepdims=keepdims)
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad, axis=None, keepdims=False):
        r"""Update the gradient for the sum operation.

        Shape:
            - inputs (numpy.ndarray): upstream gradient
            - outputs (numpy.ndarray): gradient with shape the same shape as inputs data :math:`N`.
        """
        # We need to keep the information on which axis the sum was made (to be broadcasting compatible)
        # We always reshape the gradient in the same axis for back-propagation
        data_keepdims = self.tensor.sum(axis=axis, keepdims=True)
        grad = grad.reshape(data_keepdims.shape) + nets.zeros_like(self.tensor)
        return grad


class Add(Operation):

    def forward(self, tensor_left, tensor_right):
        data = tensor_left.data + tensor_right.data
        requires_grad = tensor_left.requires_grad or tensor_right.requires_grad
        device = tensor_left.device
        return nets.Tensor(data, requires_grad=requires_grad, device=device)

    @staticmethod
    def backward(grad, tensor):
        r"""Update the gradient for the addition.

        Shape:
            - inputs (Tensor): upstream gradient with shape the same shape as inputs data :math:`N`.
            - outputs (Tensor): downstream gradient with shape the same shape as inputs data :math:`N`.
        """
        # Sum out added dims
        ndims_added = grad.ndim - tensor.data.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        # Sum across broadcasted (but non-added dims)
        for i, dim in enumerate(tensor.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
        
    def backward_left(self, grad):
        grad = self.backward(grad, self.tensor_left)
        return grad

    def backward_right(self, grad):
        grad = self.backward(grad, self.tensor_right)
        return grad


class Reshape(Function):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, tensor):
        data = tensor.data.reshape(self.shape)
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        return grad.reshape(self.tensor.shape)


def reshape(tensor, shape):
    r"""Reshape a ``Tensor``.

    Args:
        t (Tensor): tensor to transform
        shape (tuple): new shape of ``t``

    Returns:
        Tensor
    """
    func = Reshape(shape)
    out = func(tensor)
    return out


def sum(tensor, axis=None, keepdims=False):
    r"""Compute the sum of all elements in a tensor, and update the gradients and hooks.

    .. math::
        \text{sum}(T) = \sum_{idx} t_{idx}

    Args:
        t (Tensor): tensor to get the sum from
        axis (int): axis to sum
        keepdims (bool): keep the same dimension in the resulting ``Tensor`` as the input if set to ``True``.
            Default is ``False``.

    Returns:
        Tensor: summed tensor
    """
    op_sum = Sum()
    out = op_sum(tensor, axis=axis, keepdims=keepdims)
    return out


def add(tensor1, tensor2):
    r"""Add two tensor-like together.

    .. math::
        T_{out} = T_1 + T_2

    Args:
        t1 (Tensor like): tensor to add
        t2 (Tensor like): second tensor to add with

    Returns:
        Tensor: the sum of two Tensor-like object
    """
    op_add = Add()
    out = op_add(tensor1, tensor2)
    return out


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

    return nets.Tensor(data, requires_grad=requires_grad, hooks=hooks)


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
    data = ops.multiply(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    hooks = []

    if t1.requires_grad:
        def grad_fn1(grad):
            r"""Update the gradient from t1 for the the multiplication operation, :math:`grad = grad \times T_2`.

            Shape:
                - inputs (numpy.ndarray): upstream gradient with shape the same shape as inputs data :math:`T_1`.
                - outputs (numpy.ndarray): downstream gradient with shape the same shape as inputs data :math:`T_1`.
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
                - inputs (numpy.ndarray): upstream gradient with shape the same shape as inputs data :math:`T_2`.
                - outputs (numpy.ndarray): downstream gradient with shape the same shape as inputs data :math:`T_2`.
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

    return nets.Tensor(data, requires_grad=requires_grad, hooks=hooks)


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
    data = 1 / t.data
    hooks = []

    if requires_grad:
        def grad_fn(grad):
            r"""Update the gradient for the inverse operation, :math:`grad = grad \times \frac{-1}{T^2}`.

            Shape:
                - inputs (numpy.ndarray): upstream gradient.
                - outputs (numpy.ndarray): downstream gradient.
            """
            return - 1 / (t.data ** 2) * grad

        hooks.append(Hook(t, grad_fn))

    return nets.Tensor(data, requires_grad=requires_grad, hooks=hooks)


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

    return nets.Tensor(data, requires_grad=requires_grad, hooks=hooks)


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
            bigger_grad = ops.zeros_like(t.data)
            if grad.shape != bigger_grad.shape:
                bigger_grad[indices] = grad
            else:
                bigger_grad = grad
            return bigger_grad

        hooks.append(Hook(t, grad_fn))

    return nets.Tensor(data, requires_grad=requires_grad, hooks=hooks)


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
