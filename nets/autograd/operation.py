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
from abc import ABC, abstractmethod
from nets.cuda import numpy_or_cupy, scalars_to_device

# NETS package
import nets
from .hook import Hook


class Operation(ABC):
    
    __slots__ = 'tensor1', 'tensor2'
    
    def __init__(self):
        super(Operation, self).__init__()
        self.tensor1 = None
        self.tensor2 = None

    @abstractmethod
    def forward(self, tensor1, tensor2):
        raise NotImplementedError

    @abstractmethod
    def backward1(self, grad):
        raise NotImplementedError

    @abstractmethod
    def backward2(self, grad):
        raise NotImplementedError

    def __call__(self, tensor1, tensor2):
        # Save the inputs in `cache`
        scalars_to_device(tensor1, tensor2)
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        # Result of the operation
        out = self.forward(tensor1, tensor2)

        # Save `hooks` for autograd
        if tensor1.requires_grad:
            out.register_hook(Hook(tensor1, self.backward1))
        if tensor2.requires_grad:
            out.register_hook(Hook(tensor2, self.backward2))

        return out

    def __repr__(self):
        return f'<Op: {self.__class__.__name__} on {self.device.upper()}>'


class Add(Operation):
    r"""Add two tensors together.

    .. math::
        \text{add}(t_1, t_2) = t_1 + t_2
    """

    def forward(self, tensor1, tensor2):
        data = tensor1.data + tensor2.data
        requires_grad = tensor1.requires_grad or tensor2.requires_grad
        device = tensor1.device
        return nets.Tensor(data, requires_grad=requires_grad, device=device)

    @staticmethod
    def backward(grad, tensor):
        # Sum out added dims
        ndims_added = grad.ndim - tensor.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        # Sum across broadcasted (but non-added dims)
        for i, dim in enumerate(tensor.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def backward1(self, grad):
        grad = self.backward(grad, self.tensor1)
        return grad

    def backward2(self, grad):
        grad = self.backward(grad, self.tensor2)
        return grad


class Multiply(Operation):
    r"""Elementwise multiplication of two tensors.

    .. math::
        \text{multiply}(t_1, t_2) = t_1 \times t_2
    """

    def forward(self, tensor1, tensor2):
        nc = numpy_or_cupy(tensor1, tensor2)
        data = nc.multiply(tensor1.data, tensor2.data)
        requires_grad = tensor1.requires_grad or tensor2.requires_grad
        device = tensor1.device
        return nets.Tensor(data, requires_grad=requires_grad, device=device)

    @staticmethod
    def backward(grad, t1, t2):
        grad = grad * t2
        ndims_added = grad.ndim - t1.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        # Sum across broadcasted (but non-added dims)
        for i, dim in enumerate(t1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
                
        return grad

    def backward1(self, grad):
        grad = self.backward(grad, self.tensor1, self.tensor2)
        return grad

    def backward2(self, grad):
        grad = self.backward(grad, self.tensor2, self.tensor1)
        return grad


class Dot(Operation):
    r"""Dot product of two matrices.

    .. math::
        \begin{*align}
            \text{dot}(t1, t2)  &= t_{out}
                                &= t_1 \dot t_2 \\
            \quad where \quad t_{i, j}^{[out]} = \sum_{k=1}^{n} t_{i, k}^{[1]} \times
            t_{k, j}^{[2]}
        \end{*align}
    """

    def forward(self, tensor1, tensor2):
        data = tensor1.data @ tensor2.data
        requires_grad = tensor1.requires_grad or tensor2.requires_grad
        device = tensor1.device
        return nets.Tensor(data, requires_grad=requires_grad, device=device)

    def backward1(self, grad):
        return grad @ self.tensor2.T

    def backward2(self, grad):
        return self.tensor1.T @ grad


class Where(Operation):
    r"""Transformation regarding a condition.

    .. math::
        \text{where}(t) =
            \begin{cases}
              t_1, &\quad if \quad condition \\
              t_2, &\quad else.
            \end{cases}

    .. note::
        The shape of the input tensors must be the same.

    * :attr:`cond` (bool): condition to merge two tensors.
    """

    __slots__ = 'tensor1', 'tensor2', 'cond'

    def __init__(self, cond):
        self.cond = cond

    @staticmethod
    def scalars_to_device(cond, tensor1, tensor2):
        if tensor1.shape == ():
            if tensor2.device != 'cpu' or cond.device != 'cpu':
                tensor1.cuda()
        if tensor2.shape == ():
            if tensor1.device != 'cpu' or cond.device != 'cpu':
                tensor2.cuda()

    def forward(self, tensor1, tensor2):
        scalars_to_device(self.cond, tensor1, tensor2)
        nc = numpy_or_cupy(tensor1, tensor2)
        data = nc.where(self.cond.data, tensor1.data, tensor2.data)
        requires_grad = tensor1.requires_grad or tensor2.requires_grad
        device = tensor1.device
        return nets.Tensor(data, requires_grad=requires_grad, device=device)

    def backward1(self, grad):
        return grad * nets.where(self.cond, 1, 0)

    def backward2(self, grad):
        return grad * nets.where(self.cond, 0, 1)
