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
import numpy as np
import copy

# NETS package
import nets
from nets.cuda import numpy_or_cupy, scalars_to_device
from .utils import inv_permutation
from .hook import Hook


class Function(ABC):
    r"""
    Defines a function with a compatible autograd system. The ``forward`` method is used to compute 
    the output of the function and the ``backward`` method is used to compute the gradient of a given tensor
    through the said function.

    The gradients are computed using the chain rule, i.e.

    .. math::


    * :attr:`tensor` (Tensor): input tensor used in the function.
    """
    
    __slots__ = 'tensors'

    def __init__(self):
        super(Function, self).__init__()
        self.tensors = None

    @abstractmethod
    def forward(self, *tensors):
        r"""Compute the output :math:`f(t)` of the function.

        Args:
            tensor (Tensor): input tensor :math:`t`.

        Returns:
            Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad):
        r"""Compute the gradient of the function :math:`f` at :math:`t`:

        .. math::
            f^‎\prime (t) = \frac{df}{dt}(t)

        Args:
            grad (Tensor): upstream gradient.

        Returns:
            Tensor
        """
        raise NotImplementedError

    def __call__(self, *tensors):
        # Cache the inputs
        self.tensors = (*tensors,)
        scalars_to_device(*self.tensors)
        # Compute a single pass
        out = self.forward(*tensors)
        for tensor in self.tensors:
            if tensor.requires_grad:
                out.register_hook(Hook(tensor, self.backward))
        return out


    def __repr__(self):
        return f'<Function: {self.__class__.__name__}>'


class Add(Function):
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
    def single_backward(grad, tensor):
        # Sum out added dims
        ndims_added = grad.ndim - tensor.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        # Sum across broadcasted (but non-added dims)
        for i, dim in enumerate(tensor.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def backward(self, grad):
        tensor1, tensor2 = self.tensors
        return (self.single_backward(grad, tensor1), 
                self.single_backward(grad, tensor2))


class Multiply(Function):
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
    def single_backward(grad, t1, t2):
        grad = grad * t2
        ndims_added = grad.ndim - t1.ndim
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        # Sum across broadcasted (but non-added dims)
        for i, dim in enumerate(t1.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
                
        return grad

    def backward(self, grad):
        tensor1, tensor2 = self.tensors
        return (self.single_backward(grad, tensor1, tensor2), 
                self.single_backward(grad, tensor2, tensor1))


class Dot(Function):
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

    def backward(self, grad):
        tensor1, tensor2 = self.tensors
        return (grad @ tensor2.T, 
                tensor1.T @ grad)


class Where(Function):
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

    def __init__(self, cond):
        self.cond = cond

    @staticmethod
    def _move_scalars_to_device(cond, tensor1, tensor2):
        if tensor1.shape == ():
            if tensor2.device != 'cpu' or cond.device != 'cpu':
                tensor1.cuda()
        if tensor2.shape == ():
            if tensor1.device != 'cpu' or cond.device != 'cpu':
                tensor2.cuda()

    def forward(self, tensor1, tensor2):
        self._move_scalars_to_device(self.cond, tensor1, tensor2)
        nc = numpy_or_cupy(tensor1, tensor2)
        data = nc.where(self.cond.data, tensor1.data, tensor2.data)
        requires_grad = tensor1.requires_grad or tensor2.requires_grad
        device = tensor1.device
        return nets.Tensor(data, requires_grad=requires_grad, device=device)

    def backward(self, grad):
        return (grad * nets.where(self.cond, 1, 0), 
                grad * nets.where(self.cond, 0, 1))


class Sum(Function):
    r"""Compute the sum of all elements in a tensor, and update the computational graph.

    .. math::
        \text{sum}(t) = \sum_{idx} t_{idx}
    """

    def __init__(self, axis=None, keepdims=False):
        super(Sum, self).__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, tensor):
        data = tensor.data.sum(axis=self.axis, keepdims=self.keepdims)
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        # We need to keep the information on which axis the sum was made (to be broadcasting compatible)
        # We always reshape the gradient in the same axis for back-propagation
        tensor, = self.tensors
        data_keepdims = tensor.sum(axis=self.axis, keepdims=True)
        grad = grad.reshape(data_keepdims.shape) + nets.zeros_like(tensor)
        return grad


class Transpose(Function):
    r"""Permutation a tensor object.

    * :attr:`indices` (tuple, optional): index to transpose.
    """

    def __init__(self, indices):
        self.indices = indices

    def forward(self, tensor):
        if self.indices is None:
            self.indices = tuple(range(tensor.ndim - 1, -1, -1))
        data = tensor.data.transpose(*self.indices)
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        indices_back = tuple(inv_permutation(self.indices))
        grad = grad.transpose(*indices_back)
        return grad


class Reshape(Function):
    r"""Change the shape of a tensor.

    * :attr:`shape`(tuple): new shape.
    """

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, tensor):
        data = tensor.data.reshape(*self.shape)
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        tensor, = self.tensors
        return grad.reshape(*tensor.shape)


class Pad(Function):

    # S(t, padding, constant_values=0):
    r"""Reshape a ``Tensor`` to a bigger size and add a ``padding`` on the side, with a ``0`` constant value.
    Args:
        t (Tensor): tensor to transform
        padding (tuple): padding dimensions
        constant_values (scalar, optional): scalar affected in the padding
    Returns:
        Tensor
    """

    def __init__(self, padding, constant_values=0):
        super(Pad, self).__init__()
        self.padding = padding
        self.constant_values = constant_values

    def forward(self, tensor):
        nc = numpy_or_cupy(tensor)
        data = nc.pad(tensor.data, pad_width=self.padding,
                      constant_values=self.constant_values)
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        return nets.unpad(grad, self.padding)


class Max(Function):
    r"""Get the maximum from a ``Tensor``.

    * :attr:`axis` (int): index of the axis to search.
    """

    def __init__(self, axis=None):
        super(Max, self).__init__()
        self.axis = axis

    def forward(self, tensor):
        nc = numpy_or_cupy(tensor)
        data = nc.max(tensor.data, axis=self.axis)
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        tensor, = self.tensors
        bigger_grad = nets.zeros_like(tensor)
        nc = numpy_or_cupy(grad)
        if self.axis is None:
            # If there is no axis, the argmax is the location of he maximum single element
            max_indices = nets.unravel_index(
                nets.argmax(tensor), tensor.shape)
            bigger_grad[max_indices] = grad
        else:
            # If there is an axis, we reconstruct the bigger matrix by 'rolling' on this axis
            max_indices = nets.argmax(tensor, axis=self.axis)
            for i, roll in enumerate(nets.rollaxis(bigger_grad, self.axis)):
                roll += (max_indices == i).astype(int) * grad

        return bigger_grad


class Neg(Function):
    r"""Oppose the values of a tensor.

    .. math::
        T_{out} = - T
    """

    def __init__(self):
        super(Neg, self).__init__()

    def forward(self, tensor):
        data = -tensor.data
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        return -grad


class Inverse(Function):
    r"""Inverse all elements of a tensor.

    .. math::
        \text{inverse}(t) = \frac{1}{t}
    """

    def forward(self, tensor):
        data = 1 / tensor.data
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad, *args, **kwargs):
        tensor, = self.tensors
        return - 1 / (tensor ** 2) * grad


class Slice(Function):
    r"""Slice a tensor from given indices.

    * :attr:`indices` (tuple, int, :): indices to extract data.
    """

    def __init__(self, indices):
        super(Slice, self).__init__()
        self.indices = indices

    def forward(self, tensor):
        data = tensor.data[self.indices]
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        tensor, = self.tensors
        bigger_grad = nets.zeros_like(tensor)
        if grad.shape != bigger_grad.shape:
            bigger_grad[self.indices] = grad
        else:
            bigger_grad = grad
        return bigger_grad


class Pow(Function):
    r"""Power a tensor-like object.

    .. math::
        \text{pow}(t) = t^2

    * :attr:`power` (int): power to elevate a tensor.
    """

    def __init__(self, power):
        super(Pow, self).__init__()
        self.power = power

    def forward(self, tensor):
        data = tensor.data ** self.power
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        tensor, = self.tensors
        return grad * self.power * tensor ** (self.power - 1)


class Sqrt(Function):
    r"""Square root of a tensor-like object.

    .. math::
        \text{sqrt} = \sqrt{t}
    """

    def forward(self, tensor):
        nc = numpy_or_cupy(tensor)
        data = nc.sqrt(tensor.data)
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        tensor, = self.tensors
        return - 1 / (2 * nets.sqrt(tensor)) * grad


class Exp(Function):
    r"""Exponentiation of a tensor.

    .. math::
        \text{exp}(t) = e^{t}
    """

    def forward(self, tensor):
        nc = numpy_or_cupy(tensor)
        data = np.exp(tensor.data)
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        tensor, = self.tensors
        return grad * nets.exp(tensor)


class Log(Function):
    r"""Logarithm of a tensor.
    """

    def forward(self, tensor):
        nc = numpy_or_cupy(tensor)
        data = nc.log(tensor.data)
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        tensor, = self.tensors
        return grad * nets.div(1, tensor)


class Tanh(Function):
    r"""Hyperbolic tangent of a tensor.

    .. math::
        \text{tanh}(t) = \frac{e^{t} - e^{-t}}{e^{t} + e^{-t}}
    """

    def forward(self, tensor):
        nc = numpy_or_cupy(tensor)
        data = nc.tanh(tensor.data)
        return nets.Tensor(data, requires_grad=tensor.requires_grad, device=tensor.device)

    def backward(self, grad):
        tensor, = self.tensors
        return grad * (1 - nets.tanh(tensor) ** 2)
