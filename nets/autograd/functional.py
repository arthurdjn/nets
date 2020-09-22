# File: functions.py
# Creation: Tuesday September 8th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin

# Basic imports
import numpy as np

# NETS package
from nets.cuda import numpy_or_cupy
import nets
import nets.autograd.function as fc
import nets.autograd.operation as op


def transpose(t, indices=None):
    r"""Transpose a tensor regarding indices.

    Args:
        t (Tensor like): tensor to transpose.
        indices (iterable): indices to perform the permutation. Default to ``None``.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    func = fc.Transpose(indices)
    return func(t)


def reshape(t, shape):
    r"""Reshape a tensor.

    Args:
        t (Tensor like): tensor to transform.
        shape (tuple): new shape of the input tensor.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    func = fc.Reshape(shape)
    return func(t)


def flatten(t):
    r"""Reshape in a 1-dimensional a tensor.

    Args:
        t (Tensor): tensor get reshape.

    Returns:
        Tensor
    """
    return reshape(t, (t.size,))


def pad(t, padding):
    r"""pad a tensor.

    Args:
        t (Tensor like): tensor to transform.
        padding (tuple): padding size of the tensor.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    func = fc.Pad(padding)
    return func(t)


def unpad(t, padding):
    """Unpad a tensor.

    Args:
        t (Tensor): array to unpad.
        padding (tuple): padding size of the tensor.

    Returns:
        numpy.ndarray
    """
    slices = []
    for pad in padding:
        e = None if pad[1] == 0 else -pad[1]
        slices.append(slice(pad[0], e))
    return t[tuple(slices)]


def max(t, axis=None):
    r"""pad a tensor.

    Args:
        t (Tensor like): tensor to transform.
        axis (tuple): index of the axis to search. Default is ``None``.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    func = fc.Max(axis)
    return func(t)


def argmax(t, axis=None):
    r"""Get the indices of maximum elements from a tensor.

    Args:
        t (Tensor): tensor get maximum indices from
        axis (int, optional): index of the axis. Default is ``None``.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    nc = numpy_or_cupy(t)
    if axis is None:
        data = nc.unravel_index(nc.argmax(t.data), t.shape)
    else:
        data = nc.argmax(t.data, axis=axis)
    return nets.Tensor(data, device=t.device)


def neg(t):
    r"""Oppose the values of a tensor.

    .. math::
        \text{neg}(t) = -t

    Args:
        tensor (Tensor like): tensor to oppose.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    func = fc.Neg()
    return func(t)


def sum(t, axis=None, keepdims=False):
    r"""Compute the sum of all elements in a tensor, and update the gradients and hooks.

    .. math::
        \text{sum}(t) = \sum_{idx} t_{idx}

    Args:
        tensor (Tensor like): tensor to get the sum from.
        axis (int): axis to sum.
        keepdims (bool): keep the same dimension in the resulting ``Tensor`` as the input if set to ``True``.
            Default is ``False``.

    Returns:
        Tensor: summed tensor
    """
    t = nets.to_tensor(t)
    func = fc.Sum(axis=axis, keepdims=keepdims)
    return func(t)


def add(t1, t2):
    r"""Add two tensor-like together.

    .. math::
        \text{add}(t_1, t_2) = t_1 + t_2

    Args:
        t1 (Tensor like): tensor to add.
        t2 (Tensor like): second tensor to add with.

    Returns:
        Tensor: the sum of two Tensor-like object.
    """
    t1 = nets.to_tensor(t1)
    t2 = nets.to_tensor(t2)
    operation = op.Add()
    return operation(t1, t2)


def sub(t1, t2):
    r"""Subtract two tensor-like object

    .. math::
        \text{sub}(t_1, t_2) = t_1 - t_2

    Args:
        t1 (Tensor like): tensor to subtract.
        t2 (Tensor like): second tensor to subtract with.

    Returns:
        Tensor
    """
    t1 = nets.to_tensor(t1)
    t2 = nets.to_tensor(t2)
    return add(t1, neg(t2))


def multiply(t1, t2):
    r"""Elementwise multiplication of two tensors.

    .. math::
        \text{multiply}(t_1, t_2) = t_1 \times t_2

    Args:
        t1 (Tensor like): tensor to multiply.
        t2 (Tensor like): second tensor to multiply with.

    Returns:
        Tensor: the elementwise multiplication.
    """
    t1 = nets.to_tensor(t1)
    t2 = nets.to_tensor(t2)
    operation = op.Multiply()
    return operation(t1, t2)


def inverse(t):
    r"""Inverse a tensor-like object.

    .. math::
        \text{inverse}(t) = \frac{1}{t}

    Args:
        t (Tensor like): tensor to inverse.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    func = fc.Inverse()
    return func(t)


def div(t1, t2):
    r"""Divide all elements of two tensors.

    .. math::
        \text{div} = t_1 \times \frac{1}{t_2}

    .. note::
        The two input tensors must have the same shape.

    Args:
        t1 (Tensor like): tensor to multiply.
        t2 (Tensor like): tensor to invert.

    Returns:
        Tensor
    """
    t1 = nets.to_tensor(t1)
    t2 = nets.to_tensor(t2)
    return multiply(t1, inverse(t2))


def dot(t1, t2):
    r"""Dot product of two matrices.

    .. math::
        \begin{*align}
            \text{dot}(t1, t2)  &= t_{out}
                                &= t_1 \dot t_2 \\
            \quad where \quad t_{i, j}^{[out]} = \sum_{k=1}^{n} t_{i, k}^{[1]} \times
            t_{k, j}^{[2]}
        \end{*align}

    Args:
        t1 (Tensor like): tensor to multiply.
        t2 (Tensor like): second tensor to multiply with.

    Returns:
        Tensor: the elementwise multiplication.
    """
    t1 = nets.to_tensor(t1)
    t2 = nets.to_tensor(t2)
    operation = op.Dot()
    return operation(t1, t2)


def slice(t, indices):
    r"""Slice a tensor from given indices.

    Args:
        t (Tensor): tensor to slice.
        indices (tuple, int, :): indices to extract data.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    func = fc.Slice(indices)
    return func(t)


def where(cond, t1, t2):
    r"""Transformation regarding a condition.

    .. math::
        \text{where}(t) =
            \begin{cases}
              t_1, &\quad if \quad condition \\
              t_2, &\quad else.
            \end{cases}

    Args:
        cond (bool): condition to merge two tensors.
        t1 (Tensor): input tensor to compare.
        t2 (Tensor): input tensor to compare.

    Returns:
        Tensor
    """
    t1 = nets.to_tensor(t1)
    t2 = nets.to_tensor(t2)
    operation = op.Where(cond)
    return operation(t1, t2)


def maximum(t1, t2):
    r"""Get the maximum between two tensors.

    Args:
        t1 (Tensor): input tensor to compare.
        t2 (Tensor): input tensor to compare.

    Returns:
        Tensor
    """
    t1 = nets.to_tensor(t1)
    t2 = nets.to_tensor(t2)
    return where(t1 > t2, t1, t2)


def minimum(t1, t2):
    r"""Get the minimum between two tensors.

    Args:
        t1 (Tensor): input tensor to compare.
        t2 (Tensor): input tensor to compare.

    Returns:
        Tensor
    """
    return where(t1 > t2, t2, t1)


def pow(t, power):
    r"""Power a tensor-like object.

    .. math::
        \text{pow}(t) = t^2

    Args:
        t (Tensor): input tensor.
        power (int): power.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    func = fc.Pow(power)
    return func(t)


def sqrt(t):
    r"""Square root of a tensor-like object.

    Args:
        t (Tensor): input tensor.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    func = fc.Sqrt()
    return func(t)


def exp(t):
    r"""Exponentiation of a tensor.

    .. math::
        \text{exp}(t) = e^{t}

    Args:
        t (Tensor): input tensor.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    func = fc.Exp()
    return func(t)


def log(t):
    r"""Logarithm of a tensor.

    Args:
        t (Tensor): input tensor.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    func = fc.Log()
    return func(t)


def softmax(t, axis=0):
    r"""Compute the softmax function using an implementation that prevent numerical instability.
    The :math:`i^{th}` element of a softmax function evaluated 
    on a vector :math:`x \in \mathbb{R}^n` is given by:

    .. math::
        \text{softmax}(x_{i}) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}

    Args:
        t (Tensor): input tensor.
        axis (int): axis to consider for the ``softmax``. Default is ``0``.

    See :class:`~nets.nn.activation.Softmax` for the activation implementation.
    """
    t = nets.to_tensor(t)
    s = sum(exp(t), axis=axis, keepdims=True)
    return exp(t - log(s))


def tanh(t):
    r"""Hyperbolic tangent of a tensor.

    .. math::
        \text{tanh}(t) = \frac{e^{t} - e^{-t}}{e^{t} + e^{-t}}

    Args:
        t (Tensor): input tensor.

    .. image:: /images/functional_tanh.png

    Example:
        >>> import nets
        >>> tensor = nets.Tensor([-5, 2, 6, -2, 4])
        >>> nets.tanh(tensor)

    See :class:`~nets.nn.activation.Tanh` for the activation implementation.
    """
    t = nets.to_tensor(t)
    func = fc.Tanh()
    return func(t)


def tanh_prime(t):
    r"""First order derivative of the hyperbolic tangent.

    .. math::
        \text{tanh'}(t) = 1 - \text{tanh}^2(t)

    Args:
        t (Tensor): input tensor.

    .. image:: /images/functional_tanh_prime.png

    Example:
        >>> import nets 
        >>> tensor = nets.Tensor([-5, 2, 6, -2, 4])
        >>> nets.tanh(tensor)

    See :class:`~nets.nn.activation.Tanh` for the activation implementation.
    """
    t = nets.to_tensor(t)
    return 1 - pow(tanh(t), 2)


def sigmoid(t):
    r"""Sigmoid function.

    .. math::
        \text{sigmoid}(t) = \frac{1}{1 + e^{-t}}

    .. warning::
        The current implementation does not prevent numerical instability.

    Args:
        t (Tensor): input tensor.

    .. image:: /images/functional_sigmoid.png

    Example:
        >>> import nets
        >>> tensor = nets.Tensor([-5, 2, 6, -2, 4])
        >>> nets.sigmoid(tensor)

    See :class:`~nets.nn.activation.Sigmoid` for the activation implementation.
    """
    t = nets.to_tensor(t)
    return 1.0 / (1.0 + exp(-t))


def sigmoid_prime(t):
    r"""First order derivative of ``sigmoid`` function, defined as:

    .. math::
        \text{sigmoid'}(x) = (1 - \text{sigmoid}(x))

    Args:
        t (Tensor): input tensor.

    .. image:: images/functional_sigmoid_prime.png

    Example:
        >>> import nets
        >>> tensor = nets.tensor([-5, 2, 6, -2, 4])
        >>> nets.sigmoid_prime(in_array)

    See :class:`~nets.nn.activation.Sigmoid` for the activation implementation.
    """
    t = nets.to_tensor(t)
    return sigmoid(t) * (1 - sigmoid(t))


def relu(t):
    r"""``relu`` is a standard activation function, defined as:

    .. math::
        \text{relu(t)} = \max{(0, t)}

    Args:
        t (Tensor): input tensor.

    .. image:: /images/functional_relu.png

    Example:
        >>> import nets
        >>> tensor = nets.tensor([-5, 2, 6, -2, 4])
        >>> tensor = relu(tensor)

    See :class:`~nets.nn.activation.ReLU` for the activation implementation.
    """
    t = nets.to_tensor(t)
    return maximum(nets.zeros_like(t), t)


def relu_prime(t):
    r"""First order derivative of the ``relu`` function.

    .. math::
        \text{relu'(t)} =
            \begin{cases}
                1, &\quad t \ge 0 \\
                0, &\quad t < 0.
            \end{cases}

    Args:
        t (Tensor): input tensor.

    .. image:: images/functional_relu_prime.png

    Example:
        >>> import nets
        >>> tensor = nets.tensor([-5, 2, 6, -2, 4])
        >>> relu_prime(tensor)

    See :class:`~nets.nn.activation.ReLU` for the activation implementation.
    """
    t = nets.to_tensor(t)
    return where(t >= 0, nets.ones_like(t), nets.zeros_like(t))


def leaky_relu(t, alpha=0.01):
    r"""Variation from the original ``relu`` function, 
    which can prevent 'dying' ``relu`` thanks to a slope ``alpha``.

    .. math::
        \text{leaky_relu(t)} = \max{(\alpha \times t, t)}

    Args:
        t (Tensor): input tensor.
        alpha (float, optional): slope towards :math:`-\infty`. 

    .. image:: /images/functional_leaky_relu.png

    Example:
        >>> import nets
        >>> tensor = nets.tensor([-5, 2, 6, -2, 4])
        >>> alpha = 0.1
        >>> leaky_relu(tensor, alpha)

    See :class:`~nets.nn.activation.LeakyReLU` for the activation implementation.
    """
    t = nets.to_tensor(t)
    return where(t > 0, t, t * alpha)


def leaky_relu_prime(t, alpha=0.01):
    r"""First order derivative of ``leaky_relu`` function.

    .. math::
        \text{leaky_relu'(x)} =
            \begin{cases}
                1, &\quad x \ge 0 \\
                \alpha, &\quad x < 0.
            \end{cases}

    Args:
        t (Tensor): input tensor.
        alpha (float, optional): slope towards :math:`-\infty`.

    .. image:: /images/functional_leaky_relu_prime.png

    Example:
        >>> import nets
        >>> tensor = nets.tensor([-5, 2, 6, -2, 4])
        >>> alpha = 0.1
        >>> leaky_relu_prime(tensor, alpha)

    See :class:`~nets.nn.activation.LeakyReLU` for the activation implementation.
    """
    t = nets.to_tensor(t)
    return where(t > 0, nets.ones_like(t), alpha * nets.ones_like(t))


ITERABLE = (list, tuple)


# TODO: use the ``Function`` class to wrap this function
def concatenate(iterable):
    r"""Concatenate multiples ``Tensor`` from an iterable.

    .. note::
        The ``Tensor`` in ``iterable`` should and must have the same shape.

    Args:
        iterable (tuple, list): list containing ``Tensor`` to concatenate.

    Returns:
        Tensor: the concatenation of all ``Tensor``.
    """
    assert isinstance(iterable, ITERABLE), (f'iterable type {type(iterable)} unsupported for `concatenate` function.'
                                            f'Types currently supported are list, tuple.')
    requires_grad = False
    hooks = []
    nc = numpy_or_cupy(*iterable)
    data = nc.array([])
    for idx, t in enumerate(iterable):
        t = nets.to_tensor(t)
        requires_grad = t.requires_grad or requires_grad
        if data.size == 0:
            data = t.data
        else:
            data = nc.concatenate((data, t.data))
        if t.requires_grad:
            def grad_fn(grad):
                return grad[idx:idx+t.shape[0]]
            hooks.append(nets.Hook(t, grad_fn))

    tensor = nets.Tensor(data, requires_grad, device=iterable[0].device)
    for hook in hooks:
        tensor.register_hook(hook)
    return tensor


# TODO: use the ``Function`` class to wrap this function
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
    requires_grad = t.requires_grad or value.requires_grad

    if t.size == 0:
        data = [value.data]
    elif value.size == 0:
        data = [t.data]
    else:
        data = t.data.tolist()
        data.append(value.data)
    tensor = nets.Tensor(data, requires_grad, device=t.device)

    if t.requires_grad:
        def grad_fn(grad):
            return grad[:-1]
        tensor.register_hook(nets.Hook(t, grad_fn))

    if value.requires_grad:
        def grad_fn(grad):
            return grad[-1]
        tensor.register_hook(nets.Hook(t, grad_fn))

    return tensor
