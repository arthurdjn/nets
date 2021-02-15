"""
This modules defines more complex operations, and defines the most popular functions such as:

- exp,
- log,
- tanh,
- sigmoid;
- relu...
"""

import numpy as np
import nets
from .hook import Hook


def where(cond, t1, t2):
    r"""Transformation regarding a condition.

    .. math::

        T_{out} =
                        \begin{cases}
                          T_1, &\quad if \quad condition \\
                          T_2, &\quad else.
                        \end{cases}

    Args:
        cond (bool): condition to merge two tensors
        t1 (Tensor): input tensor to merge
        t2 (Tensor): input tensor to merge

    Returns:
        Tensor
    """
    t1 = nets.to_tensor(t1)
    t2 = nets.to_tensor(t2)
    # TODO: handle broadcasting with where(): sum across the broadcast dimension
    assert t1.shape == t2.shape, f"tensors should have the same shape. Got t1.shape={t1.shape}, t2.shape={t2.shape}"

    cond = nets.to_array(cond)
    data = np.where(cond, t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    hooks = []
    if t1.requires_grad:
        def grad_fn(grad):
            return grad * np.where(cond, 1, 0)
        hooks.append(Hook(t1, grad_fn))
    if t2.requires_grad:
        def grad_fn(grad):
            return grad * np.where(cond, 0, 1)
        hooks.append(Hook(t2, grad_fn))

    return nets.Tensor(data, requires_grad, hooks)


def maximum(t1, t2):
    r"""Get the maximum between two tensors.

    Args:
        t1 (Tensor): input tensor to merge
        t2 (Tensor): input tensor to merge

    Returns:
        Tensor
    """
    return where(t1 > t2, t1, t2)


def minimum(t1, t2):
    r"""Get the minimum between two tensors.

    Args:
        t1 (Tensor): input tensor to merge
        t2 (Tensor): input tensor to merge

    Returns:
        Tensor
    """
    return where(t1 > t2, t2, t1)


def pow(t, power):
    r"""Power a tensor-like object.

    .. math::

        T_{out} = T^2

    Args:
        t (Tensor like): reference tensor
        power (int): power to elevate a tensor

    Returns:
        Tensor
    """
    assert type(power) == int, "unsupported type {} for power. Currently supported type: int".format(type(power))
    t = nets.to_tensor(t)
    data = t.data ** power
    requires_grad = t.requires_grad
    hooks = []
    # Update the gradient
    if requires_grad:
        hooks.append(Hook(t, lambda grad: grad * power * t.data ** (power - 1)))

    return nets.Tensor(data, requires_grad, hooks)


def sqrt(t):
    r"""Square root of a tensor-like object.

    .. math::

        T_{out} = T^2

    Args:
        t (Tensor like): reference tensor
        power (int): power to elevate a tensor

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    data = np.sqrt(t.data)
    requires_grad = t.requires_grad
    hooks = []
    # Update the gradient
    if requires_grad:
        hooks.append(Hook(t, lambda grad: - 1 / (2 * np.sqrt(t.data)) * grad))

    return nets.Tensor(data, requires_grad, hooks)


def exp(t):
    r"""Exponentiation f a tensor.

    .. math::

        T_{out} = \text{exp}(T)

    Args:
        t (Tensor): input tensor to transform

    Returns:
        Tensor
    """
    data = np.exp(t.data)
    requires_grad = t.requires_grad
    hooks = []
    # Update the gradient
    if requires_grad:
        hooks.append(Hook(t, lambda grad: grad * data))

    return nets.Tensor(data, requires_grad, hooks)


def log(t):
    r"""Compute the logarithm of a tensor object.

    .. math::

        T_{out} = \text{log}(T)

    Args:
        t (Tensor like): input tensor

    Returns:
        Tensor
    """
    data = np.log(t.data)
    requires_grad = t.requires_grad
    hooks = []
    # Update the gradient
    if requires_grad:
        hooks.append(Hook(t, lambda grad: grad * np.divide(1, t.data)))

    return nets.Tensor(data, requires_grad, hooks)


def softmax(x, axis=0):
    # type: (Array, Optional[int]) -> Tensor
    r"""``softmax`` function. The implementation chosen is not straight forward to prevent numerical instability.

    The :math:`i^{th}` element of a softmax function evaluated on a vector :math:`x \in \mathbb{R}^n` is given by:

    .. math::

        \text{softmax}(x_{i}) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}

    Args:
        axis (int): axis to consider for the ``softmax``. Default is ``0``.

    Shape:
        - input: x (numpy.array): input to compute the ``softmax`` function on.
        - output: y (numpy.array): hyper-plane of the input.

    See :class:`~nets.nn.activation.Softmax` for the activation implementation.
    """
    e = nets.exp(x)
    s = nets.sum(e, axis=axis, keepdims=True)
    t = x - nets.log(s)
    soft = nets.exp(t)
    return soft


def tanh(t):
    # type: (Tensor) -> Tensor
    r"""``tanh`` standard function, definead as:

    .. math::
        \text{tanh}(T) = \frac{e^{T} - e^{-T}}{e^{T} + e^{-T}}

    Shape:
        - input: x (numpy.array): input to compute the ``tanh`` function on.
        - output: y (numpy.array): ``tanh`` output, with the same shape than :math:`T`.

    .. image:: images/functional_tanh.png

    Examples::

        >>> import nets
        >>> in_array = np.array([-5, 2, 6, -2, 4])
        >>> out_array = nets.tanh(in_array)

    See :class:`~nets.nn.activation.Tanh` for the activation implementation.
    """
    data = np.tanh(t.data)
    requires_grad = t.requires_grad
    hooks = []
    if requires_grad:
        hooks.append(Hook(t, lambda grad: grad * (1 - data * data)))

    return nets.Tensor(data, requires_grad, hooks)


def tanh_prime(x):
    # type: (Array) -> Array
    r"""First order derivative of ``tanh`` function, defined as:

    .. math::
        \text{tanh'}(x) = 1 - \text{tanh}^2(x)

    Shape:
        - input: x (numpy.array): input to compute the ``tanh derivative`` function on.
        - output: y (numpy.array): gradient of the input, with the same shape than ``x``.

    .. image:: images/functional_tanh_prime.png

    Examples::

        >>> in_array = np.array([-5, 2, 6, -2, 4])
        >>> out_array = tanh(in_array)

    See :class:`~nets.nn.activation.Tanh` for the activation implementation.
    """
    return 1 - np.power(2, np.tanh(x))


def sigmoid(x):
    # type: (Array) -> Array
    r"""Vanilla ``sigmoid`` function, defined as:

    .. math::
        \text{sigmoid}(x) = \frac{1}{1 + e^{-x}}

    Shape:
        - input: x (numpy.array): input to compute the ``sigmoid`` function on.
        - output: y (numpy.array): ``sigmoid`` output, with the same shape than ``x``.

    .. image:: images/functional_sigmoid.png

    Examples::

        >>> in_array = np.array([-5, 2, 6, -2, 4])
        >>> out_array = sigmoid(in_array)

    See :class:`~nets.nn.activation.Sigmoid` for the activation implementation.
    """
    return 1.0 / (1.0 + nets.exp(-x))


def sigmoid_prime(x):
    # type: (Array) -> Array
    r"""First order derivative of ``sigmoid`` function, defined as:

    .. math::
        \text{sigmoid'}(x) = (1 - \text{sigmoid}(x))

    Shape:
        - input: x (numpy.array): input to compute the ``sigmoid derivative`` function on.
        - output: y (numpy.array): gradient of the input, with the same shape than ``x``.

    .. image:: images/functional_sigmoid_prime.png

    Examples::

        >>> in_array = np.array([-5, 2, 6, -2, 4])
        >>> out_array = sigmoid_prime(in_array)

    See :class:`~nets.nn.activation.Sigmoid` for the activation implementation.
    """
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    # type: (Array) -> Array
    r"""``relu`` is a standard activation function, defined as:

    .. math::
        \text{relu(x)} = \max{(0, x)}

    Shape:
        - input: x (numpy.array): input to compute the ``relu`` function on.
        - output: y (numpy.array): ``relu`` output, with the same shape than ``x``.

    .. image:: images/functional_relu.png

    Examples::

        >>> in_array = np.array([-5, 2, 6, -2, 4])
        >>> out_array = relu(in_array)

    See :class:`~nets.nn.activation.ReLU` for the activation implementation.
    """
    return maximum(nets.zeros_like(x), x)


def relu_prime(x):
    # type: (Array) -> Array
    r"""First order derivative of the ``relu`` function.

    .. math::
        \text{relu'(x)} =
                        \begin{cases}
                          1, &\quad x \ge 0 \\
                          0, &\quad x < 0.
                        \end{cases}

    Shape:
        - input: x (numpy.array): input to compute the ``leaky_relu`` function on.
        - output: y (numpy.array): gradient of the input, with the same shape than ``x``.

    .. image:: images/functional_relu_prime.png

    Examples::

        >>> in_array = np.array([-5, 2, 6, -2, 4])
        >>> out_array = relu_prime(in_array)

    See :class:`~nets.nn.activation.ReLU` for the activation implementation.
    """
    return where(x >= 0, nets.ones_like(x), nets.zeros_like(x))


def leaky_relu(x, alpha=0.01):
    # type: (Array, Optional[float]) -> Array
    r"""Variation from the original ``relu`` function, which can prevent 'dying' ``relu`` thanks to a slope ``alpha``.

    .. math::
        \text{leaky_relu(x)} = \max{(\alpha \times x, x)}

    Args:
        alpha (float, optional): slope towards :math:`-\infty`.

    Shape:
        - input: x (numpy.array): input to compute the ``leaky_relu`` function on.
        - output: y (numpy.array): ``leaky relu`` output, with the same shape than ``x``.

    .. image:: images/functional_leaky_relu.png

    Examples::

        >>> in_array = np.array([-5, 2, 6, -2, 4])
        >>> alpha = 0.1
        >>> out_array = leaky_relu(in_array, alpha)

    See :class:`~nets.nn.activation.LeakyReLU` for the activation implementation.
    """
    return where(x > 0, x, x * alpha)


def leaky_relu_prime(x, alpha=0.01):
    # type: (Array, Optional[float]) -> Array
    r"""First order derivative of ``leaky_relu`` function, defined as:

    .. math::
        \text{leaky_relu'(x)} =
                        \begin{cases}
                          1, &\quad x \ge 0 \\
                          \alpha, &\quad x < 0.
                        \end{cases}

    Args:
        alpha (float, optional): slope towards :math:`-\infty`.

    Shape:
        - input: x (numpy.array): input to compute the ``leaky_relu_prime`` function on.
        - output: y (numpy.array): derivative of the input, with the same shape than ``x``.

    .. image:: images/functional_leaky_relu_prime.png

    Examples::

        >>> in_array = np.array([-5, 2, 6, -2, 4])
        >>> alpha = 0.1
        >>> out_array = leaky_relu_prime(in_array, alpha)

    See :class:`~nets.nn.activation.LeakyReLU` for the activation implementation.
    """
    return where(x > 0, nets.ones_like(x), alpha * nets.ones_like(x))


