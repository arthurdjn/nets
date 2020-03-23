r"""
This module focus on basic functions implementation and their first order derivatives as well.
These functions are used as methods in their respective activation function class,
defined in the nets.nn.activation module.

Usage:

.. code-block:: default

    import numpy as np
    from nets.functional import relu

    x = np.array([1, -2, 5, 10, -7])
    y = relu(x)
    print(y)


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    array([1, 0, 5, 10, 0])
"""

import numpy as np


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
    exp = np.exp(x)
    t = x - np.log(np.sum(exp, axis=axis, keepdims=True))
    s = np.exp(t)
    return s


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
    return 1.0 / (1.0 + np.exp(-x))


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


def tanh(x):
    # type: (Array) -> Array
    r"""``tanh`` standard function, definead as:

    .. math::
        \text{tanh}(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

    Shape:
        - input: x (numpy.array): input to compute the ``tanh`` function on.
        - output: y (numpy.array): ``tanh`` output, with the same shape than ``x``.

    .. image:: images/functional_tanh.png

    Examples::

        >>> in_array = np.array([-5, 2, 6, -2, 4])
        >>> out_array = tanh(in_array)

    See :class:`~nets.nn.activation.Tanh` for the activation implementation.
    """
    return np.tanh(x)


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
    return np.maximum(0, x)


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
    return np.where(x >= 0, 1, 0)


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
    return np.where(x > 0, x, x * alpha)


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
    return np.where(x > 0, 1, alpha)