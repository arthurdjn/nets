r"""
This module defines all activation function, used in neural networks to add non-linearity from one layer to another.

Usage:

.. code-block:: default

    import numpy as np
    import nets.nn.activation as A

    # Define a ReLU activation function
    # This activation function is popular as an hidden activation function
    activation_function = A.ReLU()
    # Defines 3 1-D array of length 5, stacked together.
    batch_input = np.array([[1, -2, 5, 10, -7],
                            [2, 4, 3, -2, 5],
                            [-3, 2, 8, 4, 0]])

    # Check the result for one single pass
    batch_output = activation_function(batch_input)
    print(batch_output)

    # Check the backward pass
    backward = ReLU.backward(batch_ouput)
    print(backward)




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    array([[1, 0, 5, 10, 0],
           [2, 4, 3, 0, 5],
           [0, 2, 8, 4, 0]])
    array([[1, 0, 1, 1, 0],
           [1, 1, 1, 0, 1],
           [0, 1, 1, 1, 0]])
"""


from nets.functional import *
from nets.nn.module import Module


class Activation(Module):
    r"""
    An activation module is a transformation that modify its inputs element wise, usually it uses non-linearity
    functions.
    """

    def __init__(self, func, func_prime):
        super(Activation, self).__init__()
        self.func = func
        self.func_prime = func_prime

    def forward(self, inputs):
        self._cache['x'] = inputs
        z = self.func(inputs)
        return z

    def backward(self, grad):
        x = self._cache['x']
        return self.func_prime(x) * grad


class ReLU(Activation):
    r"""
    ReLU activation function, defined as:

    :math:`\text{ReLU.forward}(x) = \text{relu(x)} = \max{(0, x)}`

    :math:`\text{ReLU.backward}(x) = \text{relu'(x)} = \begin{cases} 1, &\quad x \ge 0 \\ 0, &\quad x < 0. \end{cases}`


    .. image:: images/activation_ReLU.png

    Examples::

        >>> activation = ReLU()
        >>> batch_input = np.array([[-5, 2, 6, -2, 4],
        ...                         [2, 5, -6, 7, -3]])
        >>> output = activation(batch_input)
        >>> gradient = activation.backward(output)

    See :func:`~nets.functional.relu` and :func:`~nets.functional.relu_derivative`
    for the functional implementation.
    """
    def __init__(self):
        super().__init__(relu, relu_prime)


class Tanh(Activation):
    r"""
    ``Tanh`` activation function.

    :math:`\text{Tanh.forward}(x) = \text{tanh}(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}`

    :math:`\text{Tanh.backward}(x) = \text{tanh'}(x) = 1 - \text{tanh}^2(x)`

    .. image:: images/activation_Tanh.png

    Examples::
    
        >>> activation = Tanh()
        >>> batch_input = np.array([[-5, 2, 6, -2, 4],
        ...                         [2, 5, -6, 7, -3]])
        >>> output = activation(batch_input)
        >>> gradient = activation.backward(output)

    See :func:`~nets.functional.tanh` and :func:`~nets.functional.tanh_derivative`
    for the functional implementation.
    """
    def __init__(self):
        super().__init__(tanh, tanh_prime)


class Softmax(Activation):
    r"""
    Softmax activation function, which should be used only at the last layer to normalize outputs values between
    :math:`[0, 1]`.
    """
    def __init__(self, axis=0):
        super().__init__(softmax, None)
        self.axis = axis

    def forward(self, inputs):
        return self.func(inputs, axis=self.axis)

    def backward(self, grad):
        raise RuntimeError('Softmax activation cannot be used during back-propagation')

# class ReLU(Activation):
#     r"""
#     ``ReLU`` activation function.
#
#     :math:`\text{ReLU.forward}(x) = \text{relu(x)} = \max{(0, x)}`
#
#     :math:`\text{ReLU.backward}(x) = \text{relu'(x)} = \begin{cases} 1, &\quad x \ge 0 \\ 0, &\quad x < 0. \end{cases}`
#
#
#     .. image:: images/activation_ReLU.png
#
#     Examples::
#
#         >>> activation = LeakyReLU()
#         >>> batch_input = np.array([[-5, 2, 6, -2, 4],
#         ...                         [2, 5, -6, 7, -3]])
#         >>> output = activation(batch_input)
#         >>> gradient = activation.backward(output)
#
#     See :func:`~nets.functional.relu` and :func:`~nets.functional.relu_derivative`
#     for the functional implementation.
#     """
#
#     def __init__(self):
#         super(ReLU, self).__init__()
#
#     def forward(self, x):
#         return relu(x)
#
#     def backward(self, x):
#         return relu_derivative(x)
#
#
# class LeakyReLU(Activation):
#     r"""
#     ``LeakyReLU`` activation function.
#
#     :math:`\text{LeakyReLU.forward}(x) = \text{leaky_relu(x)} = \max{(\alpha \times x, x)}`
#
#     :math:`\text{LeakyReLU.backward}(x) = \text{leaky_relu'(x)} = \begin{cases} 1, &\quad x \ge 0 \\ \alpha, &\quad x < 0. \end{cases}`
#
#     .. image:: images/activation_LeakyReLU.png
#
#     Examples::
#
#         >>> activation = LeakyReLU()
#         >>> batch_input = np.array([[-5, 2, 6, -2, 4],
#         ...                         [2, 5, -6, 7, -3]])
#         >>> output = activation(batch_input)
#         >>> gradient = activation.backward(output)
#
#     See :func:`~nets.functional.leaky_relu` and :func:`~nets.functional.leaky_relu_derivative`
#     for the functional implementation.
#     """
#
#     def __init__(self):
#         super(LeakyReLU, self).__init__()
#
#     def forward(self, x):
#         return leaky_relu(x)
#
#     def backward(self, x):
#         return leaky_relu_derivative(x)
#
#
# class Sigmoid(Activation):
#     r"""
#     ``Sigmoid`` activation function.
#
#     :math:`\text{Sigmoid.forward}(x) = \text{sigmoid}(x) = \frac{1}{1 + e^{-x}}`
#
#     :math:`\text{Sigmoid.backward}(x) = \text{sigmoid'}(x) = (1 - \text{sigmoid}(x))`
#
#     .. image:: images/activation_Sigmoid.png
#
#     Examples::
#
#         >>> activation = Sigmoid()
#         >>> batch_input = np.array([[-5, 2, 6, -2, 4],
#         ...                         [2, 5, -6, 7, -3]])
#         >>> output = activation(batch_input)
#         >>> gradient = activation.backward(output)
#
#     See :func:`~nets.functional.sigmoid` and :func:`~nets.functional.sigmoid_derivative`
#     for the functional implementation.
#     """
#
#     def __init__(self):
#         super(Sigmoid, self).__init__()
#
#     def forward(self, x):
#         return sigmoid(x)
#
#     def backward(self, x):
#         return sigmoid_derivative(x)
#
#
# class Tanh(Activation):
#     r"""
#     ``Tanh`` activation function.
#
#     :math:`\text{Tanh.forward}(x) = \text{tanh}(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}`
#
#     :math:`\text{Tanh.backward}(x) = \text{tanh'}(x) = 1 - \text{tanh}^2(x)`
#
#     .. image:: images/activation_Tanh.png
#
#     Examples::
#
#         >>> activation = Tanh()
#         >>> batch_input = np.array([[-5, 2, 6, -2, 4],
#         ...                         [2, 5, -6, 7, -3]])
#         >>> output = activation(batch_input)
#         >>> gradient = activation.backward(output)
#
#     See :func:`~nets.functional.tanh` and :func:`~nets.functional.tanh_derivative`
#     for the functional implementation.
#     """
#
#     def __init__(self):
#         super(Tanh, self).__init__()
#
#     def forward(self, x):
#         return tanh(x)
#
#     def backward(self, x):
#         return tanh_derivative(x)
#
#
# class Affine(Activation):
#     def __init__(self, weight=None, scale=1, bias=0):
#         super(Affine, self).__init__()
#         self.weight = weight
#         self.scale = scale
#         self.bias = bias
#
#     def forward(self, x):
#         if self.weight is not None:
#             x = np.dot(x, self.weight)
#         return self.scale * x + self.bias
#
#     def backward(self, x):
#         return self.scale
#
#
# class Softmax(Activation):
#
#     def __init__(self, axis=0):
#         super(Softmax, self).__init__()
#         self.axis = axis
#
#     def forward(self, x):
#         return softmax(x, axis=self.axis)
#
#     def backward(self, x):
#         return NotImplementedError