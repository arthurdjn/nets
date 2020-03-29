"""
A parameter is a trainable tensor.
"""

import numpy as np
from nets.tensor import Tensor


class Parameter(Tensor):
    r"""
    Instantiate a parameter, made of trainable data. A trainable data is a value that will be updated during
    the back-propagation, usually it refers to ``weights`` and ``biases`` of a layer.
    """

    def __init__(self, data=None, shape=None):
        # First check if there are enough information to build the Parameter
        if data is None and shape is None:
            raise ValueError('you must specify the shape or data to create a Parameter')

        # If there is no data, generate data from a uniform distribution
        if shape is not None and data is None:
            data = np.random.randn(*shape)
        # Create the Tensor
        super().__init__(data, requires_grad=True)

    @classmethod
    def scaled_weight(cls, input_dim, output_dim):
        r"""Scaled initialization from :math:`He et al.`

        Args:
            input_dim (int): dimension of the input layer
            output_dim (int): dimension of the output layer

        Returns:
            Parameter
        """
        mu = 0
        var = 2 / input_dim
        sigma = np.sqrt(var)
        weight_shape = (input_dim, output_dim)
        data = np.random.normal(loc=mu, scale=sigma, size=weight_shape)
        return Parameter(data=data)

    @classmethod
    def zero(cls, shape):
        r"""Generate a zero-Parameter

        Args:
            shape (tuple): shape of the ``Parameter``

        Returns:
            Parameter
        """
        return Parameter(data=np.zeros(shape))

    @classmethod
    def uniform(cls, shape, lower=-1, upper=1):
        r"""Generate a ``Parameter`` with data following a uniform distribution between ``lower`` and ``upper``.

        Args:
            shape (tuple): shape of the ``Parameter``
            lower (scalar): lower bound of the uniform distribution. Default is ``-1``.
            upper (scalar): upper bound of the uniform distribution. Default is ``1``.

        Returns:
            Parameter
        """
        data = np.random.uniform(lower, upper, shape)
        return Parameter(data)

    @classmethod
    def normal(cls, shape, mu=0, sigma=1):
        r"""Generate a ``Parameter`` following a normal distribution center at ``mu`` with a standard deviation of
        ``sigma``.

        Args:
            shape (tuple): shape of the ``Parameter``
            mu (scalar): mean of the normal distribution. Default is ``0``.
            sigma (scalar): standard deviation of the normal distribution. Default is ``1``.

        Returns:
            Parameter
        """
        data = np.random.normal(mu, sigma, shape)
        return Parameter(data)
