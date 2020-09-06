"""
A parameter is a trainable tensor.
"""

# Basic imports
import numpy as ops

# NETS Package
from nets.tensor import Tensor


class Parameter(Tensor):
    r"""
    Instantiate a parameter, made of trainable data. A trainable data is a value that will be updated during
    the back-propagation, usually it refers to ``weights`` and ``biases`` of a layer.
    """

    def __init__(self, data=None, shape=None):
        # First check if there are enough information to build the Parameter
        if data is None and shape is None:
            raise ValueError('You must specify the shape or data '
                             'to create a `Parameter`.')

        # If there is no data, generate data from a uniform distribution
        if shape is not None and data is None:
            data = ops.random.randn(*shape)
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
        sigma = ops.sqrt(var)
        weight_shape = (input_dim, output_dim)
        data = ops.random.normal(loc=mu, scale=sigma, size=weight_shape)
        return Parameter(data=data)

    @classmethod
    def zeros(cls, shape):
        r"""Generate a zero-Parameter

        Args:
            shape (tuple): shape of the ``Parameter``

        Returns:
            Parameter
        """
        return Parameter(data=ops.zeros(shape))

    @classmethod
    def uniform(cls, shape, low=-1, high=1):
        r"""Generate a ``Parameter`` with data following a uniform distribution between ``lower`` and ``upper``.

        Args:
            shape (tuple): shape of the ``Parameter``
            low (scalar): lower bound of the uniform distribution. Default is ``-1``.
            high (scalar): upper bound of the uniform distribution. Default is ``1``.

        Returns:
            Parameter
        """
        data = ops.random.uniform(low, high, shape)
        return Parameter(data=data)

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
        data = ops.random.normal(mu, sigma, shape)
        return Parameter(data=data)

    @classmethod
    def orthogonal(cls, shape):
        r"""Initializes weight parameters orthogonally.
        From the [exercise 02456 from DTU course](https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch).

        .. note::

            Refer to [this paper](https://arxiv.org/abs/1312.6120) for an explanation of this initialization.

        Args:
            shape (tuple): shape of dimensionality greater than 2 (weight matrix)

        Returns:
            Parameter
        """
        if len(shape) < 2:
            raise ValueError(
                "only parameters with 2 or more dimensions are supported.")

        rows, cols = shape
        data = ops.random.randn(rows, cols)

        if rows < cols:
            data = data.T

        # Compute QR factorization
        q, r = ops.linalg.qr(data)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        diag = ops.diag(r, 0)
        sign = ops.sign(diag)
        q *= sign

        if rows < cols:
            q = q.T

        data = q
        return Parameter(data=data)
