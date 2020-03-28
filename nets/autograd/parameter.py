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
        if shape is not None:
            data = np.random.randn(*shape)
        elif data is None and shape is None:
            data = []
        super().__init__(data, requires_grad=True)

    @classmethod
    def scale(cls, shape):
        pass

    @classmethod
    def uniform(cls, shape):
        pass
