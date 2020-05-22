r"""
This modules defines a ``Pooling`` layer. Usually such layer is used after a convolutional layer.
"""

import numpy as np
import nets
from .module import Module
from ._utils import im2col, col2im


class MaxPool2d(Module):
    """
    A ``Pooling`` layer extract features from a multi dimensional ``Tensor`` and map them into another one.
    This extraction is used to decrease the dimension of the input, and often used after a convolutional layer.
    """

    def __init__(self, pool_size, stride=1, pad=0):
        super().__init__()
        # Make sure the pool_size is a 2-d filter
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        # Initialize
        self.pool_size = pool_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        """Forward pass."""
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_size[0]) / self.stride)
        out_w = int(1 + (W - self.pool_size[1]) / self.stride)
        # Reshape the input into a 2-d tensor
        col = im2col(x, *self.pool_size, self.stride, self.pad)
        col = col.reshape(-1, np.product(self.pool_size))

        # Keep track of the argmax indices for manual back-propagation
        argmax = nets.argmax(col, axis=1)
        out = nets.max(col, axis=1)
        out = out.reshape(N, out_h + 2*self.pad, out_w + 2*self.pad, C).transpose(0, 3, 1, 2)

        # Save the parameters in the cache for manual back-propagation
        self._cache['x'] = x
        self._cache['argmax'] = argmax

        return out

    def backward(self, dout):
        """Manual backward pass for a MaxPool2d layer."""
        dout = dout.transpose(0, 2, 3, 1)
        # Initialize
        pool_size = np.product(self.pool_size)
        dmax = nets.zeros((dout.size, pool_size))

        # Get the cache
        x = self._cache['x']
        argmax = self._cache['argmax']
        dmax[nets.arange(argmax.size), argmax.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, x.shape, *self.pool_size, self.stride, self.pad)

        return dx

    def inner_repr(self):
        """Display the inner parameter of a CNN"""
        return f"pool_size={self.pool_size}, stride={self.stride}, pad={self.pad}"
