# File: conv.py
# Creation: Wednesday August 19th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


"""
This module defines a Convolution Neural Network (CNN).
"""

import numpy as np
import nets
from nets import Parameter
from .module import Module
from ._utils import col2im, im2col


class Conv2d(Module):
    r"""
    Convolutional Neural Networks (CNN) are a class of Neural Networks that uses convolutional filters.
    Their particularity is their ability to synthesize information and learn spatial features.
    They are mainly used in Image Analysis, but are also known as *sliding windows* in Natural Language Processing (NLP).

    ``Conv2d`` network applies a 2D convolution on a 4D tensor.
    """

    def __init__(self, in_channels, out_channels, filter_size, stride=1, pad=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        # initialized parameters follow a uniform distribution [-bound, bound]
        # more at https://pytorch.org/docs/stable/nn.html
        bound = 1 / (in_channels * np.product(filter_size))
        self.weight = Parameter.uniform((out_channels, in_channels, *filter_size), -bound, bound)
        self.bias = Parameter.zeros((out_channels, ))

    def forward(self, inputs):
        FN, C, FH, FW = self.weight.shape
        N, C, H, W = inputs.shape

        # TODO: display a warning if the stride does not match the input image size
        out_h = int((H + 2 * self.pad - FH) // self.stride) + 1
        out_w = int((W + 2 * self.pad - FW) // self.stride) + 1

        # Convolution
        col = im2col(inputs, FH, FW, self.stride, self.pad)
        col_weight = self.weight.reshape(FN, -1).T
        # Linear computation
        out = nets.dot(col, col_weight) + self.bias
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        # Save in the cache for manual back propagation
        self._cache['x'] = inputs
        self._cache['x_col'] = col
        self._cache['weight_col'] = col_weight

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.weight.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # Parameters gradient
        db = nets.sum(dout, axis=0)
        dw_col = nets.dot(self._cache['x_col'].T, dout)
        dw = dw_col.transpose(1, 0).reshape(FN, C, FH, FW)
        # Downstream gradient
        dcol = nets.dot(dout, self._cache['weight_col'].T)
        dx = col2im(dcol, self._cache['x'].shape, FH, FW, self.stride, self.pad)

        # Save the gradients
        # NOTE: we don't need to save column gradients as they wont be used during the optimization process.
        self._grads['bias'] = db
        self._grads['weight'] = dw

        return dx

    def inner_repr(self):
        """Display the inner parameter of a CNN"""
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, " \
               f"filter_size={self.filter_size}, stride={self.stride}, pad={self.pad}, " \
               f"bias={True if self.bias is not None else False}"
