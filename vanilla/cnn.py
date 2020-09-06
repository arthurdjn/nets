# File: cnn.py
# Creation: Wednesday August 19th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


"""
This modules defines a Convolution Neural Network (CNN) naively. This CNN is for test and comparisons purposes.
If you wan to use a more appropriate CNN for your models, use the ``nn.CNN`` instead.
"""

# Basic imports
from abc import ABC
import numpy as np

# NETS package
from nets.nn.modules import Module
from nets import Parameter


class CNN(Module):
    """
    Convolutional Neural Networks (CNN) are a class of Neural Networks that use convolution filters.
    Their particularity is their ability to synthesis information and learn spatial features.
    They are mainly used in Image Analysis, but are also known as *sliding windows* in Natural Language Processing.
    """

    def __init__(self, filter_n, filter_h, filter_w, pad_size=1, stride=1):
        super().__init__()
        self.pad_size = pad_size
        self.stride = stride
        self.weight = Parameter.normal((filter_n, filter_h, filter_w))
        self.bias = Parameter.zero((filter_n,))

    def forward(self, input_layer, weight, bias, pad_size=1, stride=1):
        r"""
        A naive implementation of the forward pass for a convolutional layer.

        The input consists of M data points, each with C channels, height H and
        width W. We convolve each input with C_o different filters, where each filter
        spans all C_i channels and has height H_w and width W_w.

        Args:
            input_layer: The input layer with shape (batch_size, channels_x, height_x, width_x)
            weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
            bias: Biases of shape (num_filters)

        Returns:
            output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)

        """
        # Shapes of inputs and weights
        batch_size, channels_x, height_x, width_x = input_layer.shape
        num_filters, channels_w, height_w, width_w = weight.shape

        # Raise / Errors
        assert channels_w == channels_x, ("The number of filter channels must be the same as the number of input layer channels")

        # 1/ Init the output
        height_y = 1 + (height_x + 2 * pad_size - height_w) // stride
        width_y = 1 + (width_x + 2 * pad_size - width_w) // stride
        shape = (batch_size, num_filters, height_y, width_y)
        output_layer = np.zeros(shape)

        # 2/ Creation of the convolved matrices
        # Pad the input images
        height_w2 = height_w // 2
        width_w2 = width_w // 2
        padding = ((0, 0), (0, 0), (height_w2, height_w2),
                   (width_w2, width_w2))
        input_layer_pad = np.pad(
            input_layer, pad_width=padding, mode='constant', constant_values=0)

        # 3/ For loops...
        for i in range(batch_size):
            for j in range(num_filters):
                for k in range(channels_x):
                    image = input_layer_pad[i, k, :, :]
                    kernel = weight[j, k, :, :]
                    for p in range(height_y):
                        for q in range(width_y):
                            # Convolution product
                            sub_image = image[p * stride:p * stride +
                                              height_w, q * stride:q * stride + width_w]
                            output_layer[i, j, p,
                                         q] += np.sum(sub_image * kernel)
                # Add the bias
                output_layer[i, j, :, :] += bias[j]

        return output_layer

    def backward(self, output_layer_gradient, input_layer, weight, bias, pad_size=1, stride=1):
        """
        A naive implementation of the backward pass for a convolutional layer.

        Args:
            output_layer_gradient: Gradient of the loss L wrt the next layer y, with shape
                (batch_size, num_filters, height_y, width_y)
            input_layer: Input layer x with shape (batch_size, channels_x, height_x, width_x)
            weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
            bias: Biases of shape (num_filters)

        Returns:
            input_layer_gradient: Gradient of the loss L with respect to the input layer x
            weight_gradient: Gradient of the loss L with respect to the filters w
            bias_gradient: Gradient of the loss L with respect to the biases b

        """
        batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
        batch_size, channels_x, height_x, width_x = input_layer.shape
        num_filters, channels_w, height_w, width_w = weight.shape

        assert num_filters == channels_y, ("The number of filters must be the same as the number of output layer channels")
        assert channels_w == channels_x, ("The number of filter channels be the same as the number of input layer channels")

        # 1/ Pad the input images
        height_w2 = height_w // 2
        width_w2 = width_w // 2
        padding = ((0, 0), (0, 0), (height_w2, height_w2), (width_w2, width_w2))

        # 2/ bias gradient
        bias_gradient = np.zeros_like(bias)
        # Sum all in the filter axis
        for i in range(batch_size):
            for p in range(height_y):
                for q in range(width_y):
                    bias_gradient += output_layer_gradient[i, :, p, q]

        # 3/ weight gradient
        weight_gradient = np.zeros_like(weight)
        input_layer_pad = np.pad(input_layer, pad_width=padding, mode='constant', constant_values=0)
        # Compute the weight gradient
        # --> cf jupyter-notebook formula
        # Reminder : weight.shape = (num_filters, channels_w, height_w, width_w)
        #                         = (     j     ,     k     ,    r    ,    s   )
        for i in range(batch_size):
            for j in range(num_filters):
                for k in range(channels_x):
                    for r in range(height_w):
                        for s in range(width_w):
                            for p in range(height_y):
                                for q in range(width_y):
                                    weight_gradient[j, k, r, s] += output_layer_gradient[i, j, p, q] * \
                                        input_layer_pad[i, k, stride * p + r, stride * q + s]

        # 4/ input gradient
        input_layer_gradient = np.zeros_like(input_layer)
        input_layer_gradient_pad = np.pad(
            input_layer_gradient, pad_width=padding, mode='constant', constant_values=0)

        # Compute the input_layer gradient
        # --> cf jupyter-notebook formula
        # Reminder : input_layer.shape = (batch_size, channels_x, height_x, width_x)
        #                              = (     i    ,     k     ,    p    ,    q   )
        for i in range(batch_size):
            for j in range(num_filters):
                for k in range(channels_x):
                    for r in range(height_w):
                        for s in range(width_w):
                            for p in range(height_y):
                                for q in range(width_y):
                                    input_layer_gradient_pad[i, k, stride * p + r, stride * q + s] += \
                                        output_layer_gradient[i, j, p, q] * weight[j, k, r, s]

        # Delete padding
        input_layer_gradient = input_layer_gradient_pad[:,
                                                        :, height_w2:-height_w2, width_w2:-width_w2]

        return input_layer_gradient, weight_gradient, bias_gradient
