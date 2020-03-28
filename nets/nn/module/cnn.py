"""
This module defines a Convolution Neural Network (CNN) naively. This CNN is for test and comparisons purposes.
If you wan to use a more appropriate CNN for your models, use the ``CNN`` instead.
"""

import numpy as np
from .module import Module


class CNNNaive(Module):
    """
    Convolutional Neural Networks (CNN) are a class of Neural Networks that use convolution filters.
    Their particularity is their ability to synthesis information and learn spatial features.
    They are mainly used in Image Analysis, but are also known as *sliding windows* in Natural Language Processing.
    """
    def __init__(self):
        pass

    def forward(self, *inputs):
        pass

    def conv_layer_forward(self, input_layer, weight, bias, pad_size=1, stride=1):
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
        # TODO: Task 2.1

        # Shapes of inputs and weights
        batch_size, channels_x, height_x, width_x = input_layer.shape
        num_filters, channels_w, height_w, width_w = weight.shape

        # Raise / Errors
        assert channels_w == channels_x, (
            "The number of filter channels must be the same as the number of input layer channels")

        # 1/ Init the output
        height_y = 1 + (height_x + 2 * pad_size - height_w) // stride
        width_y = 1 + (width_x + 2 * pad_size - width_w) // stride
        shape = (batch_size, num_filters, height_y, width_y)
        output_layer = np.zeros(shape)

        # 2/ Creation of the convolved matrices
        # Pad the input images
        height_w2 = height_w // 2
        width_w2 = width_w // 2
        padding = ((0, 0), (0, 0), (height_w2, height_w2), (width_w2, width_w2))
        input_layer_pad = np.pad(input_layer, pad_width=padding, mode='constant', constant_values=0)

        # 3/ For loops...
        for i in range(batch_size):
            for j in range(num_filters):
                for k in range(channels_x):
                    image = input_layer_pad[i, k, :, :]
                    kernel = weight[j, k, :, :]
                    for p in range(height_y):
                        for q in range(width_y):
                            # Convolution product
                            sub_image = image[p * stride:p * stride + height_w, q * stride:q * stride + width_w]
                            output_layer[i, j, p, q] += np.sum(sub_image * kernel)
                # Add the bias
                output_layer[i, j, :, :] += bias[j]

        return output_layer