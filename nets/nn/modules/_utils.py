"""
Some utility functions and helper in ``CNN`` modules.

.. warning::

    some functions defined in this modules are not compatible with the autograd system as they do in-place operations.

"""

import numpy as np
import nets


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    r"""Transform 4 dimensional images to 2 dimensional array.
    From stanford university cs231n assignment 2.
    More [here](http://cs231n.stanford.edu/).

    Args:
        input_data (Tensor): 4 dimensional input images (The number of images, The number of channels, Height, Widht)
        filter_h (int): height of filter
        filter_w (int): width of fitter
        stride (int): the interval of stride
        pad (int): the interval of padding

    Returns:
        col (Tensor): 2 dimnesional tensor

    .. warning::

        This function is not compatible with ``autograd``system. The resulting ``Tensor`` has no links to a
        previous computational graph, and in addition its gradient is set to ``None``.

    """
    # Extract the shape from one's image
    N, C, H, W = input_data.shape

    # Make sure that the convolution can be executed
    # TODO: replace by a warning
    assert (H + 2 * pad - filter_h) % stride == 0, f'invalid parameters, (H + 2 * pad - filter_h) % stride != 0, got ' \
                                                   f'H={H}, pad={pad}, filter_h={filter_h}, stride={stride}'
    assert (W + 2 * pad - filter_w) % stride == 0, f'invalid parameters, (W + 2 * pad - filter_w) % stride != 0, got ' \
                                                   f'W={W}, pad={pad}, filter_w={filter_w}, stride={stride}'
    # Initialize the output dimensions
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    # Pad the input data
    padding = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    image = nets.pad(input_data, padding)
    # Initialize the output
    col = nets.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # For loops...
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = image[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """Inverse of im2col.

    Args:
        col (Tensor): 2 dimensional array
        input_shape (tuple): the shape of original input images
        filter_h (int): height of filter
        filter_w (int): width of filter
        stride (int): the interval of stride
        pad (int): the interval of padding

    Returns:
        image (Tensor): original images

    .. warning::

        This function is not compatible with ``autograd``system. The resulting ``Tensor`` has no links to a
        previous computational graph, and in addition its gradient is set to ``None``.

    """
    # Extract the shape from one's image
    N, C, H, W = input_shape

    # Make sure that the convolution can be executed
    # TODO: replace by a warning
    assert (H + 2 * pad - filter_h) % stride == 0, f'invalid parameters, (H + 2 * pad - filter_h) % stride != 0, got ' \
                                                   f'H={H}, pad={pad}, filter_h={filter_h}, stride={stride}'
    assert (W + 2 * pad - filter_w) % stride == 0, f'invalid parameters, (W + 2 * pad - filter_w) % stride != 0, got ' \
                                                   f'W={W}, pad={pad}, filter_w={filter_w}, stride={stride}'

    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C,
                      filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    image = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            image[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return image[:, :, pad:H + pad, pad:W + pad]
