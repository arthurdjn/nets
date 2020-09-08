# File: functions.py
# Creation: Tuesday September 8th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# NETS package
import nets


def softmax(x, axis=0):
    r"""Compute the softmax function using an implementation that prevent numerical instability.
    The :math:`i^{th}` element of a softmax function evaluated 
    on a vector :math:`x \in \mathbb{R}^n` is given by:

    .. math::
        \text{softmax}(x_{i}) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}

    Args:
        axis (int): axis to consider for the ``softmax``. Default is ``0``.

    Shape:
        - input: x (numpy.array): input to compute the ``softmax`` function on.
        - output: y (numpy.array): hyper-plane of the input.

    See :class:`~nets.nn.activation.Softmax` for the activation implementation.
    """
    e = nets.exp(x)
    s = nets.sum(e, axis=axis, keepdims=True)
    t = x - nets.log(s)
    soft = nets.exp(t)
    return soft

