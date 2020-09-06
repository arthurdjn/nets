"""
Defines elementary functions used in Neural Network layers.
"""

import numpy as np
import nets


def dropout(t, prob=0.5):
    r"""Zeros elements from a ``Tensor`` with a probability ``prob``.

    .. math::

        \text{dropout}(T) = T \times Z \quad \text{where} Z = (z_{i})_{i} \quad and z_i =
                                                                                        \begin{cases}
                                                                                          1, &\quad p \ge prob \\
                                                                                          0, &\quad else.
                                                                                        \end{cases}

    Args:
        t (Tensor): tensor to zeros
        prob (float [0, 1]): probability to zero an element

    Returns:
        Tensor: input tensor with some zeros
    """
    # Randomly generates number following a uniform distribution between [0, 1]
    probabilities = np.random.uniform(low=0.0, high=1.0, size=t.shape)
    # Generate a mask of (0, 1). 0 means probabilities[index] > prob, 1 else.
    mask = np.where(probabilities > prob, 0, 1)
    mask = nets.Tensor(mask)
    # Applies the mask to the tensor to get the dropout (elementwise multiplication)
    t_drop = t * mask
    return t_drop


# TODO: defines a single convolution filter
def conv2d(t, filter):
    pass
