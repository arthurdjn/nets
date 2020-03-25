import numpy as np


def one_hot(Y, num_classes):
    r"""Perform one-hot encoding on input Y.

    .. math::

        \text{Y'}_{i, j} =
                    \begin{cases}
                      1, &\quad if \quad Y_i = 0 \\
                      0, &\quad else
                    \end{cases}

    Args:
        Y (Tensor): 1D tensor of classes indices of length :math:`N`
        num_classes (int): number of classes :math:`c`

    Returns:
        Tensor: one hot encoded tensor of shape :math:`(N, c)`
    """
    batch_size = len(Y)
    Y_tilde = np.zeros((batch_size, num_classes))
    Y_tilde[np.arange(batch_size), Y] = 1
    return Y_tilde.astype(int)
