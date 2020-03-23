import numpy as np


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length batch_size with integer
    values in range [0, num_classes - 1]. The encoded matrix Y_tilde will
    be a [num_classes, batch_size] shaped matrix with values:

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else

    Parameters
    ----------
    Y : numpy.ndarray
        Batched 1D array of predictions.
    num_classes : int
        Number of classes.

    Returns
    -------
    Y_tilde : numpy.ndarray
        One hot encoded vector of shape [num_classes, batch_size].

    """
    batch_size = len(Y)
    Y_tilde = np.zeros((batch_size, num_classes))
    Y_tilde[np.arange(batch_size), Y] = 1
    return Y_tilde
