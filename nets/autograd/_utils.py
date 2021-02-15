import numpy as np


def _reshape_keepdims(array, axis=0):
    new_shape = [array.shape[axis]] + [1] * (array.ndim - 1)
    new_shape = tuple(new_shape)
    return np.sum(array, axis=axis).reshape(new_shape)


def _slice_keepdims(array, indices):
    new_idx = []
    if isinstance(indices, int):
        new_idx.append(indices)
    else:
        for s in indices:
            if isinstance(s, int):
                new_idx.append([s])
            else:
                new_idx.append(s)
        new_idx = tuple(new_idx)
    return array[new_idx]


def numpy_unpad(x, pad_width):
    """Unpad an array.

    Args:
        x (numpy.ndarray): array to unpad
        pad_width (tuple): padding

    Returns:
        numpy.ndarray
    """
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]


def inv_permutation(permutation):
    """Get the inverse of a permutation. Used to invert a transposition for example.

    Args:
        permutation (list or tuple): permutation to invert.

    Returns:
        list
    """
    inverse = [0] * len(permutation)
    for i, p in enumerate(permutation):
        inverse[p] = i
    return inverse



