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
