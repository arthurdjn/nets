# File: cuda.py
# Creation: Sunday September 6th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import numpy
import logging


CUDA_AVAILABLE = False
try:
    import cupy
    CUDA_AVAILABLE = True
except ModuleNotFoundError:
    pass


__all__ = [
    "cuda_available",
    "numpy_or_cupy",
    "scalars_to_device"
]


def cuda_available():
    """Check if `CUDA` is available.

    Returns:
        bool
    """
    return CUDA_AVAILABLE


def numpy_or_cupy(*tensors):
    r"""Load either NumPy or CuPy library, 
    depending if the computations on the tensors are made on CPU or GPU.

    Returns:
        module
    """
    device = numpy.mean([t.device == 'cuda' for t in tensors])
    if device == 1:
        return cupy
    elif device == 0:
        return numpy
    else:
        logging.error(f"Cannot compute from tensors on different devices. "
                      f"Got {', '.join([t.device for t in tensors])}.")


def scalars_to_device(*tensors):
    device = numpy.mean([t.device == 'cuda' for t in tensors])
    # Put scalars to cuda if one tensor or more is on GPU
    if device > 0:
        for tensor in tensors:
            if tensor.shape == ():
                tensor.cuda()
