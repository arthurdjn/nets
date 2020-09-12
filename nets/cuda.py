# File: cuda.py
# Creation: Sunday September 6th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import numpy as np
import logging


def numpy_or_cupy(*tensors):
    r"""Load either NumPy or CuPy library, 
    depending if the computations on the tensors are made on CPU or GPU.

    Returns:
        module
    """
    module = __import__("numpy")
    device = np.mean([t.device == 'cuda' for t in tensors])

    if device == 1:
        return __import__("cupy")
    elif device == 0:
        return module
    else:
        logging.error(f"Cannot compute from tensors on different devices. "
                      f"Got {', '.join([t.device for t in tensors])}.")


def scalars_to_device(*tensors):
    device = np.mean([t.device == 'cuda' for t in tensors])
    # Put scalars to cuda if one tensor or more is on GPU
    if device > 0:
        for tensor in tensors:
            if tensor.shape == ():
                tensor.cuda()
