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
    devices = [tensor.device == 'cuda' for tensor in tensors]
    if np.mean(devices) == 1:
        try:
            return __import__("cupy")
        except Exception as error:
            logging.error(f"CuPy not imported. {error}")
    elif np.mean(devices) == 0:
        return module
    else:
        logging.error("Cannot compute from tensors on different devices.")