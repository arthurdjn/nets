"""
Defines custom errors.
"""


class BackwardCallError(Exception):
    r"""
    This error is used when the call to ``backward`` method is not legit.
    """
    pass


class CUDANotAvailableError(Exception):
    """
    This error is raised if CUDA is not available and the user wants to work with GPU.
    """
    pass
