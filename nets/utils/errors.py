"""
Defines custom errors.
"""


class BackwardCallError(Exception):
    r"""
    This error is used when the call to ``backward`` method is not legit.
    """
    pass

