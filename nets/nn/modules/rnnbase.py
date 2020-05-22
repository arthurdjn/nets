r"""
Defines the general structure of a Recurrent Network, which can takes the form of a:

- Recurrent Neural Network,
- Long Short Term Memory,
- Gated Recurrent Unit.
"""

from abc import ABC
from .module import Module


# TODO: finalize this abstract class
class RNNBase(Module, ABC):
    r"""
    Base architecture for Recurrent Networks.
    Every general recurrent layers should extends this class to inherits private attributes and
    helper methods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
