"""
Define a ``Dropout`` layer.
"""

from .module import Module
from ..functional import dropout


class Dropout(Module):
    """
    Applies a ``dropout`` to an input ``Tensor`` with probability ``prob``.
    The effect of this layer is to zeros elements of the incoming tensors,
    and cancel some neighbours effect / interactions from one layer to another.
    """

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, inputs):
        scale = 1 / (1 - self.prob) if self.prob < 1 else 0
        return scale * dropout(inputs, self.prob)

    # TODO: add a manual back-propagation for the dropout class
    # NOTE: but with autograd system, no need to worry about this
    def backward(self, outputs):
        raise NotImplementedError
