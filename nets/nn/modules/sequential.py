"""
``Sequential`` models are an ordered succession of modules.
"""


from .module import Module


class Sequential(Module):
    r"""
    A ``Sequential`` model is build from a succession of ``Modules``. All of them **must** have a
    ``forward`` method. In addition, a ``backward`` pass is created by default, running the back-propagation in all
    modules previously added. If for one of the modules the ``backward`` pass is not implemented, it will raise an error.
    """

    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.add(*modules)

    def forward(self, inputs):
        r"""Compute the forward pass for all modules within the sequential modules.

        Shape:
            - inputs (Tensor): incoming data.
            - outputs (Tensor): result of all forward pass.
        """
        for module in self.modules():
            inputs = module.forward(inputs)
        return inputs

    def backward(self, grad):
        r"""Vanilla backward pass. This pass computes local gradients from ``parameters`` saved in its ``_cache``.

        Shape:
            - inputs (Tensor): upstream gradient. The first downstream gradient is usually the ``loss``.
            - outputs (Tensor): last downstream gradient.
        """

        # TODO: further test as self.modules() is an iterator generator

        for module in reversed(self.modules()):
            grad = module.backward(grad)
        return grad
