r"""
A ``Hook`` keeps track of gradients and operations.
"""


class Hook:
    """A hook is a collection of tensors and gradients.
    """

    __slots__ = 'tensor', 'grad_fn'

    def __init__(self, tensor, grad_fn):
        self.tensor = tensor
        self.grad_fn = grad_fn

    def __repr__(self):
        grad_name = self.grad_fn.__qualname__.split('.')[0]
        return f"Hook(Tensor.id={self.tensor.id}, grad_fn={grad_name.upper()})"
