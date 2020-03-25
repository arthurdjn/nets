"""
A hook keeps track of gradients.
"""

class Hook:
    """A hook is a collection of tensors and gradients.
    """
    def __init__(self, tensor, grad_fn):
        self.tensor = tensor
        self.grad_fn = grad_fn

    def __repr__(self):
        return f"hook({self.tensor}, grad_fn={self.grad_fn})"