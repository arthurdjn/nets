# File: hook.py
# Creation: Wednesday August 19th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


r"""
A ``Hook`` keeps track of gradients and operations.
"""

# TODO: `Hook` should just be gradient functions, so users can add some `lambda x: print(x)` to print gradients easily
class Hook:
    """A hook is a collection of tensors and gradients.
    """

    __slots__ = 'tensor', 'grad_fn'

    def __init__(self, tensor, grad_fn):
        self.tensor = tensor
        self.grad_fn = grad_fn

    def __repr__(self):
        grad_name = self.grad_fn.__qualname__.split('.')[0]
        return f"Hook(tensor_id={self.tensor.id}, grad_fn={grad_name.upper()})"
