"""
Defines tensors for deep learning application. A tensor is multi-dimensional array, similar to NumPy arrays.
"""

import numpy as np
import nets


def to_array(arrayable):
    if isinstance(arrayable, np.ndarray):
        return arrayable
    elif isinstance(arrayable, Tensor):
        return np.array(arrayable.data)
    else:
        return np.array(arrayable)


def to_tensor(tensorable):
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor(object):
    """A Tensor is a multi dimensional array that track and record previous gradients, creating a dynamic
    computational graph.
    """

    def __init__(self, data, requires_grad=False, hooks=None):
        self._data = to_array(data)
        self.requires_grad = requires_grad
        self._hooks = hooks or []
        self._shape = self._data.shape
        self.ndim = self._data.ndim
        self.size = self._data.size
        self.grad = None

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data
        # Setting the data manually means we invalidate the gradient.
        self.grad = None

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, new_shape):
        raise AttributeError(
            'cannot change the shape of a tensor manually. Please create a new tensor or set a new data instead')

    @property
    def T(self):
        return nets.transpose(self)

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def backward(self, grad=None):
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data = self.grad.data + grad.data  # type: ignore

        for hook in self._hooks:
            backward_grad = hook.grad_fn(grad.data)
            hook.tensor.backward(Tensor(backward_grad))

    def item(self):
        return self.data.item()

    def tolist(self):
        return self.data.tolist()

    def numpy(self):
        return self.data

    def sum(self, axis=None):
        return nets.sum(self, axis)

    def transpose(self):
        return nets.transpose(self)

    def __repr__(self):
        string_data = np.array2string(self.data, prefix="       ",
                                      precision=4)
        requires_grad = "" if not self.requires_grad else f", requires_grad={self.requires_grad}"
        return f"tensor({string_data}{requires_grad})"

    def __len__(self):
        return len(self.data)

    def __gt__(self, other):
        return nets.gt(self, other)

    def __ge__(self, other):
        return nets.ge(self, other)

    def __lt__(self, other):
        return nets.lt(self, other)

    def __le__(self, other):
        return nets.le(self, other)

    def __eq__(self, other):
        return nets.eq(self, other)

    def __ne__(self, other):
        return nets.ne(self, other)

    def __add__(self, other):
        return nets.add(self, other)

    def __radd__(self, other):
        return nets.add(other, self)

    def __iadd__(self, other):
        self.data = self.data + nets.to_tensor(other).data
        return self

    def __neg__(self):
        return nets.neg(self)

    def __sub__(self, other):
        return nets.sub(self, other)

    def __rsub__(self, other):
        return nets.sub(other, self)

    def __isub__(self, other):
        self.data = self.data - nets.to_tensor(other).data
        return self

    def __mul__(self, other):
        return nets.multiply(self, other)

    def __rmul__(self, other):
        return nets.multiply(other, self)

    def __imul__(self, other):
        self.data = self.data * nets.to_tensor(other).data
        return self

    def __pow__(self, power, modulo=None):
        return nets.pow(self, power)

    def __truediv__(self, other):
        return nets.div(self, other)

    def __rtruediv__(self, other):
        return nets.div(other, self)

    def __itruediv__(self, other):
        self.data = self.data / nets.to_tensor(other).data
        return self

    def __matmul__(self, other):
        return nets.dot(self, other)

    def __getitem__(self, indices):
        return nets.slice(self, indices)
