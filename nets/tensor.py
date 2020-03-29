"""
Defines tensors for deep learning application. A tensor is multi-dimensional array, similar to ``numpy`` arrays.
"""

import numpy as np
import nets
from nets.utils import BackwardCallError


def to_array(arrayable):
    """Convert an object to a ``numpy.ndarray`` if possible.

    Args:
        arrayable: object to convert

    Returns:
        numpy.ndarray
    """
    if isinstance(arrayable, np.ndarray):
        return arrayable
    elif isinstance(arrayable, Tensor):
        return np.array(arrayable.data)
    else:
        return np.array(arrayable)


def to_tensor(tensorable):
    """Convert an object to a ``Tensor`` if possible.

    Args:
        tensorable: object to convert

    Returns:
        Tensor
    """
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor(object):
    """A Tensor is a multi dimensional array that track and record previous gradients, creating a dynamic
    computational graph.
    """

    # Objects instance are heavy-weight in Python.
    # Setting slots free memory, and does not keep built-in functions (__builtin__ things)
    __slots__ = '_data', 'requires_grad', '_hooks', 'grad', '_id'

    # A global parameter to track how many Tensor have been instantiate.
    # This is mainly for debugging and visualization
    _COUNTER = 0

    def __init__(self, data, requires_grad=False, hooks=None):
        self._data = to_array(data)
        self.requires_grad = requires_grad
        self._hooks = hooks or []
        self.grad = None
        # Update the tracking
        self._id = Tensor._COUNTER
        Tensor._COUNTER += 1

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data
        # Setting the data manually means we invalidate the gradient.
        self.detach()

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype

    @dtype.setter
    def dtype(self, new_dtype):
        self._data = self._data.astype(new_dtype)
        self.detach()

    @property
    def id(self):
        return self._id

    @property
    def T(self):
        return nets.transpose(self)

    def astype(self, new_type):
        r"""Set a new type to the ``Tensor``'s data.

        Args:
            new_type (type): new type to convert the data

        Returns:
            None
        """
        self.detach()
        return nets.astype(self, new_type)

    def detach(self):
        r"""Unlink the ``Tensor`` to the computational graph.
        By calling this method, the attribute ``_hooks`` and ``grad`` are set to their default values,
        ``None``.

        Returns:
            None
        """
        self.grad = None
        self._hooks = []

    def zero_grad(self):
        r"""Set to a zero ``Tensor`` the gradient. This is call when initializing a ``Tensor`` that requires gradient
        tracking, or re-initialize parameters's gradient after a training loop as they accumulate on each other.

        Returns:
            None
        """
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def backward(self, grad=None):
        r"""Compute a single backward pass on all ``Tensor`` linked to this one.
        The ``Tensor`` depending to this top-level ``Tensor``are stored in the ``_hooks`` attribute.
        The backward pass compute a gradient back-propagation on all ``Tensor`` registered in ``_hooks``.
        The backward pass gradient in ``grad`` attribute (and add upstream gradient if the ``Tensor``
        is used multiple times).

        Args:
            grad (Tensor): upstream gradient. Default is None, and will be set to ``Tensor(1.0)``, a 0-dimensional
                ``Tensor``.

        Returns:
            None

        .. note::

            To be able to back-propagate, the top-level ``Tensor`` must have ``requires_grad`` set to ``True``
            to propagate the gradient.
        """
        # Check if the backward pass is legit
        if not self.requires_grad:
            raise BackwardCallError(r"called backward on non `requires_grad` tensor. Either there was no "
                                    r"`requires_grad=True` initialization or gradients were set to `None` due to an "
                                    r"inplace operation or the computational graph was split and gradients are no "
                                    r"longer linked to this branch. Graph are usually split when a new tensor is "
                                    r"created from a `numeric` function (zero, ones, eye, identity) and "
                                    r"`requires_grad` was not specified.")
        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        # Update the gradient
        # NOTE: the gradients accumulate !
        self.grad.data = self.grad.data + grad.data  # type: ignore

        # Back-propagation in all dependencies
        hooks = self._hooks
        if hooks is not None:
            for hook in self._hooks:
                # Compute the gradient wrt the operation
                backward_grad = hook.grad_fn(grad.data)
                # Back-propagate in the tensor used in this operation
                hook.tensor.backward(Tensor(backward_grad))

        # TODO: handle properly nodes and leaf from different hooks
        # TODO: maybe add Variable class / is_leaf attributes
        # TODO: and counter to skip gradients that don't need to be set

    def item(self):
        r"""
        Get the item (float, int etc.) of a 0-dimensional ``Tensor``. It will detach the tensor from the computational
        graph by setting ``_hooks = []`` and ``grad = None`` to free memory and send this graph to the garbage collector.

        Returns:
            Any
        """
        self.detach()
        return self.data.item()

    def tolist(self):
        r"""Convert the ``Tensor`` data to a list (of list eventually).

        Returns:
            list
        """
        self.detach()
        return self.data.tolist()

    def numpy(self):
        r"""Convert the ``Tensor`` data to a ``numpy.ndarray`` object.

        Returns:
            numpy.ndarray
        """
        self.detach()
        return self.data

    def sum(self, axis=None):
        r"""Sum the data along a given axis. If no axis are specified, all values within the ``Tensor``will be summed.

        Args:
            axis (int): the index of the axis to sum on.

        Returns:
            Tensor
        """
        return nets.sum(self, axis)

    def transpose(self, *indices):
        r"""Transpose the ``Tensor``. The operation is not in-place.

        Args:
            indices (tuple): permutation

        Returns:
            Tensor
        """
        return nets.transpose(self, indices)

    def reshape(self, *shapes):
        r"""Reshape a ``Tensor`` with a new shape. The transformation is not made in-place.

        .. note::

            The new shape **must** have the same size of the actual shape.
            If its not the case, the reshape method will raise an error.

        Args:
            *shapes int: permutation

        Returns:
            Tensor
        """
        return nets.reshape(self, shapes)

    def flatten(self):
        r"""Flatten a ``Tensor`` with a new shape. The transformation is not made in-place.

        Returns:
            Tensor
        """
        return nets.flatten(self)

    def __repr__(self):
        string_data = np.array2string(self.data,
                                      prefix="       ",
                                      precision=4,
                                      separator=', ',
                                      floatmode='maxprec_equal',
                                      edgeitems=3,
                                      threshold=100,
                                      max_line_width=100)
        requires_grad = "" if not self.requires_grad else f", requires_grad={self.requires_grad}"
        return f"Tensor({string_data}{requires_grad})"

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

    def __setitem__(self, key, value):
        return nets.set(self, key, value)
