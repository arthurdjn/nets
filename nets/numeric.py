"""
The ``functional`` modules defines basic functions to generate tensors and transform data.
"""

# Basic imports
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    pass

# NETS Package
import nets
from nets.cuda import numpy_or_cupy
from nets.data import dataset


def set(t, key, value):
    r"""Set new value(s) to a tensor.

    .. warning::
        Setting manually values of a tensor will invalidate its gradients.

    Args:
        t (Tensor): tensor to compare.
        key (scalar or tensor): indices to set.
        value (scalar or tensor): new values.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    value = nets.to_tensor(value)
    # To device
    if t.device == 'cpu' and value.device != 'cpu':
        t.cuda()
    elif value.device == 'cpu' and t.device != 'cpu':
        value.cuda()
    cpu = True
    if isinstance(key, tuple):
        for k in key:
            if isinstance(k, nets.Tensor):
                if k.device != 'cpu':
                    cpu = False
    if not cpu:
        t.cuda()
        value.cuda()
        for k in key:
            if isinstance(k, nets.Tensor):
                k.cuda()
        

    if isinstance(key, nets.Tensor):
        key = key.data
    elif isinstance(key, tuple):
        keys = []
        for k in key:
            if isinstance(k, nets.Tensor):
                keys.append(k.data)
            else:
                keys.append(k)
        key = tuple(keys)
    t.data[key] = value.data

    # Setting a tensor invalidate its gradient
    t.detach()

    return t


def gt(t, other):
    r"""Return a boolean tensor for *greater than* condition.

    .. math::
        \text{gt}_{\text{other}}(t) = t > other

    Args:
        t (Tensor): tensor to compare.
        other (Tensor like): object to compare the tensor.

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    other = nets.to_tensor(other)
    data = t.data > other.data
    return nets.Tensor(data, device=t.device)


def ge(t, other):
    r"""Return a boolean tensor for *greater or equal* condition.

    .. math::
        \text{gt}_{\text{other}}(t) = t \ge other

    Args:
        t (Tensor): tensor to compare
        other (Tensor like): object to compare the tensor

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    other = nets.to_tensor(other)
    data = t.data >= other.data
    return nets.Tensor(data, device=t.device)


def lt(t, other):
    r"""Return a boolean tensor for *lower than* condition.

    .. math::
        \text{gt}_{\text{other}}(t) = t < other

    Args:
        t (Tensor): tensor to compare
        other (Tensor like): object to compare the tensor

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    other = nets.to_tensor(other)
    data = t.data < other.data
    return nets.Tensor(data, device=t.device)


def le(t, other):
    r"""Return a boolean tensor for *lower or equal* condition.

    .. math::
        \text{gt}_{\text{other}}(t) = t \le other

    Args:
        t (Tensor): tensor to compare
        other (Tensor like): object to compare the tensor

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    other = nets.to_tensor(other)
    data = t.data <= other.data
    return nets.Tensor(data, device=t.device)


def eq(t, other):
    r"""Return a boolean tensor for *equal* condition.

    .. math::
        \text{gt}_{\text{other}}(t) = t == other

    Args:
        t (Tensor): tensor to compare
        other (Tensor like): object to compare the tensor

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    other = nets.to_tensor(other)
    cond = t.data == other.data
    return nets.Tensor(cond, device=t.device)


def ne(t, other):
    r"""Return a boolean tensor for *not equal* condition.

    .. math::
        \text{gt}_{\text{other}}(t) = t not other

    Args:
        t (Tensor): tensor to compare
        other (Tensor like): object to compare the tensor

    Returns:
        Tensor
    """
    t = nets.to_tensor(t)
    other = nets.to_tensor(other)
    data = not t.data == other.data
    return nets.Tensor(data, device=t.device)


def unravel_index(indices, shape, order='C', requires_grad=False, device='cpu'):
    """Converts a flat index or array of flat indices into a tuple of coordinate arrays.   

    Args:
        indices ([type]): An integer array whose elements are indices into the flattened version of an array of dimensions shape. 
        shape ([type]): The shape of the tensor to use for unraveling indices.
        order (str, optional): Determines whether the indices should be viewed as indexing 
            in row-major (C-style, ``'C'``) or column-major (Fortran-style, ``'F``) order. Defaults to ``'C'``.
        requires_grad (bool): if ``True`` will track gradients.
        device (str): name of the device where the tensor is located. Default to ``'cpu'``.

    Returns:
        Tensor
    """
    if device == 'cpu':
        data = np.unravel_index(indices, shape, order=order)
    else:
        data = cp.unravel_index(indices, shape, order=order)
    return nets.Tensor(data, requires_grad=requires_grad, device=device)


def rollaxis(t, axis, start=0):
    """Roll the specified axis backwards, until it lies in a given position.

    Args:
        t (Tensor): Input tensor.
        axis (int): The axis to be rolled. 
            The positions of the other axes do not change relative to one another.
        start (int, optional): When ``start <= axis``, the axis is rolled back until it lies in this position. 
            When ``start > axis``, the axis is rolled until it lies before this position. 
            The default, 0, results in a "complete" roll.

    Returns:
        Tensor
    """
    nc = numpy_or_cupy(t)
    data = nc.rollaxis(t.data, axis, start=start)
    return nets.Tensor(data, requires_grad=t.requires_grad, device=t.device)


def zeros(shape, requires_grad=False, device='cpu', **kwargs):
    """Create a zeros tensor of a given shape.

    Args:
        shape (tuple): shape of the 0-tensor.
        requires_grad (bool): if ``True`` will track gradients.
        device (str): name of the device where the tensor is located. Default to ``'cpu'``.

    Returns:
        Tensor
    """
    if device == 'cpu':
        data = np.zeros(shape, **kwargs)
    else:
        data = cp.zeros(shape, **kwargs)
    return nets.Tensor(data, requires_grad=requires_grad, device=device)


def zeros_like(t, **kwargs):
    """Create a zeros tensor as the same shape of a given tensor.

    Args:
        tensor (Tensor): [description]

    Returns:
        Tensor
    """
    return zeros(t.shape, requires_grad=t.requires_grad, device=t.device, **kwargs)


def ones(shape, requires_grad=False, device='cpu', **kwargs):
    """Create a ones tensor of a given shape.

    Args:
        shape (tuple): shape of the 0-tensor.
        requires_grad (bool): if ``True`` will track gradients.
        device (str): name of the device where the tensor is located. Default to ``'cpu'``.

    Returns:
        Tensor
    """
    if device == 'cpu':
        data = np.ones(shape, **kwargs)
    else:
        data = cp.ones(shape, **kwargs)
    return nets.Tensor(data, requires_grad=requires_grad, device=device)


def ones_like(t, **kwargs):
    """Create a ones tensor as the same shape of a given tensor.

    Args:
        tensor (Tensor): [description]

    Returns:
        Tensor
    """
    return ones(t.shape, requires_grad=t.requires_grad, device=t.device, **kwargs)


def eye(size, requires_grad=False, device='cpu', **kwargs):
    """Create an eye matrix.

    Args:
        size (int): size of the matrix.
        requires_grad (bool): if ``True`` will track gradients.
        device (str): name of the device where the tensor is located. Default to ``'cpu'``.

    Returns:
        Tensor
    """
    if device == 'cpu':
        data = np.eye(size, **kwargs)
    else:
        data = cp.eye(size, **kwargs)
    return nets.Tensor(data, requires_grad=requires_grad, device=device)


def identity(size, requires_grad=False, device='cpu', **kwargs):
    """Create an identity matrix.

    Args:
        size (int): size of the matrix.
        requires_grad (bool): if ``True`` will track gradients.
        device (str): name of the device where the tensor is located. Default to ``'cpu'``.

    Returns:
        Tensor
    """
    if device == 'cpu':
        data = np.identity(size, **kwargs)
    else:
        data = cp.identity(size, **kwargs)
    return nets.Tensor(data, requires_grad=requires_grad, device=device, **kwargs)


def arange(*args, requires_grad=False, device='cpu', **kwargs):
    """Create a range of values.

    Args:
        requires_grad (bool): if ``True`` will track gradients.
        device (str): name of the device where the tensor is located. Default to ``'cpu'``.

    Returns:
        Tensor
    """
    if device == 'cpu':
        data = np.arange(*args, **kwargs)
    else:
        data = cp.arange(*args, **kwargs)

    return nets.Tensor(data, requires_grad=requires_grad, device=device)


def astype(t, new_type):
    """Create a range of values.

    Args:
        new_type (str): new type of the data.

    Returns:
        Tensor
    """
    data = t.data.astype(new_type)
    return nets.Tensor(data, requires_grad=t.requires_grad, device=t.device)
