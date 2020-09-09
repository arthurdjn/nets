r"""
Modules are the main architecture for all transformations from one tensor to another. In other words,
a neural network is a succession of modules (layer, convolution, activation...).
When building a custom neural network, your model must inherits from ``Module`` abstract class and override the
``forward`` method. Moreover, you can specify the back-propagation rule in ``backward`` method. Usually, the
``backward`` method computes the naive back-propagation using only local gradients saved in the modules's ``_cache``.
If you don't specify it, **NETS** will uses ``autograd`` functionality to compute all gradients.
"""

from collections import OrderedDict
from abc import ABC, abstractmethod
import inspect
import warnings
import json
import pickle
from nets import Parameter


class Module(ABC):
    """
    Abstract Module architecture. All models used to transform tensors should extends from this class to benefits
    ``forward`` and ``backward`` propagation rules.
    """

    def __init__(self):
        self.training = True
        self._modules = OrderedDict()
        self._params = OrderedDict()
        self._grads = OrderedDict()
        self._cache = OrderedDict()

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        """One forward step. Gradients and outputs should be saved in the ``_cache`` when training, to be able to
        perform the backward pass.
        """
        raise NotImplementedError

    def backward(self, *outputs, **kwargs):
        """One backward step."""
        raise NotImplementedError

    def train(self):
        """Set the ``training`` attribute to training mode."""
        self.training = True
        for param in self.parameters():
            param.requires_grad = True

    def eval(self):
        """Set the ``training`` attribute to evaluation mode."""
        self.training = False
        for param in self.parameters():
            param.requires_grad = False

    def add(self, *modules):
        """Add modules to the current one.

        Args:
            modules (Module): modules to add
        """
        for module in modules:
            idx = len(self._modules)
            name = f"{idx}"
            setattr(self, name, module)
            self._modules[name] = module

    def parameters(self):
        """Iterator through all parameters"""
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def modules(self):
        """Iterator through all gradients"""
        yield from self._modules.values()

    def cache(self):
        """Iterator through all cache dict"""
        for module in self.modules():
            yield module._cache

    def gradients(self):
        """Iterator through all gradients"""
        for module in self.modules():
            yield module._grads

    def zero_grad(self):
        """Zero grad all parameters within a modules"""
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self):
        r"""Save all parameters in a dictionary."""
        state = OrderedDict()
        for i, param in enumerate(self.parameters()):
            state[f'param{i}'] = param.tolist()
        return state

    def load_state(self, state_dict):
        r"""Load parameters from a ``state_dict`` dictionary."""
        for i, param in self.parameters():
            data = state_dict[f'param{i}']
            if param.shape != data.shape:
                warnings.warn(f"shape from the `state_dict` does not match model's parameter shape. "
                              f"Got {data.shape}, expected {param.shape}.", UserWarning, stacklevel=2)
            param.data = Parameter(data=data)

    def save(self, filename='model.pickle'):
        """Save a model as a PICKLE file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def save_dict(self, filename='state_dict.json'):
        """Save the state as a JSON file."""
        state = self.state_dict()
        with open(filename, 'w') as f:
            json.dump(state, f)

    def cpu(self):
        """Move the location of the tensor to the CPU.

        Returns:
            Tensor
        """
        for parameter in self.parameters():
            parameter.cpu()

    def cuda(self):
        """Move the location of the tensor to the GPU.

        Returns:
            Tensor
        """
        for parameter in self.parameters():
            parameter.cuda()

    def get_name(self):
        """Quick access to get the name of a modules.

        Returns:
            string: modules's name
        """
        return self.__class__.__name__

    def inner_repr(self):
        """Return the representation of a single modules.
        This method should be unique for each modules.

        Returns:
            string: the representation of one modules.
        """
        return ""

    def __repr__(self):
        # Representation similar to PyTorch
        string = f"{self.get_name()}("
        tab = "   "
        modules = self._modules
        if modules == {}:
            string += f'\n{tab}(parameters): {self.inner_repr()}'
        else:
            for key, module in modules.items():
                string += f"\n{tab}({key}): {module.get_name()}({module.inner_repr()})"
        return f'{string}\n)'

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def __setattr__(self, key, value):
        # First initialize the attribute we want to add
        self.__dict__[key] = value
        # Then update the inner dictionary '_modules', '_params'
        if isinstance(value, Module):
            self._modules[key] = value
        elif isinstance(value, Parameter):
            self._params[key] = value
