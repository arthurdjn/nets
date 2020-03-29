r"""
Modules are the main architecture for all transformations from one tensor to another. In other words,
a neural network is a succession of modules (layer, convolution, activation...).
When building a custom neural network, your model must inherits from ``Module`` abstract class and override the
``forward`` method. Moreover, you can specify the back-propagation rule in ``backward`` method. Usually, the
``backward`` method computes the naive back-propagation using only local gradients saved in the module's ``_cache``.
If you don't specify it, **NETS** will uses ``autograd`` functionality to compute all gradients.
"""

from abc import ABC, abstractmethod
import inspect
from nets import Parameter


class Module(ABC):
    """
    Abstract Module architecture. All models used to transform tensors should extends from this class to benefits
    ``forward`` and ``backward`` propagation rules.
    """
    def __init__(self):
        self.training = True
        self._modules = {}
        self._params = {}
        self._grads = {}
        self._cache = {}

    @abstractmethod
    def forward(self, *inputs):
        """One forward step. Gradients and outputs should be saved in the ``_cache`` when training, to be able to
        perform the backward pass.
        """
        raise NotImplementedError

    def backward(self, *outputs):
        """One backward step."""
        raise NotImplementedError

    def train(self):
        """Set the ``training`` attribute to training mode."""
        self.training = True

    def eval(self):
        """Set the ``training`` attribute to evaluation mode."""
        self.training = False

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
        """Zero grad all parameters within a module"""
        for parameter in self.parameters():
            parameter.zero_grad()

    def get_name(self):
        """Quick access to get the name of a module.

        Returns:
            string: module's name
        """
        return self.__class__.__name__

    def inner_repr(self):
        """Return the representation of a single module.
        This method should be unique for each modules.

        Returns:
            string: the representation of one module.
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

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def __setattr__(self, key, value):
        # First initialize the attribute we want to add
        self.__dict__[key] = value
        # Then update the inner dictionary '_modules', '_params'
        if isinstance(value, Module):
            self._modules[key] = value
        elif isinstance(value, Parameter):
            self._params[key] = value

