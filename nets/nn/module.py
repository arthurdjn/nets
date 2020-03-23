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
from nets.nn.utils import info_layer
from nets import Tensor, Parameter


class Module(ABC):
    """
    Abstract Module architecture. All models used to transform tensors should extends from this class to benefits
    ``forward`` and ``backward`` propagation rules.
    """
    def __init__(self, *args, **kwargs):
        self.training = True
        self._modules = {}
        self._params = {}
        self._grads = {}
        self._cache = {}

    @abstractmethod
    def forward(self, *args, **kwargs):
        """One forward step. Gradients and outputs should be saved in the ``_cache`` when training, to be able to
        perform the backward pass.
        """
        raise NotImplementedError

    @abstractmethod
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
        for module in modules:
            class_name = module.__class__.__name__.lower()
            idx = len(self._modules)
            name = f"{class_name}{idx}"
            setattr(self, name, module)
            self._modules[name] = module

    def parameters(self):
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def modules(self):
        return list(self._modules.values())

    def cache(self):
        for module in self.modules():
            yield module._cache

    def gradients(self):
        for module in self.modules():
            yield module._grads

    def __call__(self, *inputs):
        return self.forward(*inputs)


    # def info(self, tab=''):
    #     string = ""
    #     if self.modules() == {}:
    #         string += info_layer(self, tab=tab)
    #     else:
    #         for module in self.modules().values():
    #             if module.modules() == {}:
    #                 string += info_layer(module, tab=tab)
    #             else:
    #                 module.info()
    #     return string
    #
    # def __str__(self):
    #     """Display model's architecture information."""
    #     model_name = str(self.__class__.__name__)
    #     string = f"{model_name} Module:"
    #     string += '\n'
    #     string += self.info(tab='\t')
    #     return string
