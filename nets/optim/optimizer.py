from abc import ABC, abstractmethod
from nets import Tensor


class Optimizer(ABC):
    """Optimizer. Modify a model's parameters, and update its weights / biases."""

    def __init__(self, parameters):
        if isinstance(parameters, Tensor):
            raise TypeError("parameters should be an iterable, got {}".format(type(parameters)))
        elif isinstance(parameters, dict):
            parameters = parameters.values()
        params = list(parameters)
        self.parameters = params

    @abstractmethod
    def step(self):
        """Update the parameters. Should be used only with ``autograd`` system"""
        raise NotImplementedError

    def zero_grad(self):
        """Zero grad all parameters contained in ``parameters`` attribute.

        Returns:
            None
        """
        for parameter in self.parameters:
            parameter.zero_grad()

    def backward(self):
        """Update rules without ``autograd``"""
        raise NotImplementedError