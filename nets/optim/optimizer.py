from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Optimizer. Modify a model's parameters, and update its weights / biases."""

    def __init__(self, module, inplace=True):
        self.module = module
        self.parameters = module.parameters()
        self.gradients = module.gradients()
        self.inplace = inplace

    @abstractmethod
    def step(self):
        raise NotImplementedError