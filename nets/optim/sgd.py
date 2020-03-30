"""
Stochastic Gradient Descent is a popular optimizer for machine learning purposes.
"""

from nets.optim.optimizer import Optimizer
import nets


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer follows the parameters optimization:

    .. math:

        \text{updated_param}_{i} = \text{param}_{i} - \text{learning_rate} \times \text{gradient_param}_{i}

    .. note::
        A momentum can be added to this process.
    """
    def __init__(self, parameters, lr=1e-2, momentum=0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self._cache = {'velocity': [nets.zeros_like(p) for p in self.parameters]}

    def step(self):
        """Performs stochastic gradient descent with momentum."""
        for i, parameter in enumerate(self.parameters):
            # Store a moving average f the gradients
            velocity = self._cache['velocity'][i]
            # Moving average
            velocity = self.momentum * velocity - self.lr * parameter.grad
            # Inplace update
            parameter += velocity
            # Update the cache
            self._cache['velocity'][i] = velocity
