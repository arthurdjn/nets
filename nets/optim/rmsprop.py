"""
Uses the RMSProp update rule, which uses a moving average of squared gradient values to set adaptive per-parameter
learning rates.
"""

from nets.optim.optimizer import Optimizer
import nets


class RMSprop(Optimizer):
    """
    Uses the RMSProp update rule, which uses a moving average of squared gradient values to set adaptive per-parameter
    learning rates.
    """
    def __init__(self, parameters, lr=1e-2, decay=0.99, epsilon=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.decay = decay
        self.epsilon = epsilon
        self._cache = {'velocity': [nets.zeros_like(p) for p in self.parameters]}

    def step(self):
        """Performs RMSProp update."""
        for i, parameter in enumerate(self.parameters):
            # Store a moving average f the gradients
            velocity = self._cache['velocity'][i]
            # Moving average
            velocity = self.decay * velocity + (1 - self.decay) * (parameter.grad ** 2)
            # Inplace update
            parameter -= self.lr * parameter.grad / (nets.sqrt(velocity) + self.epsilon)
            # Update the cache
            self._cache['velocity'][i] = velocity
