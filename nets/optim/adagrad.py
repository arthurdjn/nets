from nets.optim.optimizer import Optimizer
import nets


class Adagrad(Optimizer):
    """
    Adagrad optimizer. More sophisticated version of ``SGD``, with learning rate decay.
    """
    def __init__(self, parameters, lr=1e-2, epsilon=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.epsilon = epsilon
        self._cache = {'decay': [nets.zeros_like(p) for p in self.parameters]}

    def step(self):
        """Performs Adagrad update rules."""
        for i, parameter in enumerate(self.parameters):
            # Moving average
            decay = self._cache['decay'][i] ** 2
            # Inplace update
            parameter += self.lr * parameter.grad / (nets.sqrt(decay) + self.epsilon)
            # Update the cache
            self._cache['decay'][i] += decay
