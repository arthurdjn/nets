from nets.optim.optimizer import Optimizer
import nets


class Adam(Optimizer):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    """
    def __init__(self, parameters, lr=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._cache = {'velocity': [nets.zeros_like(p) for p in self.parameters],
                       'momentum': [nets.zeros_like(p) for p in self.parameters],
                       't': 0}

    def step(self):
        """Performs Adam update rules."""
        t = self._cache['t'] + 1
        for i, parameter in enumerate(self.parameters):
            # Store a moving average f the gradients
            velocity = self._cache['velocity'][i]
            momentum = self._cache['momentum'][i]
            # iter
            momentum = self.beta1 * momentum + (1 - self.beta1) * parameter.grad
            momentum_t = momentum / (1 - self.beta1 ** t)
            velocity = self.beta2 * velocity + (1 - self.beta2) * (parameter.grad ** 2)
            velocity_t = velocity / (1 - self.beta2 ** t)
            # Inplace update
            parameter -= self.lr * momentum_t / (nets.sqrt(velocity_t) + self.epsilon)
            # Update the cache
            self._cache['velocity'][i] = velocity
            self._cache['momentum'][i] = momentum
        self._cache['t'] = t
