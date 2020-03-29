"""
Stochastic Gradient Descent is a popular optimizer for machine learning purposes.
"""


from nets.optim.optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer follows the parameters optimization:

    .. math:

        \text{updated_param}_{i} = \text{param}_{i} - \text{learning_rate} \times \text{gradient_param}_{i}

    """
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for parameter in self.parameters:
            parameter -= parameter.grad * self.lr
