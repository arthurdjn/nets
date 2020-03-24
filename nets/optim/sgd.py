from nets.optim.optimizer import Optimizer


# class SGD(Optimizer):
#     """Vanilla Stochastic Gradient Descent."""
#
#     def __init__(self, network, learning_rate=0.01, weight_decay=0.1, **kwargs):
#         super().__init__(network, **kwargs)
#         self.learning_rate = learning_rate
#
#
#     def step(self):
#         """Update the parameters in params according to the gradient descent update routine.
#
#         Returns:
#             params: Updated parameters dictionary.
#
#         """
#         # Update for every key element in the params file
#         for key in self.parameters:
#             self.parameters[key] = self.parameters[key] - self.learning_rate * self.gradients["d" + key]
#
#         # Update for every key element in the params file
#         for network in self.network._networks.values():
#             parameters = network.parameters()
#             gradients = network.gradients()
#             for key in parameters:
#                 inner_key = key.split('_')[0]
#                 parameters[key] = parameters[inner_key] - self.learning_rate * gradients["d" + inner_key]


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer follows the parameters optimization:

    .. math:
        \text{updated_param}_{i} = \text{param}_{i} - \text{learning_rate} \times \text{gradient_param}_{i}
    """
    def __init__(self, module, lr=0.01):
        super().__init__(module)
        self.lr = lr

    def step(self, module):
        for parameter in module.parameters():
            parameter -= parameter.grad * self.lr

    # def step(self):
    #     for module in self.module.modules():
    #         for key in module._params:
    #             module._params[key] = module._params[key] - self.lr * module._grads[key]


