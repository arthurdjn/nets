from abc import ABC, abstractmethod
import numpy as np
from nets.utils import one_hot


# class Loss(ABC):
#
#     def __init__(self):
#         self.cost_history = []
#         self.correct_history = []
#
#     @abstractmethod
#     def forward(self, *args, **kwargs):
#         raise NotImplementedError
#
#     def cost(self):
#         return self.cost_history[-1]
#
#     def correct(self):
#         return self.correct_history[-1]
#
#     def __call__(self, *args, **kwargs):
#         cost, correct = self.forward(*args, **kwargs)
#         self.cost_history.append(cost)
#         self.correct_history.append(correct)
#         return cost, correct
#
#
# class CrossEntropyLoss(Loss):
#
#     def __init__(self):
#         super(CrossEntropyLoss, self).__init__()
#
#     def forward(self, outputs, labels):
#         """Compute the cross entropy cost function.
#
#         Args:
#             outputs (numpy.array): numpy array of floats with shape (num_classes, batch_size).
#             labels (numpy.array): numpy array of floats with shape (num_classes, batch_size).
#                 Collection of one-hot encoded true input labels
#
#         Returns:
#             cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
#             num_correct: Scalar integer
#
#         """
#         num_classes, batch_size = outputs.shape
#         cost = - 1 / batch_size * np.sum(np.multiply(np.log(outputs), labels))
#
#         # Number of correct answers
#         predicted_classes = np.argmax(outputs, axis=0)
#         predicted_classes = one_hot(predicted_classes, num_classes)
#         num_correct = np.sum(predicted_classes * labels)
#
#         return cost, num_correct


class Loss:
    def loss(self, predicted, actual) -> float:
        raise NotImplementedError

    def grad(self, predicted, actual):
        raise NotImplementedError


class MSE(Loss):
    """
    MSE is mean squared error, although we're
    just going to do total squared error
    """
    def loss(self, predicted, actual) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted, actual):
        return 2 * (predicted - actual)