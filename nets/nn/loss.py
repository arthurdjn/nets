r"""
Loss functions evaluate the precision and correctness of a model's predictions. The most popular ones are ``MSE``,
``CrossEntropyLoss`` and ``Adam``. Each loss functions presents advantages, and results may vary when choosing one from
another.
"""

from abc import ABC, abstractmethod
import numpy as np

from nets.nn.module import Module
from scipy.special import softmax


class Loss(Module):
    r"""A loss function evaluate the correctness of a set of predictions regarding gold-labels.
    The predictions should be un-corrected, *ie* no transformations like ``Softmax`` should have been used before.
    The loss function will do the transformation if necessary.
    The attribute ``history`` keeps track of the cost when the loss function is called.
    """

    def __init__(self):
        self.history = []

    def forward(self, predictions, labels):
        r"""Compute the cross entropy cost function.

        Args:
            predictions (numpy.array): tensor of un-normalized floats with shape :math:`(N, c)`.
            labels (numpy.array): tensor of integer values with shape :math:`(N)`.

        Returns:
            cost (float): the cost regarding the loss function.
        """
        raise NotImplementedError

    def cost(self):
        return self.cost_history[-1]

    def correct(self):
        return self.correct_history[-1]

    def __call__(self, *inputs):
        cost = self.forward(*inputs)
        self.history.append(cost)
        return cost


class MSE(Loss):
    r"""Mean Square Error Loss, defined as:

    .. math::

        \text{MSE} = \frac{1}{N}\sum_{i=1}^{c}(predictions - labels)^2

    """

    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, predictions, labels):
        assert labels.shape == predictions.shape, \
            "labels shape {} and predictions shape {} should match".format(labels.shape, predictions.shape)
        return np.sum((predictions - labels) ** 2)

    def backward(self, predictions, labels):
        assert labels.shape == predictions.shape, \
            "labels shape {} and predictions shape {} should match".format(labels.shape, predictions.shape)
        return 2 * (predictions - labels)


class CrossEntropyLoss(Loss):
    r"""Cross Entropy Loss. First, a softmax transformation is used to map the predictions between :math:`[0, 1]`,
    then the cost is computed:

    .. math::

        \text{CrossEntropyLoss} = - \frac{1}{N} \sum_{i=1}^{c}labels_{i}\log(pred_{i})

    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, predictions, labels):
        assert labels.dtype == int, "unsupported labels type {} for cross entropy loss".format(predictions.dtype)
        num_classes, batch_size = predictions.shape
        predictions = softmax(predictions)
        cost = - 1 / batch_size * np.sum(np.multiply(np.log(predictions), labels))
        # Number of correct answers
        # predicted_classes = np.argmax(outputs, axis=0)
        # predicted_classes = one_hot(predicted_classes, num_classes)
        # num_correct = np.sum(predicted_classes * labels)
        return cost

    def backward(self, predictions, labels):
        assert labels.dtype == int, "unsupported labels type {} for cross entropy loss".format(predictions.dtype)
        predictions = softmax(predictions)
        return predictions - labels
