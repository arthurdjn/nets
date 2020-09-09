r"""
Loss functions evaluate the precision and correctness of a model's predictions. The most popular ones are ``MSE``,
``CrossEntropyLoss`` and ``Adam``. Each loss functions presents advantages, and results may vary when choosing one from
another.
"""

# Basic imports
from abc import ABC

# NETS package
import nets
from nets.nn.modules import Module


class Loss(Module, ABC):
    r"""A loss function evaluate the correctness of a set of predictions regarding gold-labels.
    The predictions should be un-corrected, *ie* no transformations like ``Softmax`` should have been used before.
    The loss function will do the transformation if necessary.
    The attribute ``history`` keeps track of the cost when the loss function is called.
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions, labels):
        r"""Compute the cross entropy cost function.

        Args:
            predictions (numpy.array): tensor of un-normalized floats with shape :math:`(N, c)`.
            labels (numpy.array): tensor of integer values with shape :math:`(N)`.

        Returns:
            cost (float): the cost regarding the loss function.
        """
        raise NotImplementedError

    def __call__(self, *inputs):
        cost = self.forward(*inputs)
        return cost


class MSELoss(Loss):
    r"""Mean Square Error Loss, defined as:

    .. math::

        \text{MSE} = \frac{1}{N}\sum_{i=1}^{c}(predictions - labels)^2

    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions, labels):
        assert labels.shape == predictions.shape, \
            "labels shape {} and predictions shape {} should match".format(labels.shape, predictions.shape)
        return nets.sum((predictions - labels) ** 2)

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
        super().__init__()

    def forward(self, predictions, labels):
        assert labels.dtype == int, "unsupported labels type {} for cross entropy loss".format(predictions.dtype)
        batch_size, _ = predictions.shape
        predictions = nets.softmax(predictions, axis=1)
        cost = nets.Tensor(- 1 / batch_size, device=predictions.device) * nets.sum(nets.log(predictions) * labels)
        return cost

    def backward(self, predictions, labels):
        assert labels.dtype == int, "unsupported labels type {} for cross entropy loss".format(predictions.dtype)
        predictions = nets.softmax(predictions)
        return predictions - labels
