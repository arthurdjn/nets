from abc import ABC, abstractmethod

# Data science
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import time
import json
import os
import copy

import nets
from nets.utils import append2dict, progress_bar, describe_stats


def get_metrics(gold_labels, predictions, labels=None):
    """Compute metrics for given predictions. This method returns basic
    metrics as a dictionary.

    Args:
        gold_labels (Tensor): true labels.
        predictions (Tensor): predicted labels.
        labels (list, optional): list of classes indices.

    Returns:
        dict

    """
    if isinstance(gold_labels, nets.Tensor):
        gold_labels = gold_labels.numpy()
    if isinstance(predictions, nets.Tensor):
        predictions = predictions.numpy()
    metrics = {"accuracy": float(accuracy_score(gold_labels, predictions)),
               "precision": float(precision_score(gold_labels, predictions, labels=labels, average='macro')),
               "recall": float(recall_score(gold_labels, predictions, labels=labels, average='macro')),
               "macro_f1": float(f1_score(gold_labels, predictions, labels=labels, average='macro')),
               "confusion_matrix": confusion_matrix(gold_labels, predictions, labels=labels).tolist()}
    return metrics


class Solver(ABC):
    r"""Train and evaluate models.

    Args:
        model (Module): model to optimize or test.
        criterion (Loss): loss function.
        optimizer (Optimizer): optimizer for weights and biases.

    Attributes::
        model (Module): model to optimize or test.
        checkpoint (dict): checkpoint of the best model tested.
        criterion (Loss): loss function.
        optimizer (Optimizer): optimizer for weights and biases.
        performance (dict): performance and scores accumulated during the training loop.
        
    """

    def __init__(self, model=None, criterion=None, optimizer=None):
        self.model = model
        self.best_model = None
        self.checkpoint = {'epoch': None,
                           'model_name': None,
                           'model_state_dict': None,
                           'optimizer_name': None,
                           'optimizer_state_dict': None,
                           'train': None,
                           'eval': None
                           }
        self.criterion = criterion
        self.optimizer = optimizer

        # Performances
        self.performance = None
        self.reset()

    def reset(self):
        """Reset the performance dictionary.

        Returns:
            None
        """
        self.performance = {"train": {"loss": [],
                                      "accuracy": [],
                                      "precision": [],
                                      "recall": [],
                                      "macro_f1": [],
                                      "confusion_matrix": []},
                            "eval": {"loss": [],
                                     "accuracy": [],
                                     "precision": [],
                                     "recall": [],
                                     "macro_f1": [],
                                     "confusion_matrix": []}}

    @abstractmethod
    def train(self, iterator, *args, **kwargs):
        r"""Train one time the model on iterator data.

        Args:
            iterator (Iterator): iterator containing batch samples of data.

        Returns:
            dict: the performance and metrics of the training session.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, iterator, *args, **kwargs):
        r"""Evaluate one time the model on iterator data.

        Args:
            iterator (Iterator): iterator containing batch samples of data.

        Returns:
            dict: the performance and metrics of the training session.
        """
        raise NotImplementedError

    def _update_checkpoint(self, epoch, results_train=None, results_eval=None):
        r"""Update the model's checkpoint. Keep track of its epoch, state, optimizer,
        and performances. In addition, it saves the current model in `best_model`.

        Args:
            epoch (int): epoch at the current training state.
            results_train (dict, optional): metrics for the training session at epoch. The default is None.
            results_eval (dict, optional): metrics for the evaluation session at epoch. The default is None.

        """
        self.best_model = copy.deepcopy(self.model)
        self.checkpoint = {'epoch': epoch,
                           'model_name': self.best_model.__class__.__name__,
                           'model_state_dict': self.best_model.state_dict(),
                           'optimizer_name': self.optimizer.__class__.__name__,
                           'criterion_name': self.criterion.__class__.__name__,
                           'train': results_train,
                           'eval': results_eval
                           }

    def save(self, filename=None, dirpath=".", checkpoint=True):
        r"""Save the best torch model.

        Args:
            filename (str, optional): name of the model. The default is "model.pt".
            dirpath (str, optional): path to the desired foldre location. The default is ".".

        Returns:
            None.
        """
        filename = 'model_' + self.checkpoint['model_name'] + '.pt' if filename is None else filename
        # Save the best model
        path = os.path.join(dirpath, filename)
        nets.save(self.best_model, path)

        # Save its checkpoint
        if checkpoint:
            checkname = 'checkpoint_' + filename.split('.')[-2].split('_')[-1] \
                        + '_epoch' + str(self.checkpoint['epoch']) + '.pt'
            checkpath = os.path.join(dirpath, checkname)
            nets.save(self.checkpoint, checkpath)

    def get_accuracy(self, y_tilde, y):
        r"""Compute accuracy from predicted classes and gold labels.

        Args:
            y_tilde (Tensor): 1D tensor containing the predicted classes for each predictions
                in the batch. This tensor should be computed through `get_predicted_classes(y_hat)` method.
            y (Tensor): gold labels. Note that y_tilde an y must have the same shape.

        Returns:
            float: the mean of correct answers.
        """
        assert y_tilde.shape == y.shape, ("predicted classes and gold labels should have the same shape")
        correct = (y_tilde == y).astype(float)  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc

    def run(self, epochs, train_iterator, eval_iterator, *args, **kwargs):
        r"""Train and evaluate a model X times. During the training, both training
        and evaluation results are saved under the `performance` attribute.

        Args:
            epochs (int): number of times the model will be trained.
            train_iterator (Iterator): iterator containing batch samples of data.
            eval_iterator (Iterator): iterator containing batch samples of data.
            verbose (bool, optional): if `True` display a progress bar and metrics at each epoch.
                The default is True.

        Returns:
            None

        Examples::

            >>> performer = MySolver(model, criterion, optimizer)
            >>> # Train & eval EPOCHS times
            >>> EPOCHS = 10
            >>> performer.run(EPOCHS, train_iterator, eval_iterator, verbose=True)
                Epoch:        1/10
                Training:     100% | [==================================================] | Time: 2m 26s
                Validation:   100% | [==================================================] | Time: 0m 4s
                Stats Training:    | Loss: 0.349 | Acc: 84.33% | Prec.: 84.26% | Rec.: 84.33% | F1: 84.26%
                Stats Evaluation:  | Loss: 0.627 | Acc: 72.04% | Prec.: 72.22% | Rec.: 72.17% | F1: 72.22%
            >>> # ...
        """
        # By default, print a log each epoch
        verbose = True if 'verbose' not in {*kwargs} else kwargs['verbose']
        # Keep track of the best model
        best_eval_accuracy = 0

        # Train and evaluate the model epochs times
        for epoch in range(epochs):
            if verbose:
                print("Epoch:\t{0:3d}/{1}".format(epoch + 1, epochs))

            # Train and evaluate the model
            results_train = self.train(train_iterator, *args, **kwargs)
            results_eval = self.evaluate(eval_iterator, *args, **kwargs)
            # Update the eval dictionary by adding the results at the
            # current epoch
            append2dict(self.performance["train"],
                        results_train)
            append2dict(self.performance["eval"],
                        results_eval)

            if verbose:
                print("\tStats Train: | " + describe_stats(results_train))
                print("\t Stats Eval: | " + describe_stats(results_eval))
                print()
            # We copy in memory the best model
            if best_eval_accuracy < self.performance["eval"]["accuracy"][-1]:
                best_eval_accuracy = self.performance["eval"]["accuracy"][-1]
                self._update_checkpoint(epoch + 1, results_train, results_eval)
