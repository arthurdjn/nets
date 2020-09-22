import numpy as np
from .module import Module
from nets.nn.activation import *


class DNN(Module):
    """
    Dense Neural Network.

    Attributes:
        layer_dimensions (list(int)): ``list`` of ``int`` containing all layer dimensions. The dimension at index ``i``
            is the value of ``layer_dimensions`` at this index.
        hidden_dimensions (list(int)): ``list`` of ``int`` containing only hidden dimensions.
        training (bool): Boolean to indicate if we are training or not. This function can namely be
            used for inference only, in which case we do not need to store the features
            values.
        activation_hidden (Activation): activation function used in hidden layers.
        activation_output (Activation): activation function used in the last layer.
        _parameters (dict): ``dict`` containing values of weights and biases for each layers.
            - ``weight_{i}``: contains the weight matrix of layer ``i``
            - ``bias_{i}``: contains the bias array of layer ``i``
        _cache (dict): ``dict`` with
            - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
            - the activations A^[l] = activation(Z^[l]) for l in [1, L].
            We cache them in order to use them when computing gradients in the back propagation.

    """

    def __init__(self, layer_dimensions, activation_hidden=ReLU()):
        super().__init__()
        # Add layers
        self.layer_dimensions = layer_dimensions
        self.hidden_dimensions = layer_dimensions[1: -1]
        # Add activation functions
        assert isinstance(activation_hidden, Activation), "unrecognized activation function type"
        self.activation_hidden = activation_hidden
        self.activation_output = Softmax(axis=1)
        # Add weights and biases
        self.init_parameters()

    def init_parameters(self):
        """Initialize the parameters dictionary.
        For Dense Neural Network, the parameters are either weights :math:`W` or biases :math:`b`.

        """
        # Create the weights and biases
        for i in range(1, len(self.layer_dimensions)):
            # Initialization from He et al.
            mu = 0
            var = 2 / self.layer_dimensions[i]
            sigma = np.sqrt(var)
            weight_shape = (self.layer_dimensions[i - 1], self.layer_dimensions[i])
            weight = np.random.normal(loc=mu, scale=sigma, size=weight_shape)
            bias = np.zeros((self.layer_dimensions[i], ))

            # Saving in the parameters dict
            layer_weight = "w_" + str(i)
            self._parameters[layer_weight] = weight
            layer_b = "b_" + str(i)
            self._parameters[layer_b] = bias

    def forward(self, inputs):
        """One forward step.

        Args:
            inputs (numpy.array): float numpy array with shape (n^[0], batch_size). Input image batch.

        Returns:
            outputs (numpy.array): float numpy array with shape (n^[L], batch_size). The output predictions of the
                network, where n^[L] is the number of prediction classes. For each input i in the batch,
                output[c, i] gives the probability that input ``i`` belongs to class ``c``.

        """
        depth = len(self.layer_dimensions) - 1
        z = inputs
        # Add the outputs to the features
        if self.training:
            layer_a = "a_0"
            self._cache[layer_a] = z
        # 1/ Iterates through the depth of the neural network
        for i in range(1, depth + 1):
            # 1.1/ Get the weights and biases from the params
            layer_w = "w_" + str(i)
            layer_b = "b_" + str(i)
            weight = self._parameters[layer_w]
            bias = self._parameters[layer_b]
            # 1.2/ Compute the outputs
            z = np.dot(z, weight) + bias
            # Add the outputs to the features
            if self.training:
                layer_z = "z_" + str(i)
                self._cache[layer_z] = z
            # 2/ Hidden Activation
            # The activation only occurs in the hidden layers
            if i < depth:
                z = self.activation_hidden(z)
                # Add the outputs to the features
                if self.training:
                    layer_a = "a_" + str(i)
                    self._cache[layer_a] = z
        # 3/ Output
        outputs = self.activation_output(z)
        # Add the outputs to the features
        if self.training:
            layer_a = "a_" + str(depth)
            self._cache[layer_a] = outputs

        return outputs

    def backward(self, outputs, labels):
        """Update parameters using backpropagation algorithm.

        Args:
            outputs (numpy.array): matrix of floats with shape (num_classes, batch_size).
            labels (numpy.array): numpy array of integers with shape (num_classes, batch_size).
                Collection of one-hot encoded true input labels.

        Returns:
            grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                    - the gradient of the weights, grad_W^[l] for l in [1, L].
                    - the gradient of the biases grad_b^[l] for l in [1, L].
        """
        # Layers & shape
        depth = len(self.layer_dimensions) - 1
        # num_classes, batch_size = outputs.shape
        batch_size, num_classes = outputs.shape
        coefficient = 1 / batch_size
        # 1/ First case : last layer -> output
        layer_a = "a_" + str(depth - 1)
        a = self._cache[layer_a]
        Jz = outputs - labels
        # Weights gradients
        dw = coefficient * np.dot(a.T, Jz)
        db = coefficient * np.sum(Jz, axis=0)
        self._grad["dw_" + str(depth)] = dw
        self._grad["db_" + str(depth)] = db
        # 2/ Second case : inside the layers
        for i in range(depth - 1, 0, -1):
            # Get the weights and biases
            layer_w = "w_" + str(i + 1)
            layer_a = "a_" + str(i - 1)
            layer_z = "z_" + str(i)
            w = self._parameters[layer_w]
            a = self._cache[layer_a]
            z = self._cache[layer_z]
            # Gradients
            Jz = self.activation_hidden.backward(z) * np.dot(Jz, w.T)
            db = coefficient * np.sum(Jz, axis=0)
            dw = coefficient * np.dot(a.T, Jz)
            self._grad["dw_" + str(i)] = dw
            self._grad["db_" + str(i)] = db




















