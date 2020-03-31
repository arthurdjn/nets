r"""
Defines a basic Recurrent Neural Network.
"""

import copy
from .rnnbase import RNNBase
from nets import Parameter
import nets


class RNN(RNNBase):
    """
    Recurrent neural network (RNN) is a type of neural network that has been successful in modelling sequential data,
    e.g. language, speech, protein sequences, etc.

    A RNN performs its computations in a cyclic manner, where the same computation is applied to every sample
    of a given sequence. The idea is that the network should be able to use the previous computations as some form
    of memory and apply this to future computations.
    From the [exercise 02456 from DTU course](https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch).
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Initialize all the weights (input - hidden - output)
        self.weight_ih = Parameter.orthogonal(shape=(input_dim, hidden_dim))
        self.weight_hh = Parameter.orthogonal(shape=(hidden_dim, hidden_dim))
        self.weight_ho = Parameter.orthogonal(shape=(hidden_dim, input_dim))
        # Initialize all the biases (hidden - output)
        self.bias_h = Parameter.zeros(shape=(hidden_dim,))
        self.bias_o = Parameter.zeros(shape=(input_dim,))
        # Initialize the first hidden cell
        self.hidden_0 = nets.zeros(shape=(1, hidden_dim))

    # TODO: deprecate this
    def set_hidden_0(self, hidden_cell):
        """Set the first hidden cell."""
        assert isinstance(hidden_cell, list), '``hidden_states`` should be a list containing ``Tensor`` objects.'
        self.hidden_0 = hidden_cell

    def forward(self, inputs):
        """
        Computes the forward pass of a vanilla RNN.

        Args:
         inputs (Tensor): sequence of inputs to be processed
        """
        hidden_states = nets.Tensor([])
        outputs = nets.Tensor([])
        # Initialize hidden_cell_0 (with zeros)
        hidden_state = self.hidden_0
        # For each element in input sequence
        for t in range(inputs.shape[0]):
            # Compute new hidden state
            hidden_state = nets.tanh(nets.dot(inputs[t], self.weight_ih) +
                                     nets.dot(hidden_state, self.weight_hh) + self.bias_h)
            # Compute output
            out = nets.sigmoid(nets.dot(hidden_state, self.weight_ho) + self.bias_o)
            # Save results and continue
            outputs = nets.append(outputs, out)
            hidden_states = nets.append(hidden_states, hidden_state)

        return outputs, hidden_states

    # TODO: manual backpropagation
    def backward(self, outputs):
        raise NotImplementedError
