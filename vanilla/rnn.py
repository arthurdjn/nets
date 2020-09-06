r"""
Defines a vanilla Recurrent Neural Network (RNN).
"""

from nets.nn.modules.rnnbase import RNNBase
from nets import Parameter
import nets


# TODO: batch template (batch_size, seq_length, inputs_length)
class RNN(RNNBase):
    """
    Recurrent neural network (RNN) is a type of neural network that has been successful for modelling sequential data,
    e.g. language, speech, protein sequences, etc.

    A RNN performs its computations in a cyclic manner, where the same computation is applied to every sample
    of a given sequence. The idea is that the network should be able to use the previous computations as some form
    of memory and apply this to future computations.
    From the `exercise 02456 from DTU course <https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch>`__.

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
        self.weight_h0 = nets.zeros(shape=(1, hidden_dim))

    def forward(self, inputs):
        r"""
        Computes the forward pass of a vanilla RNN.

        .. math::

            h_{0} = 0

            h_t = \text{tanh}(x W_{ih} + h_{t-1} W_{hh} + b_{h})

            y = h_t W_{ho} + b_{o}


        Args:
            inputs (Tensor): sequence of inputs to be processed

         Returns:
             outputs (Tensor): predictions :math:`y`.
             hidden_states (Tensor): concatenation of all hidden states :math:`h_t`.
        """
        hidden_states = nets.Tensor([self.weight_h0])
        outputs = nets.Tensor([])
        # Initialize hidden_cell_0 (with zeros)
        hidden_state = hidden_states[0]
        # For each element in input sequence
        for t in range(inputs.shape[0]):
            # Compute new hidden state
            hidden_state = nets.tanh(nets.dot(inputs[t], self.weight_ih) +
                                     nets.dot(hidden_state, self.weight_hh) + self.bias_h)
            # Compute output
            out = nets.sigmoid(
                nets.dot(hidden_state, self.weight_ho) + self.bias_o)
            # Save results and continue
            outputs = nets.append(outputs, out)
            hidden_states = nets.append(hidden_states, hidden_state)

        # Save in the cache (for manual back-propagation)
        self._cache['hidden_states'] = hidden_states

        return outputs, hidden_states

    def backward(self, dout):
        """
        Computes the backward pass of a vanilla RNN.
        Save gradients parameters in the ``_grads`` parameter.

        Args:
            dout (Tensor): upstream gradient.

        Returns:
            Tensor: downstream gradient
        """
        # Initialize gradients as zero
        dw_ih = nets.zeros_like(self.weight_ih)
        dw_hh = nets.zeros_like(self.weight_hh)
        dw_ho = nets.zeros_like(self.weight_ho)
        db_h = nets.zeros_like(self.bias_h)
        db_o = nets.zeros_like(self.bias_o)
        # Get the cache
        hidden_states = self._cache['hidden_states']
        inputs = self._cache['x']

        # Keep track of hidden state derivative and loss
        dh_t = nets.zeros_like(hidden_states[0])

        # For each element in output sequence
        # NB: We iterate backwards s.t. t = N, N-1, ... 1, 0
        for t in reversed(range(dout.shape[0])):
            # Back-propagate into output sigmoid
            do = nets.sigmoid_prime(dout[t])
            db_o += do
            # Back-propagate into weight_ho
            dw_ho += nets.dot(hidden_states[t].T, do)
            # Back-propagate into h_t
            dh = nets.dot(do, self.weight_ho.T) + dh_t
            # Back-propagate through non-linearity tanh
            df = nets.tanh_prime(hidden_states[t]) * dh
            db_h += df
            # Back-propagate into weight_ih
            dw_ih += nets.dot(inputs[t].T, df)
            # Back-propagate into weight_hh
            dw_hh += nets.dot(hidden_states[t - 1].T, df)
            dh_t = nets.dot(df, self.weight_hh.T)

        # TODO: dx grad
        # dx = nets.dot(dout, self.weight_ih)

        # Save gradients
        self._grads["weight_ih"] = dw_ih
        self._grads["weight_hh"] = dw_hh
        self._grads["weight_ho"] = dw_ho
        self._grads["bias_h"] = db_h
        self._grads["bias_o"] = db_o
