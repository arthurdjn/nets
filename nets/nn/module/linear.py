r"""
Linear layers are widely used either for Dense Neural Module or Convolutional Neural Module, to cite the most
popular. The architecture is made of two sets of ``neurons`` aka ``perceptrons``, all connected with weights and biases.
"""

from .module import Module
from nets.nn.activation import *
from nets import Parameter
from nets.utils import deprecated


class Linear(Module):
    r"""
    A linear layer is made of a weight matrix :math:`W` of shape :math:`(\text{input_dim}, \text{output_dim})`
    and a bias vector :math:`b` of shape :math:`(\text{output_dim})`. The linear transformation for an incoming vector
    :math:`x` of shape :math:`(N, \text{input_dim})` results in a vector :math:`y` of shape :math:`(N, \text{output_dim})`:

    .. math::
        y = x W + b

    Examples::

        >>> model = Linear(5, 10)
        >>> batch_input = np.array([[-5, 2, 6, -2, 4],
        ...                         [2, 5, -6, 7, -3]])
        >>> batch_pred = model(batch_input)
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_params()
        # self.weight = Parameter(shape=(self.input_dim, self.output_dim))
        # self.bias = Parameter(shape=(self.output_dim,))

    @deprecated("weights and biases will be initialized in another function soon. "
                "This method will be then deprecated in the next update")
    def init_params(self, mode=None):
        r"""Initialize the parameters dictionary.
        For Dense Neural Module, the parameters are either weights or biases. They are saved in the dictionary
        `_params` with the following keys: weight matrix ``w``, bias vector ``b``.
        The initialization can be changed with ``mode`` parameter, between the default uniform :math:`\mathcal{U}(0, 1)`
        initialization or use :math:`\text{He et al.}  \quad \mathcal{N} (0, \frac{1}{input_dim})`.
        """
        if mode == 'uniform':
            self.weight = Parameter.uniform((self.input_dim, self.output_dim))
            self.bias = Parameter.uniform((self.output_dim,))
        else:
            mu = 0
            var = 2 / self.input_dim
            sigma = np.sqrt(var)
            weight_shape = (self.input_dim, self.output_dim)
            self.weight = Parameter(np.random.normal(loc=mu, scale=sigma, size=weight_shape))
            self.bias = Parameter(np.zeros((self.output_dim,)))

    def forward(self, x):
        r""" Forward pass. Compute the linear transformation and save the results in the `_cache`, which will be
        used during the backward pass.

        Shape:
            - input (numpy.array): batch inputs of shape.
            - output (numpy.array): results of the linear transformation,
                of shape :math:`(N, \text{input_dim})`, with :math:`N = \text{batch_size}`.
        """
        assert x.shape[1] == self.input_dim, 'dot product impossible with ' \
                                             'shape {} and {}'.format(x.shape, self._params['weight'].shape)
        # Linear combination
        z = nets.dot(x, self.weight) + self.bias
        # Keep track of inputs for naive back-propagation
        self._cache['x'] = x
        return z

    def backward(self, dout):
        r"""Backward pass for a single Linear layer.

        Shape:
            - input (Tensor): upstream gradient.
            - output (Tensor): downstream gradient after a linear transformation.
        """
        coef = 1 / dout.shape[0]  # Normalize by the batch_size
        # Get parameters
        x = self._cache['x']
        w = self.weight
        # Compute the gradients
        dw = coef * np.dot(x.T, dout)
        db = coef * np.sum(dout, axis=0)
        # Save the parameters' gradient
        self._grads['b'] = db
        self._grads['w'] = dw
        # Return the new downstream gradient
        dx = np.dot(dout, w.T)
        return dx

    def inner_repr(self):
        """Display the inner parameter of a Linear layer"""
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, " \
               f"bias={True if self.bias is not None else False}"
