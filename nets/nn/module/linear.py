r"""
Linear layers are widely used either for Dense Neural Module or Convolutional Neural Module, to cite the most
popular. The architecture is made of two sets of ``neurons`` aka ``perceptrons``, all connected with weights and biases.
"""

from .module import Module
from nets.nn.activation import *
from nets import Parameter


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
        super(Linear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.init_params()
        self.weight = Parameter(shape=(self.input_dim, self.output_dim))
        self.bias = Parameter(shape=(self.output_dim,))

    def init_params(self, mode=None):
        r"""Initialize the parameters dictionary.
        For Dense Neural Module, the parameters are either weights or biases. They are saved in the dictionary
        `_params` with the following keys: weight matrix ``w``, bias vector ``b``.
        The initialization can be changed with ``mode`` parameter, between the default uniform :math:`\mathcal{U}(0, 1)`
        initialization or use :math:`\text{He et al.}  \quad \mathcal{N} (0, \frac{1}{input_dim})`.
        """
        print('Deprecated.')
        # if mode == 'uniform':
        #     mu = 0
        #     var = 2 / self.output_dim
        #     sigma = np.sqrt(var)
        #     weight_shape = (self.input_dim, self.output_dim)
        #     self._params["w"] = Parameter(np.random.normal(loc=mu, scale=sigma, size=weight_shape))
        #     self._params["b"] = Parameter(np.zeros((self.output_dim,)))
        # else:
        #     self._params["w"] = Parameter(shape=(self.input_dim, self.output_dim))
        #     self._params["b"] = Parameter(shape=(self.output_dim,))
        pass

    def forward(self, x):
        r""" Forward pass. Compute the linear transformation and save the results in the `_cache`, which will be
        used during the backward pass.

        Shape:
            - input (numpy.array): batch inputs of shape.
            - output (numpy.array): results of the linear transformation,
                of shape :math:`(N, \text{input_dim})`, with :math:`N = \text{batch_size}`.
        """
        assert x.shape[1] == self.input_dim, 'dot product impossible with ' \
                                             'shape {} and {}'.format(x.shape, self._params['w'].shape)
        # Linear combination
        z = x @ self.weight + self.bias
        # Keep track of inputs for naive back-propagation
        self._cache['x'] = x
        return z

    def backward(self, grad):
        r"""Backward pass for a single Linear layer.

        Shape:
            - input (Tensor): upstream gradient.
            - output (Tensor): downstream gradient after a linear transformation.
        """
        coef = 1 / grad.shape[0]  # Normalize by the batch_size
        # Get parameters
        x = self._cache['x']
        w = self.weight
        # Compute the gradients
        dz = grad
        dw = coef * np.dot(x.T, dz)
        db = coef * np.sum(dz, axis=0)
        # Save the parameters' gradient
        self._grads['b'] = db
        self._grads['w'] = dw
        # Return the new downstream gradient
        grad = np.dot(dz, w.T)
        return grad

    def inner_repr(self):
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, bias={True if self.bias is not None else False}"


class Sequential(Module):
    r"""
    Sequential models are an ordered succession of modules.
    """

    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.add(*modules)

    def forward(self, inputs):
        r"""Compute the forward pass for all modules within the sequential module.

        Shape:
            - inputs (Tensor): incoming data.
            - outputs (Tensor): result of all forward pass.
        """
        for module in self.modules():
            inputs = module.forward(inputs)
        return inputs

    def backward(self, grad):
        r"""Vanilla backward pass. This pass computes local gradients from ``parameters`` saved in its ``_cache``.

        Shape:
            - inputs (Tensor): upstream gradient. The first downstream gradient is usually the ``loss``.
            - outputs (Tensor): last downstream gradient.
        """
        for module in reversed(self.modules()):
            grad = module.backward(grad)
        return grad
