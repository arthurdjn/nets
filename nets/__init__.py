# Main package
from nets.tensor import Tensor, to_numpy, to_cupy, to_tensor, tensor2string
from nets.numeric import *

from nets.autograd import Hook, Parameter

# Autograd package
# Import the functions compatible with automatic differentiation
from nets.autograd.functional import *
