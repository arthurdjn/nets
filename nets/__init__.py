# Main package
from nets.tensor import Tensor, to_array, to_tensor
from nets.autograd.hook import Hook
from nets.autograd.parameter import Parameter
from nets.numeric import *

# Autograd package
# Import the functions compatible with automatic differentiation
from nets.autograd.ops import *
from nets.autograd.functions import *
from nets.autograd.numeric import *
