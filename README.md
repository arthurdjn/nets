![NETS](img/nets.png)


**NETS** is a Python package for deep learning application and networks construction
from scratch using **numpy**
and an autograd system (you can switch to vanilla gradient update too).


# Get Started

### About
**NETS** is a vanilla Deep Learning framework, made using only **numpy**.
This project was first introduced as an assignment I made at the 
[University of Oslo](https://www.uio.no/studier/emner/matnat/ifi/IN5400/), which is similar to the second
assignment from [Stanford University](http://cs231n.stanford.edu/syllabus.html) if you are curious.

However, this project was rebuild to make it entirely *object-oriented like*.
In addition, the back-propagation and update rules where changed, using the **autograd** system.
**NETS** was highly inspired from [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/)

### Requirements

All packages within **NETS** are made from scratch, using mainly **numpy**. However, some additional 
packages can offer a better experience if installed (saving checkpoints and models for example).

- **numpy**
- **json** (Optional)
- **time** (Optional)
- **pandas** (Optional)
- **scipy** (Optional)
- **sklearn** (Optional)

### Installation

To install this package from [PyPi](https://pypi.org)

````css
$ pip install nets
````

or from this repository
````css
$ git clone https://github.com/arthurdjn/nets
$ cd nets
$ pip install .
````

### Documentation

The documentation and tutorials are in process and will be released soon. 
You will find some tutorials and application on how to get started or build a similar package.

# Example

**NETS** provides a basic neural network structure so you can create your own with numpy. You will need to
wrap your arrays in a ``Tensor`` class to keep track of the gradients.

![NETS](img/xor.gif)

## Building a model

A model is a ``Module``subclass, where biases, weights and parameters transformations are computed.
All modules have a ``forward`` method, that MUST be overwritten. 
This method will compute the forward propagation from an input tensor, and compute the transformation. 
If using the ``autograd`` system, no back-propagation need to be added. However, 
if you prefer to manually compute the gradients, you will need to override the ``backward`` method.

Your ``Model`` should inherits from the ``Module`` class and override
the ``forward`` method.

````python
import nets
import nets.nn as nn

class Model(nn.Module):
    """
    Create your own model.
    The attributes should be your submodels used during the forward pass.
    You don't have to necessary affect the activation function as an attribute, 
    unless you want to set a manual backward pass.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialization
        super().__init__() # Don't forget to add this line
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, inputs):
        # Forward pass
        out1 = nets.tanh(self.layer1(inputs))
        out2 = nets.tanh(self.layer2(out1))
        return self.layer3(out2)
    
model = Model(10, 100, 2)

# Let's check the architecture
model
````

Out:
````pycon
Model(
   (layer1): Linear(input_dim=10, output_dim=100, bias=True)
   (layer2): Linear(input_dim=100, output_dim=100, bias=True)
   (layer3): Linear(input_dim=100, output_dim=5, bias=True)
)
````

Again, this is really similar to what **PyTorch** offers.

# Tutorials

* 1 - [Getting Started with NETS](https://github.com/arthurdjn/nets/blob/master/Getting%20Started%20with%20NETS.ipynb) ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)((https://colab.research.google.com/github/arthurdjn/nets/blob/master/Getting%20Started%20with%20NETS.ipynb)

    This tutorial highlights the main part and modules of **NETS**.

* 1 - [Simple Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb)


* 2 - [Build an Autograd System with NumPy]()

    To be released
    
* 3 - [Build a Deep Learning Library From Scratch]()

    To be released
    
* 4 - [Build a CNN with NumPy]()

    To be released

# References

Here is a list of tutorials and lectures/assignment that helped to develop **NETS**

- [PyTorch documentation](https://pytorch.org)
- [PyTorch autograd tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Joel Grus autograd tutorial](https://github.com/joelgrus/autograd/tree/part06)
- [Joel Grus autograd live coding](https://www.youtube.com/watch?v=RxmBukb-Om4)
- [Stanford University cs231n 2nd assignment](http://cs231n.github.io/)
- [University of Oslo in5400 1st assignment](https://www.uio.no/studier/emner/matnat/ifi/IN5400/)