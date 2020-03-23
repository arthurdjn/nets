![NETS](docs/source/images/nets.png)

**NETS** is a Python package for deep learning application and networks construction
from scratch using **numpy**
and an autograd system (you can switch to vanilla gradient update too).


# Getting Started

### About
**NETS** is a vanilla Deep Learning framework, made using only **NumPy**.
This project was first introduced as an assignment I made at the 
[University of Oslo](https://www.uio.no/studier/emner/matnat/ifi/IN5400/), which is similar to the second
assignment from [Stanford University](http://cs231n.stanford.edu/syllabus.html).

However, this project was rebuild to make it entirely *object-oriented like*.
Moreover, the back-propagation and update rules where changed, using the **autograd** system.
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

To install this package from [PyPi](https://pypi.org):

````css
$ pip install nets
````

or from this repository:
````css
$ git clone https://github.com/arthurdjn/nets
$ cd nets
$ pip install .
````

### Documentation

The documentation and tutorials can be found at [ReadTheDocs](http://nets.readthedocs.io). 
You will find some tutorials and application on how to get started or build a similar package.

