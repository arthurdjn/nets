====
nets
====

**NETS** is divided in five main sub-packages:

- `autograd <nets.autograd>`_ refers to the autograd system used,

- `nn <nets.nn>`_ refers to the core of the package, *ie* deep network models,

- `optim <nets.optim>`_ refers to optimizers used during the backpropagation to update weights and biases,

- `data <nets.data>`_ refers to array container, like training and testing examples with their labels,

- `datasets <nets.datasets>`_ refers to popular machine learning libraries for image analysis and NLP.

====
nets
====

.. automodule:: nets.tensor
    :members:

=============
nets.autograd
=============

nets.autograd.hook
==================

.. automodule:: nets.autograd.hook
    :members:

nets.autograd.parameter
=======================

.. automodule:: nets.autograd.parameter
    :members:

nets.autograd.ops
=================

.. automodule:: nets.autograd.ops
    :members:



=======
nets.nn
=======

The main core of **NETS** is located in **nn** package.
The neural networks implemented are the most popular ones:

- `Linear Neural Network <nets.nn.linear>`_,

- `Dense Neural Network <nets.nn.dnn>`_,

- `Convolutional Neural Network <nets.nn.cnn>`_,

- `Recurent Neural Network <nets.nn.rnn>`_.


nets.nn.activation
====================

.. automodule:: nets.nn.activation
    :members:

nets.nn.linear
==============

.. automodule:: nets.nn.linear
    :members:

nets.nn.dnn
============

.. automodule:: nets.nn.dnn
    :members:

nets.nn.cnn
============

.. automodule:: nets.nn.cnn
    :members:

nets.nn.loss
=============

.. automodule:: nets.nn.loss
    :members:

nets.nn.utils
==============

.. automodule:: nets.nn.utils
    :members:



==========
nets.optim
==========

nets.optim.optimizer
====================

.. automodule:: nets.optim.optimizer
    :members:

nets.optim.sgd
==============

.. automodule:: nets.optim.sgd
    :members:



=============
nets.datasets
=============

nets.datasets.dataset
=====================

.. automodule:: nets.datasets.dataset
    :members:

nets.datasets.cifar
===================

.. automodule:: nets.datasets.cifar
    :members:

nets.datasets.mnist
===================

.. automodule:: nets.datasets.mnist
    :members:

nets.datasets.svhn
==================

.. automodule:: nets.datasets.svhn
    :members:



=====
utils
=====

nets.functional
===============

.. automodule:: nets.functional
    :members:

nets.utils
==========

.. automodule:: nets.utils
    :members: