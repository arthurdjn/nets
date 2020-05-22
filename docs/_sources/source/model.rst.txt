
========
Creation
========

Data Processing
===============

The dataset can be found in the *data* folder, called *signal_20_obligatory1_train.tsv.gz*.

The dataset is made of Part of Speech (POS) tags for every words, meaning that the type ("NOUN", "VERB", "ADJ" etc.) are stacked on each words with an underscore.
A first step was then to split the words in half, to keep only the word and not its type.
Then, the BoW and the vocabulary can be created.


These functionalities are coded in the *pynews.data* module. 



Feature Tuning
==============

Before changing the structure of the model, we explored differents Bag of Words features implementation
by varying the vocabulary size, the preprocessing and building the vocabulary before and after the split.
For example, with a vocabulary size of 4000 we did not see improvement in the performance. In addition,
using the Part of Speech (POS) tags did not help to optimize the results. Because of difficulties and time
restriction we did not create the vocabulary after splitting into training and development parts.


Hyperparameters
===============

A brief training session to evaluate the performance with different hyper parameters was firstly performed.
The hyper parameters used are described on the table below.

.. table:: Hyperparameters

    +-------------------+-----------+
    |Parameter          |Value      |
    +-------------------+-----------+
    |Split Train/Dev    | .9        |
    +-------------------+-----------+
    |Vocabulary         | 3000      |
    +-------------------+-----------+
    |Batch size         | 32        |
    +-------------------+-----------+
    |Learning Rate      | .09       |
    +-------------------+-----------+
    |Epochs             | 250       |
    +-------------------+-----------+


Architecture
============

Then, five different models were trained on Saga’s server with different layout. These models differ in their
number of hidden layers, and their architectures are presented in the tables below.


.. table:: First Model

    +-----------+-----------+------------+
    | Layers    | Neurons   | Activation |     
    +-----------+-----------+------------+  
    | Input     | 3000      | ReLU       |
    +-----------+-----------+------------+
    | Hidden    | 150       | Linear     |
    +-----------+-----------+------------+
    | Output    | 20        | Softmax    |
    +-----------+-----------+------------+


.. table:: Second Model

    +-----------+-----------+------------+
    | Layers    | Neurons   | Activation |     
    +-----------+-----------+------------+  
    | Input     | 3000      | ReLU       |
    +-----------+-----------+------------+
    | Hidden 1  | 150       | ReLU       |
    +-----------+-----------+------------+
    | Hidden 2  | 150       | Linear     |
    +-----------+-----------+------------+
    | Output    | 20        | Softmax    |
    +-----------+-----------+------------+


.. table:: Third Model

    +-----------+-----------+------------+
    | Layers    | Neurons   | Activation |     
    +-----------+-----------+------------+  
    | Input     | 3000      | ReLU       |
    +-----------+-----------+------------+
    | Hidden 1  | 150       | ReLU       |
    +-----------+-----------+------------+
    | Hidden 2  | 150       | ReLU       |
    +-----------+-----------+------------+
    | Hidden 3  | 150       | Linear     |
    +-----------+-----------+------------+
    | Output    | 20        | Softmax    |
    +-----------+-----------+------------+


.. table:: Fourth Model

    +-----------+-----------+------------+
    | Layers    | Neurons   | Activation |     
    +-----------+-----------+------------+  
    | Input     | 3000      | ReLU       |
    +-----------+-----------+------------+
    | Hidden 1  | 150       | ReLU       |
    +-----------+-----------+------------+
    | Hidden 2  | 150       | ReLU       |
    +-----------+-----------+------------+
    | Hidden 3  | 150       | ReLU       |
    +-----------+-----------+------------+
    | Hidden 4  | 150       | Linear     |
    +-----------+-----------+------------+  
    | Output    | 20        | Softmax    |
    +-----------+-----------+------------+


.. table:: Fifth Model

    +-----------+-----------+------------+
    | Layers    | Neurons   | Activation |     
    +-----------+-----------+------------+  
    | Input     | 3000      | ReLU       |
    +-----------+-----------+------------+
    | Hidden 1  | 150       | ReLU       |
    +-----------+-----------+------------+
    | Hidden 2  | 150       | ReLU       |
    +-----------+-----------+------------+
    | Hidden 3  | 150       | ReLU       |
    +-----------+-----------+------------+
    | Hidden 4  | 150       | ReLU       |
    +-----------+-----------+------------+
    | Hidden 5  | 150       | Linear     |
    +-----------+-----------+------------+
    | Output    | 20        | Softmax    |
    +-----------+-----------+------------+

==========
Evaluation
==========

After we trained all models, there was no specific model that stood out. However, we tried to differenciate
them regarding four indicators : the accuracy, macro-F1, precision and recall.

Result
======

As shown in the table below, the model 4 presents the best accuracy. Nevertheless, this might not be the
only criterion to consider, especially because of the different sources frequency. The Macro-F1 score might be
less sensitive to imbalanced class frequencies. In that case, the model 3 performs better even if its precision is
not optimal. Due to these performance, we choose the model 3 to push further the training.


.. table:: Models performance

    +-----------+-----------+-----------+-----------+-----------+-----------+
    | Model     | Accuracy  | Macro-F1  | Precision | Recall    | Run Time  |
    +-----------+-----------+-----------+-----------+-----------+-----------+
    | Model 1   | 53.33     | 38.32     | 39.04     | 42.11     | 00:22:27  |
    +-----------+-----------+-----------+-----------+-----------+-----------+
    | Model 2   | 52.35     | 32.93     | 36.84     | 31.58     | 00:25:43  |
    +-----------+-----------+-----------+-----------+-----------+-----------+
    | Model 3   | 52.81     | 43.59     | 41.27     | 49.44     | 00:27:04  |
    +-----------+-----------+-----------+-----------+-----------+-----------+
    | Model 4   | 53.55     | 42.22     | 44.11     | 31.58     | 00:28:40  |
    +-----------+-----------+-----------+-----------+-----------+-----------+
    | Model 5   | 53.52     | 42.00     | 46.80     | 43.27     | 00:25:41  |
    +-----------+-----------+-----------+-----------+-----------+-----------+


The mean and standard deviations of the metrics when running the chosen model three times is displayed in
the table below.


.. table:: Model 3 performance

    +-----------+-----------+-----------+
    | Metric    | Average   | STD       |
    +-----------+-----------+-----------+
    | Accuracy  | 53.31     | 0.29      |
    +-----------+-----------+-----------+
    | Macro-F1  | 34.07     | 4.07      |
    +-----------+-----------+-----------+
    | Precision | 33.85     | 4.13      |
    +-----------+-----------+-----------+
    | Recal     | 37.27     | 5.47      |
    +-----------+-----------+-----------+
    | Run Time  | 42.73     | 10.28     |
    +-----------+-----------+-----------+


Conclusion
==========

As shown in the results, the performance are really different from one training to the other. Most likely this
is due to us no setting a random seed, leading to different weights, bias and also affecting the optimizer.
