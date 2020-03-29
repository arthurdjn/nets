r"""
Load and preprocess the CIFAR-10 dataset.
"""

import os
import pickle
import numpy as np
from nets.data.dataset import Dataset
import nets


class CIFAR10(Dataset):
    r"""CIFAR-10 dataset, available at https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images
    per class. There are 50000 training images and 10000 test images, with the following classes:

    =====   ===========
    Label   Description
    =====   ===========
    0       airplane
    1       automobile
    2       bird
    3       cat
    4       deer
    5       dog
    6       frog
    7       horse
    8       ship
    9       truck
    =====   ===========

    .. note::

        The dataset is divided into five training batches and one test batch, each with 10000
        images. The test batch contains exactly 1000 randomly-selected images from each class. The
        training batches contain the remaining images in random order, but some training batches
        may contain more images from one class than another. Between them, the training batches
        contain exactly 5000 images from each class.
    """
    urls = ['https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz']
    name = ''
    dirname = 'cifar10'

    def __init__(self, filepath, transform=None):
        data = None
        labels = []

        # Open the training data
        if filepath[-1] == "_":
            for idx in range(1, 6):
                filename = filepath + str(idx)
                with open(filename, 'rb') as f:
                    data_dict = pickle.load(f, encoding='latin-1')
                if idx == 1:
                    data = data_dict['data']
                else:
                    data = np.vstack((data, data_dict['data']))
                labels.extend(data_dict['labels'])
            data = data.reshape(-1, 3, 32, 32).astype("float")
            data = data.transpose((0, 2, 3, 1))  # convert to HWC
            labels = np.array(labels)

        # Open the testing data or one training batch
        else:
            with open(filepath, 'rb') as f:
                test_data_dict = pickle.load(f, encoding='latin-1')
            data = test_data_dict['data']
            data = data.reshape(data.shape[0], 3, 32, 32).astype("float")
            data = data.transpose((0, 2, 3, 1))  # convert to HWC
            labels = np.array(test_data_dict['labels'])

        if transform is not None:
            data = transform(data)
        self.data = nets.Tensor(data)
        self.labels = nets.Tensor(labels)

    @classmethod
    def splits(cls, root='.data', train='data_batch_', test='test_batch', **kwargs):
        r"""Loads training, validation, and test partitions of the cifar10 dataset
        (https://www.cs.toronto.edu/~kriz/cifar.html). If the data is not already contained in
        ``root`` folder, it will download it.

        Args:
            root (string): relative or absolute path of the dataset.
            train (string): training data path
            test (string):  testing data path

        Returns:
            tuple(Dataset): training and testing datasets
        """
        path = os.path.join(root, cls.dirname, cls.name, 'cifar-10-batches-py')
        if not os.path.isdir(path):
            path = cls.download(root)
            path = os.path.join(path, 'cifar-10-batches-py')
        path_train = os.path.join(path, train)
        path_test = os.path.join(path, test)
        return CIFAR10(path_train, **kwargs), CIFAR10(path_test, **kwargs)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __setitem__(self, key, value):
        self.data[key], self.labels[key] = value

    def __len__(self):
        return len(self.data)
