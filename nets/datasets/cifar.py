import os
import pickle
import numpy as np
from .dataset import Dataset
from nets.data.examples import Examples
from nets.data import Field


class CIFAR10(Dataset):
    """CIFAR-10 dataset, available at https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

    Info from the webpage:
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images
    per class. There are 50000 training images and 10000 test images.

    The dataset is divided into five training batches and one test batch, each with 10000
    images. The test batch contains exactly 1000 randomly-selected images from each class. The
    training batches contain the remaining images in random order, but some training batches
    may contain more images from one class than another. Between them, the training batches
    contain exactly 5000 images from each class.

    The CIFAR-10 dataset contains the following classes

    Label   Description
    -------------------
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

    """

    urls = ['https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz']
    name = ''
    dirname = 'cifar10'

    def __init__(self, filepath, fields=None):
        if fields is None:
            fields = [("data", Field()), ("label", Field())]
        # Unpickle file and fill in data
        data = None
        labels = []

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
            data = data.reshape(data.shape[0], 3, 32, 32).astype("float")
            labels = np.array(labels)
        else:
            with open(filepath, 'rb') as f:
                test_data_dict = pickle.load(f, encoding='latin-1')
            data = test_data_dict['data']
            data = data.reshape(data.shape[0], 3, 32, 32).astype("float")
            labels = np.array(test_data_dict['labels'])

        values = (data, labels)
        examples = Examples(values, fields)
        super(CIFAR10, self).__init__(examples)

    @classmethod
    def splits(cls, root='.data', train='data_batch_', test='test_batch', **kwargs):
        """
        Loads training, validation, and test partitions of the cifar10 dataset
        (https://www.cs.toronto.edu/~kriz/cifar.html). If the data is not already contained in
        ``root`` folder, it will download it.

        Args:
            test:
            train:
            root (str): relative or absolute path of the dataset.

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
