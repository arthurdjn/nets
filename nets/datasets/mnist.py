import os
import pickle
import numpy as np
from .dataset import Dataset
from nets.data.examples import Examples
from nets.data import Field


class CIFAR10(Dataset):
    """
    Loads training, validation, and test partitions of the mnist dataset
    (http://yann.lecun.com/exdb/mnist/). If the data is not already contained in data_dir, it will
    try to download it.

    This dataset contains 60000 training examples, and 10000 test examples of handwritten digits
    in {0, ..., 9} and corresponding labels. Each handwritten image has an "original" dimension of
    28x28x1, and is stored row-wise as a string of 784x1 bytes. Pixel values are in range 0 to 255
    (inclusive).

    Args:
        data_dir: String. Relative or absolute path of the dataset.
        devel_size: Integer. Size of the development (validation) dataset partition.

    Returns:
        X_train: float64 numpy array with shape [784, 60000-devel_size] with values in [0, 1].
        Y_train: uint8 numpy array with shape [60000-devel_size]. Labels.
        X_devel: float64 numpy array with shape [784, devel_size] with values in [0, 1].
        Y_devel: uint8 numpy array with shape [devel_size]. Labels.
        X_test: float64 numpy array with shape [784, 10000] with values in [0, 1].
        Y_test: uint8 numpy array with shape [10000]. Labels.
    """

    urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    name = 'mnist-data-py'
    dirname = 'mnist'

    # def _load_data(filename, data_dir, header_size):
    #     """Load mnist images or labels. This tries to download the data if it is not found.
    #
    #     Args:
    #         filename: Filename of the dataset. Is appended to the root url and used to download the
    #                   data if it is not already downloaded.
    #         data_dir: String. Destination directory.
    #         header_size: uint8. Size of the header in bytes, which is 8 for labels and 16 for
    #                      images. See the mnist webpage for more info.
    #     Returns:
    #         data: uint8 numpy array
    #     """
    #     url = "http://yann.lecun.com/exdb/mnist/" + filename
    #     data_filepath = maybe_download(url, data_dir)
    #     with gzip.open(data_filepath, 'rb') as fil:
    #         data = np.frombuffer(fil.read(), np.uint8, offset=header_size)
    #     return np.asarray(data, dtype=np.uint8)
    #
    # # print("Loading MNIST data from ", data_dir)
    # X_train = _load_data('train-images-idx3-ubyte.gz', data_dir, 16).reshape((-1, 784)).T
    # Y_train = _load_data('train-labels-idx1-ubyte.gz', data_dir, 8)
    # X_test = _load_data('t10k-images-idx3-ubyte.gz', data_dir, 16).reshape((-1, 784)).T
    # Y_test = _load_data('t10k-labels-idx1-ubyte.gz', data_dir, 8)


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
            labels = np.array(labels)
        else:
            with open(filepath, 'rb') as f:
                test_data_dict = pickle.load(f, encoding='latin-1')
            data = test_data_dict['data']
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
