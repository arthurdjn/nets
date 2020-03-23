import os
import numpy as np
from ._utils import extract_to_dir, download_from_url


class Dataset(object):
    r"""
    Abstract Dataset class. All dataset for machine learning purposes can inherits from this architecture, for convenience.
    """
    urls = []
    name = ''
    dirname = ''

    def __init__(self, data):
        self.data = data
        self.fields = data.fields

    @classmethod
    def splits(cls, train=None, test=None, valid=None, root='.'):
        pass

    @classmethod
    def download(cls, root):
        """Download and unzip an online archive (.zip, .gz, or .tgz).
        Arguments:
            root (str): Folder to download data to.
            check (str or None): Folder whose existence indicates
                that the dataset has already been downloaded, or
                None to check the existence of root/{cls.name}.
        Returns:
            str: Path to extracted dataset.
        """
        path_dirname = os.path.join(root, cls.dirname)
        path_name = os.path.join(path_dirname, cls.name)
        if not os.path.isdir(path_dirname):
            for url in cls.urls:
                filename = os.path.basename(url)
                zpath = os.path.join(path_dirname, filename)
                if not os.path.isfile(zpath):
                    if not os.path.exists(os.path.dirname(zpath)):
                        os.makedirs(os.path.dirname(zpath))
                    print('download {} from {}'.format(filename, url))
                    download_from_url(url, zpath)
                extract_to_dir(zpath, path_name)

        return path_name

    def reshape(self, shape):
        self.data.reshape(shape)

    def __getitem__(self, item):
        return getattr(self.data, item)
