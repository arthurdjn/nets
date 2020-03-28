"""
This module defines how the data should be called from a datasets.
This takes into account ``shuffle`` mode.
"""

import numpy as np
import nets
from .batch import Batch


class Iterator(object):
    r"""
    An ``Iterator`` call the data in batches. These batches can be shuffled and normalized.
    Usually, you want to feed a model with these batches.
    """

    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        starts = np.arange(0, len(self.dataset), self.batch_size)

        if self.shuffle:
            np.random.shuffle(list(self.dataset))

        for start in starts:
            end = start + self.batch_size
            batch_size = min(end, len(self.dataset)) - start
            yield Batch(self.dataset[start:end], batch_size)

    def __len__(self):
        return len(self.dataset) // self.batch_size
