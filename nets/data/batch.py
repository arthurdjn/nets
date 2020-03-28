from typing import Iterator, NamedTuple
import numpy as np

from nets.tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class Batch(object):
    def __init__(self, example, batch_size):
        self.example = example
        self.batch_size = batch_size

    def normalize(self):
        pass

    def __getitem__(self, item):
        return self.example[item]

    def __len__(self):
        return self.batch_size

