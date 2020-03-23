import numpy as np


class Field(object):

    def __init__(self, transform=None, dtype=None):
        self.transform = transform
        self.dtype = dtype

    def process(self, value):
        if self.transform is not None:
            value = self.transform(value)
        if self.dtype is not None:
            value = value.astype(self.dtype)
        return value



class Examples(object):

    def __init__(self, values, fields):
        self.fields = fields
        for (value, field) in zip(values, fields):
            assert len(field) == 2, ("expected two Field but got {}".format(field))
            value = field[1].process(value)
            setattr(self, field[0], value)

    def reshape(self, shape):
        for (key, value) in self.__dict__.items():
            if isinstance(value, np.ndarray):
                setattr(self, key, value.reshape(shape))



