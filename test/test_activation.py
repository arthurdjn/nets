import unittest
import numpy as np
from nets.nn.activation import *


SEED = 2020
np.random.seed(SEED)


class TestActivation(unittest.TestCase):
    def test_ReLU(self):
        print('\nReLU')
        x = np.random.rand(10) * 2 - 1
        activation = ReLU()
        print('input: {}'.format(x))
        print('output: {}'.format(activation.forward(x)))
        print('backward: {}'.format(activation.backward(x)))

    def test_Sigmoid(self):
        print('\nSigmoid')


if __name__ == '__main__':
    unittest.main()
