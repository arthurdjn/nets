import unittest
import copy
from nets.optim import *
from nets.nn import *


SEED = 2020
np.random.seed(SEED)


class TestOptim(unittest.TestCase):
    def test_step(self):
        print('\nInit a model to test on...')
        layer_dimensions = [1, 10, 10, 2]
        model = DNN(layer_dimensions)
        parameters_step1 = copy.deepcopy(model.parameters())
        x = np.array([[1], [2], [-1], [0], [-3]]).T
        labels = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]]).T
        y = model(x)
        model.backward(y, labels)

        print("Create the optimizer...")
        optim = SGD(model.parameters(), model.gradients())
        optim.step()
        parameters_step2 = copy.deepcopy(model.parameters())
        print("Parameters updated.")


if __name__ == '__main__':
    unittest.main()
