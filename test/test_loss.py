import unittest
from nets.nn import *


SEED = 2020
np.random.seed(SEED)


class TestLoss(unittest.TestCase):
    def test_SGD(self):
        criterion = CrossEntropyLoss()
        print('\nInit a model to test on...')
        layer_dimensions = [1, 10, 10, 2]
        model = DNN(layer_dimensions)
        x = np.array([[1], [2], [-1], [0], [-3]]).T
        labels = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]]).T
        y = model(x)
        model.backward(y, labels)
        loss, correct = criterion(y, labels)
        print("loss: {}".format(loss))
        print("correct: {}".format(correct))



if __name__ == '__main__':
    unittest.main()
