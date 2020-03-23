import numpy as np
from nets.nn import DNN, ReLU, Softmax
from nets.optim import SGD
from nets.nn import CrossEntropyLoss
from nets.utils import one_hot

SEED = 1995
np.random.seed(SEED)
np.set_printoptions(precision=3)

INPUT_DIM = 10
BATCH_SIZE = 10000
INPUT = (np.random.rand(BATCH_SIZE, INPUT_DIM))
LABEL = np.where(np.sum(INPUT, axis=1) > INPUT_DIM / 2, 1, 0)
LABEL_ONE_HOT = one_hot(LABEL, 2)

LAYER_DIMENSIONS = [INPUT_DIM, 100, 50, 2]
MODEL = DNN(LAYER_DIMENSIONS, activation_hidden=ReLU(), 
            activation_output=ReLU())


def test_init():
    print('\nDense Neural Network init')
    print(MODEL)
    print("parameters not None: {}".format(MODEL._parameters is not None))


def test_forward():
    print('\nDNN forward pass')
    y = MODEL(INPUT)
    print("output: {}".format(y))
    # assert y == correct_output
    print("cache not ``None``: {}".format(MODEL._cache is not None))


def test_backward():
    print('\nDNN backward pass')
    y = MODEL(INPUT)
    MODEL.backward(y, LABEL_ONE_HOT)
    assert MODEL._cache != {}


def test_epochs():
    EPOCHS = 100
    optimizer = SGD(MODEL.parameters(), MODEL.gradients())
    criterion = CrossEntropyLoss()
    for epoch in range(EPOCHS):
        y = MODEL(INPUT)
        MODEL.backward(y, LABEL_ONE_HOT)
        optimizer.step()
        correct = np.sum(one_hot(np.argmax(y, axis=1), 2) * LABEL_ONE_HOT)
        print('\repoch: {:5d} | accuracy: {:2.2f}%'.format(epoch, correct / INPUT.shape[0] * 100), end='')


if __name__ == '__main__':
    test_init()

    test_forward()

    test_backward()
    
    # pass


