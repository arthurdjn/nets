import numpy as np
from nets.nn import DNN, ReLU, Softmax, Affine
from nets.nn.linear import Linear, Sequential
from nets.optim import SGD
from nets.nn import CrossEntropyLoss
from nets.utils import one_hot


# Normal model
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
            activation_output=Affine())

Y = MODEL(INPUT)
MODEL.backward(Y, LABEL_ONE_HOT)
OPTIMIZER = SGD(MODEL)
OPTIMIZER.step()




# Linear
SEED = 1995
np.random.seed(SEED)
np.set_printoptions(precision=3)

INPUT_DIM = 10
BATCH_SIZE = 10000
input = (np.random.rand(BATCH_SIZE, INPUT_DIM))
label = np.where(np.sum(input, axis=1) > INPUT_DIM / 2, 1, 0)
label_one_hot = one_hot(label, 2)

model = Sequential(Linear(INPUT_DIM, 100, ReLU()), Linear(100, 50, ReLU()), Linear(50, 2))
y = model(input)
model.backward(y, label_one_hot)

optimizer = SGD(model)
optimizer.step()


submodel = Sequential(Linear(2, 5), Linear(5, 2))
model2 = Sequential(model, submodel)
