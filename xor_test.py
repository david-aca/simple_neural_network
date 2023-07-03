from network import *
from activations import *

nn = NeuralNetwork([DenseLayer(2, 3, ReLu()),
                    DenseLayer(3, 5, ReLu()),
                    DenseLayer(5, 4, ReLu()),
                    DenseLayer(4, 1, ReLu())])

X = [np.array([0, 1])
    ,np.array([0, 0])
    ,np.array([1, 0])
    ,np.array([1, 1])]

Y = np.array([1, 0, 0, 1])

nn.train(X, Y, 10000)
for x in X:
    print(nn.inference(x))