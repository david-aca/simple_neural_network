from layers import *
from activations import *

#
# Neural Network 
#
class NeuralNetwork():
    def __init__(self, layers, loss_function=MSE()):
        self.loss_function = loss_function
        self.layers = layers
    
    def train(self, X, Y, epoches, learning_reate=0.001):
        for e in range(epoches):
            error = 0
            for x, y in zip(X, Y):
                output = self.inference(np.array([x]))
                grad = self.loss_function.prime(y, output)
                self.__update(self.loss_function.prime(y, output), learning_reate)
                
                error += self.loss_function(y, output)
            
            print(e + 1, '/', epoches, error / len(X))

    def __update(self, output_gradient, learning_rate):
        for l in reversed(self.layers): 
            output_gradient = l.backward_propagate(output_gradient, learning_rate)

    def inference(self, input):
        for l in self.layers:
            input = l.forward_propagate(input)
        return input