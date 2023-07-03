from abc import ABC, abstractclassmethod
import numpy as np
from activations import Sigmoid
from scipy.signal import correlate2d, convolve2d

#
# Base Layer
#
class Layer(ABC):
    @abstractclassmethod
    def forward_propagate(self, input): pass
    @abstractclassmethod
    def backward_propagate(self, output_gradient, learning_rate): pass

OutputLayer = Layer
#
# Soft Max Layer
#
# class SoftMaxLayer(OutputLayer):
#   pass

#
# Dense Layer
#
class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation=Sigmoid()):
        self.activation = activation
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward_propagate(self, input):
        self.input = input # for back propagation
        return np.transpose(self.weights @ np.matrix(self.activation( input )).T + self.bias)

    def backward_propagate(self, output_gradient, learning_reate):
        output_gradient = np.multiply(output_gradient.T, self.activation.prime(self.forward_propagate(self.input)))
        output = np.dot(self.weights.T, output_gradient.T)
        weights_gradient = np.dot(self.input.T, output_gradient).T
        self.weights -= learning_reate * weights_gradient
        self.bias -= output_gradient.T * learning_reate

        return output

#
# Convolution Layer
#
class ConvolutionLayer2d(Layer):
    def __init__(self, kernels_shape, depth):
        assert(len(kernels_shape) == 3)
        self.kernels = np.random.randn(depth, *reversed(kernels_shape))
        self.biases = None

    def __correlate(self, input, kernel):
        return np.array(sum([ correlate2d(l, k, 'valid') for l, k in zip(input, kernel) ]))
    
    def forward_propagate(self, input):
        self.input = input # for back propagation
        output = np.array([ self.__correlate(input, kernel) for kernel in self.kernels ])
        if self.biases == None: 
            self.biases = np.random.randn( *output.shape )

        return output + self.biases

    def backward_propagate(self, output_gradient, learning_rate):
        kernels_gradient = [[correlate2d(layer, ograd, 'valid') for ograd in output_gradient] for layer in self.input]
        self.kernels -= np.array(kernels_gradient) * learning_rate
        self.biases -= output_gradient * learning_rate

        input_gradient = np.zeros(self.input.shape)
        for i in range(len(self.kernels)):
            for j in range(len(self.input)):
                input_gradient[j] += convolve2d(output_gradient[i], self.kernels[i, j], "full")

        return input_gradient


#
# Reshape Layer
#
class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_propagate(self, input):
        return np.reshape(input, self.output_shape)

    def backward_propagate(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

#
# Max Polling
#
class MaxPooling(Layer):
    def __init__(self):
        pass
    def forward_propagate(self, input):
        pass
    def backward_propagate(self, output_gradient, learning_rate):
        pass