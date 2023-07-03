from abc import ABC, abstractclassmethod
import numpy as np

#
# Base Function
#
class Function(ABC):
    @abstractclassmethod
    def __call__(): pass
    @abstractclassmethod
    def prime(): pass

Activation = Function

#
# Sigmoid
#
class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp( -x ))
    def prime(self, x):
        y = self.__call__(x)
        return y * (1 - y)

#
# ReLu 
#
class ReLu(Activation):
    def __call__(self, x):
        return np.maximum(0, x)
    def prime(self, x):
        return x >= 0

#
# Tanh
#
class Tanh(Activation):
    def __call__(self, x):
        return np.tanh(x)
    def prime(self, x):
        return 1 - np.power(np.tanh(x), 2)

#
# MSE
#
class MSE(Function):
    def __call__(self, y_true, y_pred):
        return np.mean( np.power(y_true - y_pred, 2) )
    def prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)
