#!/usr/bin/env python3
"""classification"""
import numpy as np


class DeepNeuralNetwork():
    """defines a deep neural network with one
    hidden layer performing binary classification"""

    def __init__(self, nx, layers):
        """class constructor"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or layers == []:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for ly in range(self.__L):
            if type(layers[ly]) != int or layers[ly] < 0:
                raise TypeError("layers must be a list of positive integers")
            s_l = str(ly + 1)
            self.__weights["b" + s_l] = np.zeros((layers[ly], 1))
            aux = nx
            if (ly > 0):
                aux = layers[ly - 1]
            self.__weights["W" + s_l] = np.random.randn(layers[ly], aux)
            self.__weights["W" + s_l] *= np.sqrt(2/aux)

    @property
    def L(self):
        """comment"""
        return (self.__L)

    @property
    def cache(self):
        """comment"""
        return (self.__cache)

    @property
    def weights(self):
        """comment"""
        return (self.__weights)

    def sigmoid_act(self, X):
        """sigmoid activate function"""
        return (1 / (1 + np.exp(-X)))

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X
        for ly in range(1, self.L + 1):
            aux = self.cache["A" + str(ly - 1)]
            Z = np.matmul(self.weights["W" + str(ly)], aux)
            Z += self.weights["b" + str(ly)]
            self.__cache["A" + str(ly)] = self.sigmoid_act(Z)
        return self.cache["A" + str(ly)], self.cache
