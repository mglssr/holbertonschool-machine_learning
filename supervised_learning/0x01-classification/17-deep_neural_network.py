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
        for ly in range(self.L):
            if type(layers[ly]) != int or layers[ly] < 0:
                raise TypeError("layers must be a list of positive integers")
            lblb = f"b{ly + 1}"
            lblw = f"W{ly + 1}"
            self.__weights[lblb] = np.zeros((layers[ly], 1))
            aux = nx
            if (ly > 0):
                aux = layers[ly - 1]
            self.weights[lblw] = np.random.randn(layers[ly], aux)
            self.weights[lblw] *= np.sqrt(2/aux)

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