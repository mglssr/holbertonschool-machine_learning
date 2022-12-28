#!/usr/bin/env python3
"""classification"""
import numpy as np


class NeuralNetwork():
    """defines a neural network with one
    hidden layer performing binary classification"""
    def __init__(self, nx, nodes):
        """class constructor"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter"""
        return (self.__W1)

    @property
    def b1(self):
        """getter"""
        return (self.__b1)

    @property
    def A1(self):
        """getter"""
        return (self.__A1)

    @property
    def W2(self):
        """getter"""
        return (self.__W2)

    @property
    def b2(self):
        """getter"""
        return (self.__b2)

    @property
    def A2(self):
        """getter"""
        return (self.__A2)

    def sigmoid_act(self, X):
        """sigmoid activate function"""
        return (1 / (1 + np.exp(-X)))

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__A1 = self.sigmoid_act(np.matmul(self.__W1, X) + self.__b1)
        self.__A2 = self.sigmoid_act(np.matmul(self.__W2, self.__A1)
                                     + self.__b2)
        return (self.__A1, self.__A2)
