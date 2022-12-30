#!/usr/bin/env python3
"""
    module
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Deep Neural Network
    """

    def __init__(self, nx, layers):
        """ init function """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or layers == []:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for ly in range(self.L):
            if type(layers[ly]) != int or layers[ly] < 0:
                raise TypeError("layers must be a list of positive integers")
            lblb = f"b{ly + 1}"
            lblw = f"W{ly + 1}"
            self.weights[lblb] = np.zeros((layers[ly], 1))
            aux = nx
            if (ly > 0):
                aux = layers[ly - 1]
            self.weights[lblw] = (np.random.randn(layers[ly], aux)
                                       * np.sqrt(2/aux))
