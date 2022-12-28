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
        self.W1 = np.random.normal(size=(nodes, nx))  # weight hidden layer
        self.b1 = np.zeros((nodes, 1))  # bias for the hidden laye
        self.A1 = 0  # activated output for the hidden layer
        self.W2 = np.random.normal(size=(1, nodes))  # weights output neuron
        self.b2 = 0  # bias for the output neuron
        self.A2 = 0  # activated output for the output neuron
