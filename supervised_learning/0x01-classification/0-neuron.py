#!/usr/bin/env python3
"""classification"""
import numpy as np


class Neuron():
    """Neuron class"""
    def __init__(self, nx):
        """class constructor"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(size=(1, nx))
        # The weights vector for the neuron.
        self.b = 0
        # The bias for the neuron.
        self.A = 0
        # The activated output of the neuron (prediction).
