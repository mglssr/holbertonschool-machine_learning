#!/usr/bin/env python3
"""Task 2"""
import numpy as np


class MultiNormal():
    """MultiNormal Class"""
    def __init__(self, data):
        """class constructor"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.dot(data - self.mean, data.T - self.mean.T) / (n - 1)
