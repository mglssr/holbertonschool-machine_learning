#!/usr/bin/env python3
"""Task 2"""
import numpy as np


class MultiNormal():
    """MultiNormal Class"""
    def __init__(self, data):
        """class constructor"""
        if type(data) != np.ndarray:
            raise TypeError("data must be a 2D numpy.ndarray")

        n, d = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.matmul(data - self.mean, data.T - self.mean.T) / (n - 1)
