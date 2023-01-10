#!/usr/bin/env python3
"""task 2"""
import numpy as np


def shuffle_data(X, Y):
    """function that shuffles the data points in two matrices the same way"""
    a = np.random.permutation(len(X))
    return(X[a], Y[a])
