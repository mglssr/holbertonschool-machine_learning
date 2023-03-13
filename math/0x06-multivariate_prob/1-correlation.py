#!/usr/bin/env python3
"""Task 1"""
import numpy as np
mean_cov = __import__('0-mean_cov').mean_cov


def correlation(C):
    """function that calculates a correlation matrix"""
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    dia = np.corrcoef(C)
    return dia
