#!/usr/bin/env python3
"""Task 1"""
import numpy as np
mean_cov = __import__('0-mean_cov').mean_cov


def correlation(C):
    """function that calculates a correlation matrix"""
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")

    n, d = C.shape

    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")

    if n != d:
        raise ValueError("C must be a 2D square matrix")

    corr_mat = np.zeros((d, d))

    diag = np.diag(C)

    for i in range(d):

        for j in range(d):

            corr_mat[i][j] = C[i][j] / np.sqrt(diag[i] * diag[j])

    return corr_mat
