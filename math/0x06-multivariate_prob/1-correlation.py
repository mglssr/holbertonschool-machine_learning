#!/usr/bin/env python3
"""Task 1"""
import numpy as np


def correlation(C):
    """function that calculates a correlation matrix"""
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    n, d = C.shape
    corr_mat = np.zeros((d, d))

    diag = np.diag(C)

    for i in range(d):

        for j in range(d):

            corr_mat[i][j] = C[i][j] / np.sqrt(diag[i] * diag[j])

    return corr_mat
