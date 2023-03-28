#!/usr/bin/env python3
"""Task 1"""
import numpy as np


def pca(X, ndim):
    """function that performs PCA on a dataset"""
    X = X - np.mean(X, axis=0)

    U, S, Vt = np.linalg.svd(X)

    Tr = np.matmul(U[..., :ndim], np.diag(S[..., :ndim]))

    return Tr
