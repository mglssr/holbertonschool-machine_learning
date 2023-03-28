#!/usr/bin/env python3
"""Task 3"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """function that tests for the optimum number of
    clusters by variance"""
    if type(X) is not np.ndarray:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if ((X.ndim != 2 or type(kmin) is not int
         or kmin < 1 or type(iterations) is not int or iterations < 1
         or type(kmax) is not int or kmax <= kmin)):
        return None, None
    res = [kmeans(X, kmin, iterations)]
    firstvar = variance(X, res[0][0])
    d_vars = [0]
    idx = 0
    kmin += 1
    while kmin <= kmax:
        centroids, assigns = kmeans(X, kmin, iterations)
        vari = variance(X, centroids)
        res.append((centroids, assigns))
        d_vars.append(firstvar - vari)
        idx += 1
        kmin += 1
    return res, d_vars
