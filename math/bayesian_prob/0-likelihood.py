#!/usr/bin/env python3
"""Task 0"""
import numpy as np


def likelihood(x, n, P):
    """calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects"""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if int(x) < 0:
        raise ValueError("x must be an integer that is\
                        greater than or equal to 0")
    if int(x) > n:
        raise ValueError("x cannot be greater than n")
    if P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    L = P.tolist()
    for i in L:
        if i < 0 or i > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    f = np.math.factorial
    comb = f(n) / (f(n - x) * f(x))
    success = np.power(P, x)
    failure = np.power(1 - P, n - x)
    return comb * success * failure
