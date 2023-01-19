#!/usr/bin/env python3
"""task 0"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """function  that calculates the cost of a
    neural network with L2 regularization"""
    sum = 0
    for i in range(1, L + 1):
        w = weights["W" + str(i)]
        sum += np.square(np.linalg.norm(w))
    sum = (lambtha / (2 * m)) * sum
    ncost = cost + sum
    return (ncost)
