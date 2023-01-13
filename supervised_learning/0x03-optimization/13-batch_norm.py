#!/usr/bin/env python3
"""task 13"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """function that normalizes an unactivated output
    of a neural network using batch normalization"""
    mu = np.mean(Z, axis=0)
    dev = np.var(Z, axis=0)
    norma = (Z - mu) / (np.sqrt(dev - epsilon))
    y = (gamma * norma) + beta
    return (y)
