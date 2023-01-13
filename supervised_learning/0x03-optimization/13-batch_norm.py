#!/usr/bin/env python3
"""task 13"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """function that normalizes an unactivated output
    of a neural network using batch normalization"""
    mean = np.mean(Z, axis=0)
    mu = np.var(Z, axis=0)
    norma = (Z - mean) / np.sqrt(mu + epsilon)
    return (gamma * norma) + beta
