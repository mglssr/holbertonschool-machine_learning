#!/usr/bin/env python3
"""task 0"""
import numpy as np


def normalization_constants(X):
    """function that calculates the normalization"""
    mean = np.mean(X, axis=0)
    stddev = np.std(X, axis=0)
    return (mean, stddev)
