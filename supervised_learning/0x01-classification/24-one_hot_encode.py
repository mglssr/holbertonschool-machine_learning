#!/usr/bin/env python3
"""task 24"""
import numpy as np


def one_hot_encode(Y, classes):
    """one-hot encode"""
    try:
        out = np.zeros(shape=(classes, Y.shape[0]))
        for idx, val in enumerate(Y):
            out[val, idx] = 1
        return out
    except Exception as e:
        return None