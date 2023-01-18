#!/usr/bin/env python3
"""task 0"""
import numpy as np


def precision(confusion):
    """function that calculates
    the precision for each class in a confusion matrix"""
    identity = np.identity(confusion.shape[0])
    sens = np.sum(identity * confusion, axis=0)
    sens = sens / np.sum(confusion, axis=0)
    return (sens)
