#!/usr/bin/env python3
"""task 0"""
import numpy as np


def sensitivity(confusion):
    """function  that calculates the
    sensitivity for each class in a confusion matrix"""
    identity = np.identity(confusion.shape[0])
    sens = np.sum(identity * confusion, axis=1)
    sens = sens / np.sum(confusion, axis=1)
    return (sens)
