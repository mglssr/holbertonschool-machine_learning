#!/usr/bin/env python3
"""task 0"""
import numpy as np


def specificity(confusion):
    """function that calculates the specificity
    for each class in a confusion matrix"""
    summ = np.sum(confusion)
    ide = confusion * np.identity(confusion.shape[0])
    tn = summ - (np.sum(confusion - ide, axis=0) + np.sum(confusion, axis=1))
    fp = np.sum(confusion - ide, axis=0)
    spe = tn / (fp + tn)
    return (spe)
