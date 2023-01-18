#!/usr/bin/env python3
"""task 4"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """function that calculates the F1 score of a confusion matrix"""
    rec = sensitivity(confusion)
    pre = precision(confusion)
    f1 = 2 * (rec * pre) / (rec + pre)
    return (f1)
