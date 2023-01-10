#!/usr/bin/env python3
"""task 1"""
import numpy as np


def normalize(X, m, s):
    """function that normalizes (standardizes) a matrix"""
    X = (X - m) / s
    return (X)
