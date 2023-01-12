#!/usr/bin/env python3
"""task 7"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """function that updates a variable using
    the RMSProp optimization algorithm"""
    sdw = beta2 * s + (1 - beta2) * (np.square(grad))
    w = var - (alpha * (grad / (np.sqrt(sdw) + epsilon)))
    return (w, sdw)
