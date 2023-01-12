#!/usr/bin/env python3
"""task 9"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """function  that updates a variable in place using the
    Adam optimization algorithm"""
    dw = (beta1 * v) + ((1 - beta1) * grad)
    sdw = (beta2 * s) + ((1 - beta2) * np.square(grad))
    co_vdw = dw / (1 - (beta1 ** t))
    co_sdw = sdw / (1 - (beta2 ** t))
    w = var - alpha * (co_vdw / (np.sqrt(co_sdw) + epsilon))
    return (w, dw, sdw)
