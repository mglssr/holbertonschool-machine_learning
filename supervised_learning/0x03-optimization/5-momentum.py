#!/usr/bin/env python3
"""task 5"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """function that updates a variable using the gradient
    descent with momentum optimization algorithm"""
    dw = (beta1 * v) + (1 - beta1) * grad
    g = var - (alpha * dw)
    return (g, dw)
