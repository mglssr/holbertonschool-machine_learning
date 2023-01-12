#!/usr/bin/env python3
"""task 11"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """function that updates the learning rate using
    inverse time decay in numpy"""
    epsilon = global_step // decay_step
    alpha *= (1 / (1 + (decay_rate * epsilon)))
    return (alpha)
