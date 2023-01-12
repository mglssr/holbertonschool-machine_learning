#!/usr/bin/env python3
"""task 4"""
import numpy as np


def moving_average(data, beta):
    """function that calculates the weighted moving average of a data set"""
    wmal = []
    w = 0
    for idx, val in enumerate(data):
        w = beta * w + (1 - beta) * val
        wmal.append(w / (1 - beta ** (idx + 1)))
    return wmal
