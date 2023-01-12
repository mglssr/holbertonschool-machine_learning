#!/usr/bin/env python3
"""task 4"""
import numpy as np


def moving_average(data, beta):
    """function that calculates the weighted moving average of a data set"""
    wmal = []
    w = 0
    for i in range(len(data)):
        w = (beta * data[i] + (1 - beta) * data[i])
        wmal.append(w / 1 - (beta ** (data[i])))
    return wmal