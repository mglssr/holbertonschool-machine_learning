#!/usr/bin/env python3
"""task 0"""
import numpy as np


def markov_chain(P, s, t=1):
    """function that determines the probability of
    a markov chain being in a particular state after
    a specified number of iterations"""
    out = s
    for i in range(t):
        out = np.dot(out, P)
    return out
