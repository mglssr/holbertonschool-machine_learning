#!/usr/bin/env python3
"""Task 9"""


def summation_i_squared(n):
    """ function def summation_i_squared(n):
    that tcalculates sum_{i=1}^{n} i^2 =
    ∑i2=(n)(n+1)(2n+1)6"""
    if type(n) != int or n < 1:
        return
    add = n * (n+1) * (2 * n + 1) / 6
    return (int(add))
