#!/usr/bin/env python3
"""Task 9"""


def summation_i_squared(n):
    """ function def summation_i_squared(n):
    that tcalculates sum_{i=1}^{n} i^2:"""
    i = 1
    sum = 0
    while (i <= n):
        sum += i ** 2
        i += 1
    return (sum)
