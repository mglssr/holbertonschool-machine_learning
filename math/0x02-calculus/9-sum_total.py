#!/usr/bin/env python3
"""Task 9"""


def summation_i_squared(n):
    """ function def summation_i_squared(n):
    that tcalculates sum_{i=1}^{n} i^2:"""
    if type(n) not in [int, float]:
        return
    suma = [*range(n + 1)]
    add = int(sum(map(lambda i: i * i, suma)))
    return (add)
