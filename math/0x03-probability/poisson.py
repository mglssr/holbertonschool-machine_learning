#!/usr/bin/env python3
"""Poisson distribution class"""


class Poisson():
    """Poisson distribution definition"""
    def __init__(self, data=None, lambtha=1.):
        """class contructior"""
        if not data:
            if lambtha > 0:
                self.lambtha = float(lambtha)
            else:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(sum(data) / len(data))
