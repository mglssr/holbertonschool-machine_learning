#!/usr/bin/env python3
"""Exponential distribution class"""


class Exponential():
    """Exponential distribution class definition"""
    def __init__(self, data=None, lambtha=1.):
        """class constructor"""
        if data is None:
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
                self.lambtha = float(1 / (sum(data) / len(data)))
