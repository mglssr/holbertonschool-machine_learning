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

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period"""
        e = 2.7182818285
        lam = self.lambtha
        if x < 0:
            return (0)
        pdf = lam * e ** ((-1 * lam) * x)
        return (pdf)
