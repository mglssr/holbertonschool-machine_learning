#!/usr/bin/env python3
"""Poisson distribution class"""


class Poisson():
    """Poisson distribution definition"""
    def __init__(self, data=None, lambtha=1.):
        """class contructior"""
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
                self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        e = 2.7182818285
        k_fac = 1
        lam = self.lambtha
        if type(k) != int:
            k = int(k)
        if k < 0:
            return (0)
        for i in range(1, k + 1):
            k_fac = k_fac * i
        pmf = ((lam ** k) * (e ** (-1 * lam))) / k_fac
        return (pmf)
