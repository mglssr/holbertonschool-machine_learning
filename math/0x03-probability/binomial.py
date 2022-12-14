#!/usr/bin/env python3
"""Binomial distribution class"""


def factorial(x):
    """function that computes the factorial function"""
    f = 1
    for i in range(1, x + 1):
        f = f * i
    return (f)


class Binomial():
    """Binomial class distribution definition"""
    def __init__(self, data=None, n=1, p=0.5):
        """class constructor"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = float(sum(data) / len(data))
                s = 0
                i = 0
                while i < len(data):
                    s += (data[i] - mean) ** 2
                    i += 1
                var = s / len(data)
                self.n = round(mean / (- (var / mean) + 1))
                self.p = mean / self.n

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        p = self.p
        q = 1 - p
        n = self.n
        if k <= 0:
            return (0)
        else:
            k = int(k)
        nk = factorial(n) / (factorial(k) * factorial(n - k))
        pmf = nk * ((p ** k) * q ** (n - k))
        return (pmf)
