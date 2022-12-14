#!/usr/bin/env python3
"""Binomial distribution class"""


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
