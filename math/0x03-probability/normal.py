#!/usr/bin/env python3
"""Normal distribution class"""


class Normal():
    """Normal class distribution definition"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """class constructor"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            i = 0
            s = 0
            while i < len(data):
                s += ((data[i] - self.mean) ** 2)
                i += 1
            self.stddev = (s / len(data)) ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        z = (x - self.mean) / self.stddev
        return (z)

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        x = (z * self.stddev) + self.mean
        return (x)
