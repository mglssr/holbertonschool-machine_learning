#!/usr/bin/env python3
"""Normal distribution class"""


def erf(x):
    """function that computes the error function"""
    pi = 3.1415926536

    a = 2 / (pi ** 0.5)
    x_3 = (x ** 3) / 3
    x_5 = (x ** 5) / 10
    x_7 = (x ** 7) / 42
    x_9 = (x ** 9) / 216
    return (a * (x - x_3 + x_5 - x_7 + x_9))


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

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        e = 2.7182818285
        pi = 3.1415926536
        pdf = (1 / (self.stddev * ((2 * pi) ** 0.5))) *\
              (e ** ((self.z_score(x) ** 2) / -2))
        return (pdf)

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        cdf = 0.5 * (1 + erf((x - self.mean) / (self.stddev * (2 ** 0.5))))
        return (cdf)
