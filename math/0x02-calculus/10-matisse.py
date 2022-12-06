#!/usr/bin/env python3
"""Task 10"""


def poly_derivative(poly):
    """function that calculates the derivative of a polynomial"""
    dx_poly = []
    if len(poly) == 1:
        return ([0])
    i = 1
    while i < len(poly):
        if type(poly[i]) != (int, float):
            return
        dx_poly.append(i * poly[i])
        i += 1
    return (dx_poly)
