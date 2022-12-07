#!/usr/bin/env python3
"""Task 10"""


def poly_derivative(poly):
    """function that calculates the derivative of a polynomial"""
    if poly == []:
        return
    dx_poly = []
    i = 1
    try:
        while i < len(poly):
            dx_poly.append(i * poly[i])
            i += 1
    except Exception:
        return
    if dx_poly == []:
        return ([0])
    return (dx_poly)
