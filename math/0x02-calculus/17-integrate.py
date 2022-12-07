#!/usr/bin/env python3
"""Task 17"""


def poly_integral(poly, C=0):
    """function that calculates the integral of a polynomial"""
    int_poly = []
    if type(C) != int or poly == [] or type(poly) != list:
        return
    int_poly.append(C)
    if poly == [0]:
        return (int_poly)
    i = 0
    try:
        while i < len(poly):
            p = poly[i] / (i + 1)
            if int(p) == p:
                int_poly.append(int(p))
            else:
                int_poly.append(p)
            i += 1
    except Exception:
        return
    return (int_poly)
