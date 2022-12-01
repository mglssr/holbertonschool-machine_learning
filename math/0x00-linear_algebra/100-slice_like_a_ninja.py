#!/usr/bin/env python3
"""Task 15(https://towardsdatascience.com/slicing
-numpy-arrays-like-a-ninja-e4910670ceb0)"""


def np_slice(matrix, axes={}):
    """function that slices a matrix along specific axes"""
    sli = [slice(None)] * len(matrix.shape)
    for key, value in axes.items():
        sli[key] = slice(*value)
    return (matrix[sli])
