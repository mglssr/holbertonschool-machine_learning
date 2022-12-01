#!/usr/bin/env python3
"""Task 13"""


def np_cat(mat1, mat2, axis=0):
    """function that concatenates two
    matrices along a specific axis"""
    import numpy as np
    return np.concatenate((mat1, mat2), axis=axis)
