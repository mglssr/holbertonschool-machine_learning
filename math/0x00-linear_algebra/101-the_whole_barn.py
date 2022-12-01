#!/usr/bin/env python3
"""Task 16"""
import numpy as np


def add_matrices(mat1, mat2):
    """function that adds two matrices"""
    arr1 = np.array(mat1)
    arr2 = np.array(mat2)
    if arr1.shape != arr2.shape:
        return
    else:
        mat3 = (np.array(mat1) + np.array(mat2)).tolist()
        return (mat3)
