#!/usr/bin/env python3
"""Task 7"""


def cat_matrices2D(mat1, mat2, axis=0):
    """function that concatenates two matrices along a specific axis"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return
        mat3 = []
        for r in mat1:
            mat3.append(r[:])
        for r in mat2:
            mat3.append(r[:])
    if axis == 1:
        if len(mat1) != len(mat2):
            return
        mat3 = [(mat1[i] + mat2[i]) for i in range(len(mat1))]
    return (mat3)
