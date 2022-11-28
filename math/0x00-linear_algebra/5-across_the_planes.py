#!/usr/bin/env python3
"""Task 5"""


def add_matrices2D(mat1, mat2):
    """function that adds two matrices element-wise"""
    if (len(mat1) != len(mat2)):
        return
    if (len(mat1[0]) != len(mat2[0])):
        return
    mat3 = [[(mat1[i][j] + mat2[i][j]) for j in range(len(mat2))]
            for i in range(len(mat2))]
    return (mat3)
