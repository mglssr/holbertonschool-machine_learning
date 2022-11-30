#!/usr/bin/env python3
"""Task 8"""


def mat_mul(mat1, mat2):
    """function that performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None
    mat3 = []
    for i in range(len(mat1)):
        out = []
        for j in range(len(mat2[0])):
            sum = 0
            for k in range(len(mat1[0])):
                sum += mat1[i][k] * mat2[k][j]
            out.append(sum)
        mat3.append(out)
    return (mat3)
