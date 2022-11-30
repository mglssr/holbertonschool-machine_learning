#!/usr/bin/env python3
"""Task 8"""


def mat_mul(mat1, mat2):
    """function that performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return
    mat3 = []
    for i in range(len(mat1)):
        out = []
        for j in range(len(mat2[0])):
            arr = []
            for k in range(len(mat1[0])):
                arr.append((mat1[i][k]) * (mat2[k][j]))
            sum = arr[0] + arr[1]
            out.append(sum)
        mat3.append(out)
    return (mat3)
