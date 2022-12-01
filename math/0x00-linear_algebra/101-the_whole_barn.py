#!/usr/bin/env python3
"""Task 16"""


def matrix_shape(matrix):
    """function that calculates the shape of a matrix"""
    shape = []
    while (type(matrix) != int):
        shape.append(len(matrix))
        matrix = matrix[0]
    return (shape)


def add_matrices(mat1, mat2):
    """function that adds two matrices"""
    arr1 = matrix_shape(mat1)
    arr2 = matrix_shape(mat2)
    if arr1 != arr2:
        return
    else:
        mat3 = []
        if len(arr1) == 1:
            for i in range(arr1[0]):
                mat3.append(mat1[i] + mat2[i])
            return (mat3)
        else:
            for i in range(arr1[0]):
                mat3.append(add_matrices(mat1[i], mat2[i]))
        return (mat3)
