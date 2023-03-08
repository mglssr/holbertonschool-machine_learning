#!/usr/bin/env python3
"""task 0"""


def pivot_op(aii, aij):
    "resolves a simple ecuation, returns a integer"
    return (- aij) / aii


def determinant(matrix):
    """function that calculates the determinant of a matrix"""

    if len(matrix) == 0 or type(matrix[0]) != list:
        raise TypeError("matrix must be a list of lists")

    if len(matrix[0]) == 0:
        return 1

    shape = len(matrix)

    if shape != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if shape == 1:
        det = matrix[0][0]
    elif shape == 2:
        det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        new_matrix = [row[:] for row in matrix]
        det = 1
        for i in range(shape):
            pivot = matrix[i][i]
            col = []
            for j in range(shape):
                if j > i:
                    a = matrix[j][i]
                    col.append(a)
            for k in range(len(col)):
                num = pivot_op(pivot, col[k])
                comb = [num * x for x in matrix[i][:]]
                new_matrix[i + 1] = list(map(lambda a, b: a + b,
                                             comb, matrix[i + 1]))
                break
            det *= new_matrix[i][i]
    return (det)
