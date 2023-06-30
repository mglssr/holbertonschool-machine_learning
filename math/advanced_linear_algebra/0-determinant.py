#!/usr/bin/env python3
"""task 0"""


def pivot_op(aii, aij):
    "resolves a simple ecuation, returns a integer"
    return (- aij) / aii


def determinant(matrix):
    """function that calculates the determinant of a matrix"""

    if type(matrix) != list:
        raise TypeError("matrix must be a list of lists")

    shape = len(matrix)

    if shape == 0:
        raise TypeError("matrix must be a list of lists")

    if shape == 1:
        if matrix[0] == []:
            return 1
        if type(matrix[0]) != list:
            raise TypeError("matrix must be a list of lists")
        return matrix[0][0]

    for lists in matrix:
        if type(lists) != list:
            raise TypeError("matrix must be a list of lists")
        if len(lists) != shape:
            raise ValueError("matrix must be a square matrix")

    if shape == 1:
        return matrix[0][0]
    if shape == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    new_matrix = [row[:] for row in matrix]
    det = 1
    for i in range(shape-1):
        pivot = matrix[i][i]
        for j in range(shape-1, i, -1):
            if j > i:
                a = matrix[j][i]
                num = pivot_op(pivot, a)
                comb = [num * x for x in matrix[i][:]]
                new_matrix[i+1] = list(map(lambda a, b: a + b,
                                       comb, matrix[i+1]))
    for p in range(len(new_matrix)):
        det *= new_matrix[p][p]
    return round(det)
