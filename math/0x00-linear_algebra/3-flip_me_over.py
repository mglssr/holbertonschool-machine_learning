#!/usr/bin/env python3
"""Task 3"""


def matrix_transpose(matrix):
    """function that returns the transpose of a 2D matrix"""
    t_matrix = [[matrix[i][j] for i in range(len(matrix))]
                for j in range(len(matrix[0]))]
    return (t_matrix)
