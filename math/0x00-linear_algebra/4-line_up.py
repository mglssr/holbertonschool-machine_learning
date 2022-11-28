#!/usr/bin/env python3
"""Task 4"""


def add_arrays(arr1, arr2):
    """function that adds two arrays element-wise"""
    if len(arr1) != len(arr2):
        return
    arr3 = list(map(lambda a, b: a + b, arr1, arr2))
    return (arr3)
