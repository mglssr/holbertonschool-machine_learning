#!/usr/bin/env python3
"""task 3"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """function that performs back propagation over a pooling layer
    of a neural network"""
    m, h, w, c = A_prev.shape
    _, dh, dw, _ = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros(A_prev.shape)
    for img in range(m):
        for i in range(dh):
            x = i * sh
            for j in range(dw):
                y = j * sw
                for k in range(c):
                    if mode == 'max':
                        A_prev_slice = A_prev[img, x: x + kh, y: y + kw, k]
                        mask = (A_prev_slice == np.max(A_prev_slice))
                        res = mask * dA[img, i, j, k]
                        dA_prev[img, x: x + kh, y: y + kw, k] += res
                    else:
                        average_dA = dA[img, i, j, k] / (kw * kh)
                        res = np.ones((kh, kw)) * average_dA
                        dA_prev[img, x: x + kh, y: y + kw, k] += res
    return (dA_prev)
