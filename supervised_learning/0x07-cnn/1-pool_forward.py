#!/usr/bin/env python3
"""task 1"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """function that performs forward propagation over a
    pooling layer of a neural network"""
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    oh = int(((h - kh) // sh) + 1)
    ow = int(((w - kw) // sw) + 1)
    output = np.zeros((m, oh, ow, c))
    for i in range(oh):
        for j in range(ow):
            x = i * sh
            y = j * sw
            if mode == 'max':
                op = np.max(A_prev[:, x: x + kh, y: y + kw, :], axis=(1, 2))
            if mode == 'avg':
                op = np.average(A_prev[:, x: x + kh, y: y + kw, :],
                                axis=(1, 2))
            output[:, i, j, :] = op
    return (output)
