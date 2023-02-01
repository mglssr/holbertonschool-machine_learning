#!/usr/bin/env python3
"""task 6"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """function that performs pooling on images"""
    m, h, w, c = images.shape
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
                op = np.max(images[:, x: x + kh, y: y + kw, :], axis=(1, 2))
            if mode == 'avg':
                op = np.average(images[:, x: x + kh, y: y + kw, :],
                                axis=(1, 2))
            output[:, i, j, :] = op
    return (output)
