#!/usr/bin/env python3
""" task 1"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """function that performs a same convolution on grayscale images"""
    ks = np.array(kernel.shape)
    pad = np.ceil((ks - 1) / 2).astype(int)
    padding = np.pad(images, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1])),
                 'constant', constant_values=0)
    output = np.zeros(shape=images.shape)
    for x in range(output.shape[1]):
        for y in range(output.shape[2]):
            output[:, x, y] = np.sum(padding[:, x: x + ks[0], y: y + ks[1]]
                                     * kernel, axis=(1, 2))
    return (output)
