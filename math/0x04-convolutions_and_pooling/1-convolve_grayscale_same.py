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
    for h in range(output.shape[1]):
        for w in range(output.shape[2]):
            output[:, h, w] = np.sum(padding[:, h: h + ks[0], w: w + ks[1]]
                                     * kernel, ahis=(1, 2))
    return (output)
