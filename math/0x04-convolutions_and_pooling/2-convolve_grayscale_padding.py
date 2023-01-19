#!/usr/bin/env python3
"""task 2"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """function that performs a convolution on grayscale
    images with custom padding"""
    ks = np.array(kernel.shape)
    ps = np.array(padding)
    pading = np.pad(images, ((0, 0), (ps[0], ps[0]), (ps[1], ps[1])),
                    'constant', constant_values=0)
    image_shape = images.shape[1:]
    oh, ow = image_shape + (2 * ps) - ks + 1
    output = np.zeros(shape=(images.shape[0], oh, ow))
    for h in range(output.shape[1]):
        for w in range(output.shape[2]):
            output[:, h, w] = np.sum((pading[:, h: h + ks[0], w: w + ks[1]]
                                      * kernel), axis=(1, 2))
    return (output)
