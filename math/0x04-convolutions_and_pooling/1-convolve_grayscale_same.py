#!/usr/bin/env python3
""" task 1"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """function that performs a same convolution on grayscale images"""
    kh, kw = kernel.shape
    padh = np.ceil(((kh, kw) - 1) / 2).astype(int)
    padw = np.ceil(((kh, kw) - 1) / 2).astype(int)
    pading = np.pad(images, ((0, 0), (padh, padh), (padw, padw)),
                    'constant', constant_values=0)
    output = np.zeros(shape=images.shape)
    for h in range(output.shape[1]):
        for w in range(output.shape[2]):
            output[:, h, w] = np.sum((pading[:, h: h + kh,
                                             w: w + kw] * kernel), axis=(1, 2))
    return (output)
