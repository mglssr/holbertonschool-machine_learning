#!/usr/bin/env python3
""" task 1"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """function that performs a same convolution on grayscale images"""
    kh, kw = kernel.shape
    padh = int(np.ceil((kh - 1) / 2))
    padw = int(np.ceil((kh - 1) / 2))
    pading = np.pad(images, ((0, 0), (padh, padh), (padw, padw)),
                    'constant', constant_values=0)
    output = np.zeros(shape=images.shape)
    for h in range(images.shape[1]):
        for w in range(images.shape[2]):
            output[:, h, w] = np.sum((pading[:, h: h + kh,
                                             w: w + kw] * kernel), axis=(1, 2))
    return (output)
