#!/usr/bin/env python3
""" task 0"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """function that performs a valid convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    convh = int(h - kh + 1)
    convw = int(w - kw + 1)
    output = np.zeros(shape=(m, convh, convw))
    for h in range(convh):
        for w in range(convw):
            output[:, h, w] = np.sum((images[:, h: h + kh, w: w + kw]
                                      * kernel), axis=(1, 2))
    return (output)
