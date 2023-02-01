#!/usr/bin/env python3
"""task 2"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """function that performs a convolution on grayscale images"""
    kh, kw = kernel.shape
    m, h, w = images.shape
    strides = np.array(stride)
    if padding == 'same':
        ph = np.ceil((h - 1) * strides[0] + kh - h // 2).astype(int)
        pw = np.ceil((w - 1) * strides[1] + kw - w // 2).astype(int)
    elif padding == 'valid':
        ph, pw = (0, 0)
    else:
        ph, pw = padding
    oh = int(((h + 2 * ph - kh) // strides[0]) + 1)
    ow = int(((w + 2 * pw - kw) // strides[1]) + 1)
    output = np.zeros((m, oh, ow))
    padded_image = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                          'constant', constant_values=0)
    for i in range(oh):
        for j in range(ow):
            x = i * strides[0]
            y = j * strides[1]
            output[:, i, j] = np.sum((padded_image[:, x: x + kh, y: y + kw]
                                      * kernel), axis=(1, 2))
    return (output)
