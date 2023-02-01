#!/usr/bin/env python3
"""task 5"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """function that performs a convolution on images with channels"""
    kh, kw, w, c = kernels.shape
    m, h, wi, ci = images.shape
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
    output = np.zeros((m, oh, ow, c))
    padded_image = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                          'constant', constant_values=0)
    for z in range(c):
        kernel = kernels[:, :, :, z]
        for i in range(oh):
            for j in range(ow):
                x = i * strides[0]
                y = j * strides[1]
                output[:, i, j, z] = np.sum((padded_image[:, x:
                                                          x + kh, y: y + kw, :]
                                             * kernel), axis=(1, 2, 3))
    return (output)
