#!/usr/bin/env python3
"""task 2"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """function that performs a convolution on grayscale images"""
    ks = np.array(kernel.shape)
    if padding == 'same':
        pad = np.ceil((ks - 1) / stride).astype(int)
        padding = np.pad(images, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1])),
                         'constant', constant_values=0)
        output = np.zeros(shape=images.shape)
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                output[:, i, j] = np.sum(padding[:, i: i + ks[0], j: j + ks[1]]
                                         * kernel, axis=(1, 2))
    elif padding == 'valid':
        m = images.shape[0]
        convh = int((images.shape[1] - ks[0] + 1) / float(stride[0]))
        convw = int((images.shape[2] - ks[1] + 1) / float(stride[1]))
        output = np.zeros((m, convh, convw))
        for i in range(convh):
            for j in range(convw):
                output[:, i, j] = np.sum((images[:, i * stride[0]: i *
                                                 stride[0] + ks[0], j *
                                                 stride[1]: j * stride[1] +
                                                 ks[1]] * kernel), axis=(1, 2))
    return (output)
