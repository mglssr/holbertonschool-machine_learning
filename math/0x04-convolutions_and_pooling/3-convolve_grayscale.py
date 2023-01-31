#!/usr/bin/env python3
"""task 2"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """function that performs a convolution on grayscale images"""
    ks = np.array(kernel.shape)
    if padding == 'same':
        pad = np.ceil((ks - 1) / stride).astype(int)
        pading = np.pad(images, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1])),
                        'constant', constant_values=0)
        output = np.zeros(shape=images.shape)
    elif padding == 'valid':
        m = images.shape[0]
        convh = int((images.shape[1] - ks[0] + 1) / float(stride[0]))
        convw = int((images.shape[2] - ks[1] + 1) / float(stride[1]))
        output = np.zeros((m, convh, convw))
        padding = images
    else:
        ps = np.array(padding)
        pading = np.pad(images, ((0, 0), (ps[0], ps[0]), (ps[1], ps[1])),
                        'constant', constant_values=0)
        image_shape = images.shape[1:]
        oh, ow = image_shape + (2 * ps) - ks + 1
        output = np.zeros(shape=(images.shape[0], oh, ow))
    for i in range(output.shape[1]):
        for j in range(output.shape[2]):
            output[:, i, j] = np.sum((padding[:, i * stride[0]: i *
                                              stride[0] + ks[0], j *
                                              stride[1]: j * stride[1] +
                                              ks[1]] * kernel), axis=(1, 2))
    return (output)
