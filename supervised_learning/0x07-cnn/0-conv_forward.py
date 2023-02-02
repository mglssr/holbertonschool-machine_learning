#!/usr/bin/env python3
"""task 0"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """function that performs forward propagation over a convolutional layer
    of a neural network"""
    m, h_prev, w_prev = A_prev.shape[:3]
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = int((h_prev - 1) * sh + kh - h_prev // 2)
        pw = int((w_prev - 1) * sw + kw - w_prev // 2)
    else:
        ph, pw = (0, 0)
    oh = int(((h_prev + 2 * ph - kh) // sh) + 1)
    ow = int(((w_prev + 2 * pw - kw) // sw) + 1)
    output = np.zeros((m, oh, ow, c_new))
    padded_image = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                          'constant', constant_values=0)
    for i in range(oh):
        x = sh * i
        for j in range(ow):
            y = sw * j
            image_slice = padded_image[:, x: x + kh, y: y + kw, :]
            for k in range(c_new):
                output[:, i, j, k] = np.sum(
                    image_slice * W[:, :, :, k], axis=(1, 2, 3))
    return (activation(output + b))
