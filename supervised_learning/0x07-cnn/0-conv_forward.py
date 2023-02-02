#!/usr/bin/env python3
"""task 0"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """function that performs forward propagation over a convolutional layer
    of a neural network"""
    m, h, w, _ = A_prev.shape
    kh, kw, _, nc = W.shape
    sh, sw = stride
    if padding == "same":
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    else:
        ph, pw = (0, 0)
    oh = int((((h + 2 * ph - kh) / sh) + 1))
    ow = int((((w + 2 * pw - kw) / sw) + 1))
    conv = np.zeros(shape=(m, oh, ow, nc))
    padded_img = np.pad(A_prev,
                        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode="constant")
    for i in range(oh):
        x = sh * i
        for j in range(ow):
            y = sw * j
            img_slice = padded_img[:, x: x + kh, y: y + kw, :]
            for k in range(nc):
                conv[:, i, j, k] = np.sum(
                    img_slice * W[:, :, :, k], axis=(1, 2, 3))
    return activation(conv + b)
