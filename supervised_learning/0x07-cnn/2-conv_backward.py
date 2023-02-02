#!/usr/bin/env python3
"""task 2"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """function that performs back propagation over a convolutional
    layer of a neural network"""
    m, h, w, _ = A_prev.shape
    _, zh, zw, zc = dZ.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    if padding == "same":
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    else:
        ph, pw = (0, 0)
    padded_image = np.pad(A_prev,
                        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode="constant")
    dA_prev = np.zeros(padded_image.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for img in range(m):
        for i in range(zh):
            x = i * sh
            for j in range(zw):
                y = j * sw
                for k in range(zc):
                    dz = dZ[img, i, j, k]
                    slice_img = padded_image[img, x: x + kh, y: y + kw, :]
                    dW[:, :, :, k] += dz * slice_img
                    dA_prev[img, x: x + kh, y: y + kw, :] += dz * W[:, :, :, k]
    if padding == "same":
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :] 
    else:
        pass
    return (dA_prev, dW, db)
