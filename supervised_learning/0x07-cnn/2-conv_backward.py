#!/usr/bin/env python3
"""task 2"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """function that performs back propagation over a convolutional
    layer of a neural network"""
    m, h_prev, w_prev = A_prev.shape[:3]
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    m, h_new, w_new, c_new = dZ.shape
    if padding == 'same':
        ph = int((h_prev - 1) * sh + kh - h_prev // 2)
        pw = int((w_prev - 1) * sw + kw - w_prev // 2)
    if padding == 'valid':
        ph, pw = (0, 0)
    conv_map = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                          'constant', constant_values=0)
    db = np.sum(dZ, axis=(1, 2, 3), keepdims=True)
    dW = np.zeros(shape=W.shape)
    dA = np.zeros(shape=A_prev.shape)    
    for image in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    fil = W[:, :, :, f]
                    dz = dZ[image, h, w, f]
                    a = A_prev[image, h * sh : h * sh + kh, w * sw : w * sw + kw, :]
                    dW[:, :, :, f] += a * dz
                    dA[image, h * sh : h * sh + kh, w * sw : w * sw + kw, :] += dz * fil
    dA = dA[:, ph:-ph, pw:-pw, :]
    return (dA, dW, db)
