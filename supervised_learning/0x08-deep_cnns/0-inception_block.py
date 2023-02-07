#!/usr/bin/env python3
"""Task 0"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """function that builds an inception block as
    described in https://arxiv.org/pdf/1409.4842.pdf"""
    init = K.initializers.he_normal()
    act = {"activation": K.activations.relu,
           "kernel_initializer": init,
           "padding": "same"}
    F1 = K.layers.Conv2D(filters=filters[0], kernel_size=1, **act)(A_prev)
    F3R = K.layers.Conv2D(filters=filters[1], kernel_size=1, **act)(A_prev)
    F3 = K.layers.Conv2D(filters=filters[2], kernel_size=3, **act)(F3R)
    F5R = K.layers.Conv2D(filters=filters[3], kernel_size=1, **act)(A_prev)
    F5 = K.layers.Conv2D(filters=filters[4], kernel_size=5, **act)(F5R)
    Fmax = K.layers.MaxPooling2D(pool_size=3, strides=1,
                                 padding="same")(A_prev)
    FPP = K.layers.Conv2D(filters=filters[5], kernel_size=1, **act)(Fmax)
    return K.layers.concatenate([F1, F3, F5, FPP])
