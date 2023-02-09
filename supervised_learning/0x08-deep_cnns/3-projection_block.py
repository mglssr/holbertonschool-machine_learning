#!/usr/bin/env python3
"""Task 3"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """function that builds a projection block as
    described in https://arxiv.org/pdf/1512.03385.pdf"""
    init = K.initializers.he_normal()
    act = {"kernel_initializer": init,
           "padding": "same"}
    F11 = K.layers.Conv2D(filters=filters[0], kernel_size=1,
                          strides=s, **act)(A_prev)
    F11 = K.layers.BatchNormalization()(F11)
    F11 = K.layers.Activation(K.activations.relu)(F11)
    F3 = K.layers.Conv2D(filters=filters[1], kernel_size=3, **act)(F11)
    F3 = K.layers.BatchNormalization()(F3)
    F3 = K.layers.Activation(K.activations.relu)(F3)
    F12 = K.layers.Conv2D(filters=filters[2], kernel_size=1, **act)(F3)
    F12 = K.layers.BatchNormalization()(F12)
    F11_b = K.layers.Conv2D(filters=filters[2], kernel_size=1,
                            strides=s, **act)(A_prev)
    F11_b = K.layers.BatchNormalization()(F11_b)
    F12 = K.layers.BatchNormalization()(F11)
    output = K.layers.Add()([F12, F11_b])
    output = K.layers.Activation(K.activations.relu)(output)
    return output
