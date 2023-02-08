#!/usr/bin/env python3
"""Task 5"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """function that builds a dense block as described
    in https://arxiv.org/pdf/1608.06993.pdf"""
    init = K.initializers.he_normal()
    act = {"kernel_initializer": init,
           "padding": "same"}
    for ly in range(layers):
        layer = K.layers.BatchNormalization(axis=3)(X)
        layer = K.layers.Activation(K.activations.relu)(layer)
        layer = K.layers.Conv2D(4 * growth_rate, 1, **act)(layer)
        layer = K.layers.BatchNormalization(axis=3)(layer)
        layer = K.layers.Activation(K.activations.relu)(layer)
        layer = K.layers.Conv2D(growth_rate, 3, **act)(layer)
        X = K.layers.concatenate([X, layer])
        nb_filters += growth_rate
    return X, nb_filters
