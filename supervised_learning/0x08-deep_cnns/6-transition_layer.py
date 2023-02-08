#!/usr/bin/env python3
"""Task 6"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """function that builds a transition layer as described
    in https://arxiv.org/pdf/1608.06993.pdf"""
    init = K.initializers.he_normal()
    act = {"kernel_initializer": init,
           "padding": "same"}
    layer = K.layers.BatchNormalization(axis=3)(X)
    layer = K.layers.Activation(K.activations.relu)(layer)
    layer = K.layers.Conv2D(nb_filters * compression, 1, **act)(layer)
    layer = K.layers.AveragePooling2D(pool_size=2, strides=2,
                                      padding='valid')(layer)
    nb_filters *= compression
    return layer, int(nb_filters)
