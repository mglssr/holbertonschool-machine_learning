#!/usr/bin/env python3
"""Task 7"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """function that builds the DenseNet-121 architecture as
    described in https://arxiv.org/pdf/1608.06993.pdf"""
    init = K.initializers.he_normal()
    act = {"kernel_initializer": init,
           "padding": "same"}
    X = K.Input(shape=(224, 224, 3))
    nb_filters = growth_rate * 2
    o = K.layers.BatchNormalization()(X)
    o = K.layers.Activation('relu')(o)
    o = K.layers.Conv2D(64, 2, strides=2, **act)(o)
    o = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(o)
    for i in [6, 12, 24]:
        o, nb_filters = dense_block(o, nb_filters, growth_rate, i)
        o, nb_filters = transition_layer(o, nb_filters, compression)
    o, nb_filters = dense_block(o, nb_filters, growth_rate, compression)
    o = K.layers.AveragePooling2D(7, 1, padding='valid')(o)
    o = K.layers.Dense(1000, activation='softmax', kernel_initializer=init)(o)
    model = K.models.Model(inputs=X, outputs=o)
    return model
