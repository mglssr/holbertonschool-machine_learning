#!/usr/bin/env python3
"""task 1"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds a neural network with the Keras library"""
    inputs = K.Input(shape=(nx,))
    regularizer = K.regularizers.L2(lambtha)
    lays = inputs
    for i in range(len(layers)):
        lays = K.layers.Dense(layers[i],
                              activation=activations[i],
                              kernel_regularizer=regularizer)(lays)
        if i < (len(layers) - 1):
            lays = K.layers.Dropout(1 - keep_prob)(lays)
    model = K.Model(inputs=inputs, outputs=lays)
    return (model)
