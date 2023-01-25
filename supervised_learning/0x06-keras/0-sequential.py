#!/usr/bin/env python3
"""task 0"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds a neural network with the Keras library"""
    model = K.Sequential()
    regularizer = K.regularizers.L2(lambtha)
    for i in range(len(layers)):
        model.add(K.layers.Dense(layers[i],
                                 activation=activations[i],
                                 input_shape=(nx,),
                                 kernel_regularizer=regularizer))
        if i < (len(layers) - 1):
            model.add(K.layers.Dropout(1 - keep_prob))
    return (model)
