#!/usr/bin/env python3
"""task 6"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """function that creates a layer of a neural network using dropout"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=init)
    layer = layer(prev)
    layer = tf.layers.Dropout(keep_prob)(layer)
    return (layer)
