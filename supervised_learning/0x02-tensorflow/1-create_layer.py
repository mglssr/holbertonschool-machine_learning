#!/usr/bin/env python3
"""task 1"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """create layer"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, name="layer", activation=activation,
                                  kernel_initializer=init)
    return (layer(prev))
