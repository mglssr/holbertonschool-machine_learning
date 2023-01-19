#!/usr/bin/env python3
"""task 3"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """function  that creates a tensorflow layer that
    includes L2 regularization"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    reg = tf.keras.regularizers.L2(l2=lambtha)
    layer = tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=reg
    )(prev)
    return (layer)
