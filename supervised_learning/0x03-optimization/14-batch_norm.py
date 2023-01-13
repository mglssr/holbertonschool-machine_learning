#!/usr/bin/env python3
"""task 13"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """function that creates a batch normalization
    layer for a neural network in tensorflow"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, kernel_initializer=init)
    layer = layer(prev)
    mean, dev = tf.nn.moments(layer, axes=[0])
    gamma = tf.Variable(
        tf.constant(1.0, shape=[n]),
        trainable=True,
        name="gamma"
        )
    beta = tf.Variable(
        tf.constant(0.0, shape=[n]),
        trainable=True,
        name="beta"
        )
    norma = tf.nn.batch_normalization(layer, mean, dev, beta, gamma, 1e-8)
    return (activation(norma))
