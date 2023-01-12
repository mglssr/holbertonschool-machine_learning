#!/usr/bin/env python3
"""task 8"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """function that creates the training operation for
    a neural network in tensorflow using the RMSProp
    optimization algorithm"""
    m_opt = tf.train.RMSPropOptimizer(
        learning_rate=alpha,
        decay=beta2,
        epsilon=epsilon
    ).minimize(loss)
    return (m_opt)
