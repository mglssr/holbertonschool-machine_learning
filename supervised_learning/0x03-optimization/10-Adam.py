#!/usr/bin/env python3
"""task 10"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """function that creates the training operation for a neural
    network in tensorflow using the Adam optimization algorithm"""
    m_op = tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    ).minimize(loss)
    return (m_op)
