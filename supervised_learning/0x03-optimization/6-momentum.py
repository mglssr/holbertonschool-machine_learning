#!/usr/bin/env python3
"""task 6"""
import numpy as np
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """function that creates the training operation for a
    neural network in tensorflow using the gradient descent
    with momentum optimization algorithm"""
    m_opt = tf.train.MomentumOptimizer(
        learning_rate=alpha,
        momentum=beta1
    ).minimize(loss)
    return (m_opt)
