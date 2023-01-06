#!/usr/bin/env python3
"""task 4"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """function that creates the training operation for the network"""
    tr = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return (tr)
