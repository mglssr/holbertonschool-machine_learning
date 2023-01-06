#!/usr/bin/env python3
"""task 3"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """function that calculates the accuracy of a prediction"""
    corr_pred = tf.argmax(y_pred, 1)
    equal = tf.equal(tf.argmax(y, 1), corr_pred)
    acc = tf.reduce_mean(tf.cast(equal, tf.float32))
    return (acc)
