#!/usr/bin/env python3
"""task 12"""
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """function that creates a learning rate decay operation
    in tensorflow using inverse time decay"""
    dec = tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return (dec)
