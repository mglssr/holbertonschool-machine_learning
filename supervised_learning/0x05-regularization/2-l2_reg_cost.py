#!/usr/bin/env python3
"""task 2"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """function that calculates the cost of a
    neural network with L2 regularization"""
    loss = cost + tf.losses.get_regularization_losses()
    return (loss)
