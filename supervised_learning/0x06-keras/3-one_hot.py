#!/usr/bin/env python3
"""task 3"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """function that converts a label vector into a one-hot matrix"""
    one = K.utils.to_categorical(labels, num_classes=classes)
    return (one)
