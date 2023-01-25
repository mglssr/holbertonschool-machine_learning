#!/usr/bin/env python3
"""task 10"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """function that saves a model's weights"""
    if not filename.endswith(save_format):
        filename += save_format
    network.save_weights(filename)


def load_weights(network, filename):
    """function that loads a model's weights"""
    network.load_weights(filename)
