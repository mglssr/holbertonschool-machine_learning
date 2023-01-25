#!/usr/bin/env python3
"""task 12"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """function that makes a prediction using a neural network"""
    output = network.predict(data, verbose=verbose)
    return (output)
