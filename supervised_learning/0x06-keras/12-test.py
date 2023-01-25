#!/usr/bin/env python3
"""task 12"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """function that tests a neural network"""
    test = network.evaluate(
        data,
        labels,
        verbose=verbose
    )
    return (test)
