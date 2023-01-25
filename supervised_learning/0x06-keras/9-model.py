#!/usr/bin/env python3
"""task 8"""
import tensorflow.keras as K


def save_model(network, filename):
    """funtion that saves an entire model"""
    network.save(filename)


def load_model(filename):
    """function that loads an entire model"""
    return (K.models.load_model(filename))
