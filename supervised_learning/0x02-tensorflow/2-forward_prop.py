#!/usr/bin/env python3
"""task 2"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ that creates the forward propagation
    graph for the neural network"""
    prev = x
    for ly in range(1, len(layer_sizes) - 1):
        layer = create_layer(prev, layer_sizes[ly], activations[ly])
        prev = layer
    return (layer)
