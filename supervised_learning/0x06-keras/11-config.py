#!/usr/bin/env python3
"""task 10"""
import tensorflow.keras as K


def save_config(network, filename):
    """function saves a model's configuratoin in JSON format"""
    json_config = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(json_config)


def load_config(filename):
    """function that loads a model with a specific configuration"""
    with open(filename) as json_file:
        json_config = json_file.read()
    new_model = K.models.model_from_json(json_config)
    return (new_model)
