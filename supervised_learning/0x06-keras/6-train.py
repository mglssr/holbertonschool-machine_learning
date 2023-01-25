#!/usr/bin/env python3
"""task 6"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""
    callback = []
    if validation_data is not None and early_stopping:
        callback.append(K.callbacks.EarlyStopping(patience=patience))
    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          validation_data=validation_data,
                          shuffle=shuffle,
                          callbacks=callback)
    return (history)
