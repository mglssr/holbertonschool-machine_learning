#!/usr/bin/env python3
"""task 6"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""
    if validation_data != None and early_stopping == True:
        callback = K.callbacks.EarlyStopping(monitor="val_loss",
                                             patience=patience)
    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          validation_data=validation_data,
                          shuffle=shuffle,
                          callbacks=[callback])
    return (history)