#!/usr/bin/env python3
"""task 6"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""
    callback = []
    if validation_data is not None:
        if early_stopping:
            callback.append(K.callbacks.EarlyStopping(patience=patience))
        if learning_rate_decay:
            def decayed_learning_rate(step):
                return alpha / (1 + decay_rate * step)
            callback.append(K.callbacks.LearningRateScheduler(
                decayed_learning_rate,
                verbose=1
                ))
    if save_best:
        callback.append(K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True
        ))
    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          validation_data=validation_data,
                          shuffle=shuffle,
                          callbacks=callback)
    return (history)
