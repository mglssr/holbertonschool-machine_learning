#!/usr/bin/env python3
"""Task 6"""
import tensorflow.keras as K


def lenet5(X):
    """function that builds a modified version of the LeNet-5 architecture
    using keras"""
    init = K.initializers.he_normal()
    act = {"activation": K.activations.relu, "kernel_initializer": init}
    o = K.layers.Conv2D(filters=6, kernel_size=5, padding='same', **act)(X)
    o = K.layers.MaxPooling2D(pool_size=2, strides=2)(o)
    o = K.layers.Conv2D(filters=16, kernel_size=5, padding='valid', **act)(o)
    o = K.layers.MaxPooling2D(pool_size=2, strides=2)(o)
    o = K.layers.Flatten()(o)
    o = K.layers.Dense(120, **act)(o)
    o = K.layers.Dense(84, **act)(o)
    o = K.layers.Dense(10, kernel_initializer=init)(o)
    softmax = K.layers.Softmax()(o)
    model = K.Model(inputs=X, outputs=softmax)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return (model)
