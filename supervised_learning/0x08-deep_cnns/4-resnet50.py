#!/usr/bin/env python3
"""Task 4"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """function that builds the ResNet-50 architecture
    as described in https://arxiv.org/pdf/1512.03385.pdf"""
    init = K.initializers.he_normal()
    act = {"kernel_initializer": init, "padding": "same"}

    def block(lis, prev, amount, s=2):
        o = projection_block(prev, lis, s)
        for i in range(amount - 1):
            o = identity_block(o, lis)
        return o
    X = K.Input(shape=(224, 224, 3))
    out = K.layers.Conv2D(64, 7, strides=2, **act)(X)
    out = K.layers.BatchNormalization(axis=3)(out)
    out = K.layers.Activation(K.activations.relu)(out)
    out = K.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(out)
    out = block([64, 64, 256], out, 3, 1)
    out = block([128, 128, 512], out, 4)
    out = block([256, 256, 1024], out, 6)
    out = block([512, 512, 2048], out, 3)
    out = K.layers.AveragePooling2D(7, 1,
                                    padding='valid')(out)
    out = K.layers.Dense(1000, activation='softmax',
                         kernel_initializer=init)(out)
    model = K.models.Model(inputs=X, outputs=out)
    return model
