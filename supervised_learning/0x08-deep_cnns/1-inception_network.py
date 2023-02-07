#!/usr/bin/env python3
"""Task 1"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """function that builds an inception network as
    described in https://arxiv.org/pdf/1409.4842.pdf"""
    init = K.initializers.he_normal()
    act = {"activation": K.activations.relu,
           "kernel_initializer": init,
           "padding": "same"}
    o = K.Input(shape=(224, 224, 3))
    conv = K.layers.Conv2D(64, 7, strides=2, **act)(o)
    max_pool = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                     padding="same")(conv)
    conv2 = K.layers.Conv2D(192, 3, strides=1, **act)(max_pool)
    max_pool2 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                      padding="same")(conv2)
    i3a = inception_block(max_pool2, [64, 96, 128, 16, 32, 32])
    i3b = inception_block(i3a, [128, 128, 192, 32, 96, 64])
    max_pool3 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                      padding="same")(i3b)
    i4a = inception_block(max_pool3, [192, 96, 208, 16, 48, 64])
    i4b = inception_block(i4a, [160, 112, 224, 24, 64, 64])
    i4c = inception_block(i4b, [128, 128, 256, 24, 64, 64])
    i4d = inception_block(i4c, [112, 144, 288, 32, 64, 64])
    i4e = inception_block(i4d, [256, 160, 320, 32, 128, 128])
    max_pool4 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                      padding="same")(i4e)
    i5a = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])
    i5b = inception_block(i5a, [384, 192, 384, 48, 128, 128])
    avg_pool = K.layers.AveragePooling2D(pool_size=7, strides=1,
                                         padding='valid')(i5b)
    dropout = K.layers.Dropout(.4)(avg_pool)
    output = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=init)(dropout)
    model = K.Model(inputs=o, outputs=output)
    return model
