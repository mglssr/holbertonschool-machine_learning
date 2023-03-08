#!/usr/bin/env python3
"""Task 0"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """pre-processes the data for your model"""
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(x_train, y_train)
    X_test, Y_test = preprocess_data(x_test, y_test)
    
    base_model = K.applications.DenseNet169(
        include_top=True,
        weights="imagenet"
    )
    base_model.trainable = False
    activation = K.activations.relu
    
    Input = K.Input(shape=(32, 32, 3))
    output = K.layers.Resizing(224, 224)(Input)
    output = base_model(output, training=False)
    output = K.layers.Flatten()(output)
    output = K.layers.Dense(500, activation=activation)(output)
    output = K.layers.Dropout(0.2)(output)
    output = K.layers.Dense(10, activation='softmax')(output)
    model = K.Model(inputs=Input, outputs=output)
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer=K.optimizers.Adam(),
        metrics=[K.metrics.BinaryAccuracy()]
    )
    
    model.fit(
        x=X_train,
        y=Y_train,
        validation_data=(X_test, Y_test),
        batch_size=300,
        epochs=5,
        verbose=True
    )
    
    model.save('cifar10.h5')
