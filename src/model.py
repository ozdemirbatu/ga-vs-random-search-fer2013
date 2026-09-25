
import keras
from keras import layers

from . import config


# Build and compile a model from config.decode() output 
def build_model(params):
    dropout = float(params["dropout"])
    inputs = keras.Input(shape=config.INPUT_SHAPE)
    x = inputs

    for n_filters in (params["filters_1"], params["filters_2"], params["filters_3"]):
        for _ in range(2):
            x = layers.Conv2D(n_filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout * 0.5)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# Parameter count for a configuration
def count_params(params):
    model = build_model(params)
    total = model.count_params()
    keras.backend.clear_session()
    return int(total)
