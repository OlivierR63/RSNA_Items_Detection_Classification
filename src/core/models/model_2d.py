# coding: utf-8

import tensorflow as tf
from tensorflow.keras import layers, Model

class Backbone2D:
    def __init__(self, model_name="MobileNetV2", input_shape=(224, 224, 3)):
        self._input_shape = input_shape
        self._model_name = model_name

        # On peut charger depuis Keras ou un modèle custom
        if self._model_name == "MobileNetV2":
            base = tf.keras.applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
        else:
            # Import externe ou modèle custom ici
            base = tf.keras.Sequential([
                layers.Input(shape=input_shape), # Define input here
                layers.Conv2D(32, 3, activation='relu')
            ])
            
        self._model = Model(inputs=base.input, outputs=base.output)

    def get_model(self):
        return self._model
