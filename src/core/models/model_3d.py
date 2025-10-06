# src/core/models/model_3d.py

import tensorflow as tf
from tensorflow.keras import layers, models

class CNN3D(tf.keras.Model):
    """Modèle 3D pour l'analyse des séries DICOM."""

    def __init__(self, input_shape: tuple = (64, 64, 64, 1), num_classes: int = 3):
        super(CNN3D, self).__init__()
        self.conv1 = layers.Conv3D(32, (3, 3, 3), activation="relu", input_shape=input_shape)
        self.pool1 = layers.MaxPooling3D((2, 2, 2))
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.flatten(x)
        return self.dense(x)
