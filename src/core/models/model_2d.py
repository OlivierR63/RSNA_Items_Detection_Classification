# src/core/models/model_2d.py
import tensorflow as tf
from tensorflow.keras import layers, models

class YoloSegmentation2D(tf.keras.Model):
    """Modèle 2D pour la segmentation (exemple simplifié)."""

    def __init__(self, input_shape: tuple = (256, 256, 1), num_classes: int = 2):
        super(YoloSegmentation2D, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape)
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.flatten(x)
        return self.dense(x)
