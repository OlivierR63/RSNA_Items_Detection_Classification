# coding: utf-8


import tensorflow as tf
from tensorflow.keras import layers, models

class ModelFactory:
    @staticmethod
    def create_model(model_config: dict) -> tf.keras.Model:
        """Crée un modèle 3D pour la segmentation."""
        if model_config["type"] == "cnn3d":
            model = models.Sequential([
                layers.Conv3D(32, (3, 3, 3), activation="relu", input_shape=model_config["input_shape"]),
                layers.MaxPooling3D((2, 2, 2)),
                layers.Conv3D(64, (3, 3, 3), activation="relu"),
                layers.MaxPooling3D((2, 2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(model_config["num_classes"], activation="softmax")
            ])
            return model
        else:
            raise ValueError(f"Type de modèle inconnu: {model_config['type']}")