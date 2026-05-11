# coding: utf-8

import logging
import tf_keras
import os
from src.config.config_loader import ConfigLoader

os.environ["KERAS_BACKEND"] = "tensorflow"


class Backbone2D:
    """
    Factory class for generating 2D feature extraction backbones.

    This class handles the instantiation of pre-trained spatial models (e.g., MobileNetV2, ResNet50)
    based on YAML configuration. It supports selective freezing for fine-tuning and
    provides metadata about output dimensions for downstream components.
    """

    def __init__(self, logger: logging.Logger):

        self._logger = logger
        self._config = ConfigLoader().get()

        try:
            models_cfg = self._config.get("models", None)
            backbone_cfg = models_cfg.get("backbone_2d", None)
            raw_shape = backbone_cfg.get("img_shape", None)

            self._input_shape = tuple(int(dim) for dim in raw_shape)

            self._model_name = backbone_cfg.get("type", None)
            self._freeze = backbone_cfg.get("freeze", False)
            self._model = self._build_backbone()

        except Exception as e:
            msg = f"Failed to build 2D Backbone: {e}"
            self._logger.error(msg, exc_info=True)
            raise RuntimeError(msg)

    def _build_backbone(self) -> tf_keras.Model:

        builders = {
            "MobileNetV2": self._build_mobilenet,
            "ResNet50": self._build_resnet50
        }

        return builders[self._model_name]()

    def _build_mobilenet(self):
        height, width, channels = self._input_shape

        base_model = tf_keras.applications.MobileNetV2(
            input_shape=(int(height), int(width), int(channels)),
            include_top=False,
            weights="imagenet"
        )

        base_model.trainable = not self._freeze

        input_tensor = tf_keras.layers.Input(
            shape=(height, width, channels),
            dtype="float32",
            name="input_2d_main"
        )

        output = base_model(input_tensor, training=not self._freeze)

        return tf_keras.Model(
            inputs=input_tensor,
            outputs=output,
            name="backbone_2d_model"
        )

    def _build_resnet50(self):

        height, width, channels = self._input_shape

        input_tensor = tf_keras.layers.Input(
            shape=(height, width, channels),
            dtype="float32",
            name="input_2d_main"
        )

        base_model = tf_keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=input_tensor
        )

        base_model.trainable = not self._freeze

        # Ensure the output is returned through the functional API model
        return tf_keras.Model(
            inputs=input_tensor,
            outputs=base_model.output,
            name="backbone_2d_model"
        )

    def get_model(self) -> tf_keras.Model:
        """
        Returns the constructed Keras Model instance
        """
        return self._model

    def get_output_shape(self):
        """
        Returns the shape of the feature produced by the backbone
        used by the 3D Aggregator to set temporal layers
        """
        return self._model.output_shape
