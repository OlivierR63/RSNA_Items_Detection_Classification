# coding: utf-8

import logging
import tensorflow as tf
from keras import layers, Model
import os
os.environ["KERAS_BACKEND"] = "tensorflow"


class Backbone2D:
    """
    Factory class for generating 2D feature extraction backbones.

    This class handles the instantiation of pre-trained spatial models (e.g., MobileNetV2, ResNet50)
    based on YAML configuration. It supports selective freezing for fine-tuning and
    provides metadata about output dimensions for downstream components.
    """

    def __init__(self, config: dict, logger: logging.Logger):

        self._logger = logger
        self._config = config

        try:
            models_cfg = config.get("models", None)

            if models_cfg is None:
                error_msg = (
                    "Fatal error: the parameter 'models' is required but was not found. "
                    "Please check your YAML file structure."
                )
                self._logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg)

            backbone_cfg = models_cfg.get("backbone_2d", None)
            if backbone_cfg is None:
                error_msg = (
                    "Fatal error: the parameter 'models -> backbone_2d' is required "
                    "but was not found. Please check your YAML file structure."
                )
                self._logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg)

            raw_shape = backbone_cfg.get("img_shape", None)
            if raw_shape is None:
                error_msg = (
                    "Fatal error: the parameter 'models -> backbone_2d -> img_shape' "
                    "is required but was not found. "
                    "Please check your YAML file structure."
                )
                self._logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg)

            self._input_shape = tuple(int(dim) for dim in raw_shape)

            if len(self._input_shape) != 3:
                error_msg = f"img_shape must have 3 dimensions (H,W,C), got {self._input_shape}"
                self._logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg)

            self._model_name = backbone_cfg.get("type", None)
            if self._model_name is None:
                error_msg = (
                    "Fatal error: the parameter 'models -> backbone_2d -> type' is required "
                    "but was not found. Please check your YAML file structure."
                )
                self._logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg)

            self._freeze = backbone_cfg.get("freeze", False)
            if self._freeze is None:
                error_msg = (
                    "Fatal error: the parameter 'models -> backbone_2d -> freeze' is required "
                    "but was not found. Please check your YAML file structure."
                )
                self._logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg)

            self._model = self._build_backbone()

        except Exception as e:
            msg = f"Failed to build 2D Backbone: {e}"
            self._logger.error(msg, exc_info=True)
            raise RuntimeError(msg)

    def _build_backbone(self) -> Model:

        builders = {
            "MobileNetV2": self._build_mobilenet,
            "ResNet50": self._build_resnet50
        }

        if self._model_name not in builders:
            msg_error = f"Unsupported backbone: {self._model_name}"
            self._logger.error(msg_error, exc_info=True)
            raise ValueError(msg_error)

        return builders[self._model_name]()

    def _build_mobilenet(self):
        height, width, channels = self._input_shape

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(int(height), int(width), int(channels)),
            include_top=False,
            weights="imagenet"
        )

        base_model.trainable = not self._freeze

        input_tensor = layers.Input(
            shape=(height, width, channels),
            dtype="float32",
            name="input_2d_main"
        )

        output = base_model(input_tensor, training=not self._freeze)

        return Model(
            inputs=input_tensor,
            outputs=output,
            name="backbone_2d_model"
        )

    def _build_resnet50(self):

        height, width, channels = self._input_shape

        input_tensor = layers.Input(
            shape=(height, width, channels),
            dtype="float32",
            name="input_2d_main"
        )

        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=input_tensor
        )

        base_model.trainable = not self._freeze

        # Ensure the output is returned through the functional API model
        return Model(
            inputs=input_tensor,
            outputs=base_model.output,
            name="backbone_2d_model"
        )

    def get_model(self) -> Model:
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
