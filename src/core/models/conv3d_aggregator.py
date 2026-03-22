# -*- coding: utf-8 -*-

from keras import layers
from src.core.models.temporal_padding_layer import TemporalPaddingLayer


class Conv3DAggregator(layers.Layer):
    def __init__(self, config, backbone_2d_output_shape, logger, series_depth, **kwargs):
        """
        Initializes the 3D aggregator as a functional Keras Layer.

        Args:
            config (dict): Configuration dictionary containing model hyperparameters.
            backbone_2d_output_shape (tuple): The output shape of the 2D backbone
                (used to define the 3D volume dimensions).
            logger (logging.Logger): Logger instance for tracking the build process.
            series_depth (int): The fixed number of frames/slices expected for 3D operations.
            **kwargs: Standard Keras layer keyword arguments (e.g., name).
        """
        super().__init__(**kwargs)
        self._series_depth = series_depth
        self._backbone2d_output_shape = backbone_2d_output_shape
        self._config = config
        self._logger = logger

    def call(self, x, suffix=""):
        """
        Executes the 3D aggregation logic to condense sequence features.

        This method pads the temporal dimension, reshapes the tensor into a 5D
        volume, applies successive 3D convolutions for spatio-temporal feature
        extraction, and finally reduces the volume via Global Average Pooling.

        Args:
            x (tf.Tensor): Input tensor from the 2D backbone.
            suffix (str): Identifier for consistent layer naming across branches.

        Returns:
            tf.Tensor: Aggregated feature vector ready for the global fusion layer.
        """
        # 1. Prepare static dimensions
        # Using negative indexing ensures we grab the spatial/channel dims
        # regardless of whether the shape tuple includes the batch dimension.
        height, width, channels = self._backbone2d_output_shape[-3:]

        # 2. Use the custom layer instead of Lambda
        x_lay = TemporalPaddingLayer(
            target_depth=self._series_depth,
            height=height,
            width=width,
            channels=channels,
            name=f"temporal_padding_fixed_{suffix}"
        )(x)

        x_lay = layers.Reshape(
            (self._series_depth, height, width, channels),
            name=f"shape_anchor_{suffix}"
        )(x_lay)

        # 3. 3D Convolutional blocks
        model_3d_cfg = self._config['models']['head_3d']
        filters = model_3d_cfg.get('filters', 64)

        # Now Conv3D can safely call .as_list() on its input shape
        # First 3D convolutional layer
        x_lay = layers.Conv3D(
            filters=filters,
            kernel_size=(3, 3, 3),
            activation='relu',
            padding='same',
            name=f"conv3d_1_{suffix}"
        )(x_lay)

        x_lay = layers.BatchNormalization(name=f"bn_3d_1_{suffix}")(x_lay)

        # Second 3D Convolutional layer
        x_lay = layers.Conv3D(
            filters=filters*2,
            kernel_size=(3, 3, 3),
            padding='same',
            activation='relu',
            name=f"conv3d_2_{suffix}"
        )(x_lay)

        x_lay = layers.BatchNormalization(name=f"bn_3d_2_{suffix}")(x_lay)

        # 4. Global Dimensionality Reduction
        # Condenses the 3D volume into a feature vector for the final classifier
        x_lay = layers.GlobalAveragePooling3D(name=f"global_pool_3d_{suffix}")(x_lay)

        return x_lay

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.
        This allows the model to be saved and reloaded with all hyperparameters.
        """
        # Retrieve the configuration from the parent class
        # (often empty if the superclass does not implement it)
        config = super().get_config()

        # ALL the arguments required by __init__ are included here
        config.update({
            "config": self._config,
            "backbone_2d_output_shape": self._backbone2d_output_shape,
            "series_depth": self._series_depth
        })
        return config
