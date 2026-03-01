# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
from src.core.models.model_3d import BaseAggregator
from src.core.models.temporal_padding_layer import TemporalPaddingLayer


class Conv3DAggregator(BaseAggregator):
    def __init__(self, config, logger, series_depth):
        """
        Initializes the 3D aggregator.
            - param config: Configuration dictionary containing model hyperparameters.
            - param logger: Logger instance for tracking the build process.
            - param max_series_depth: The fixed number of frames/slices expected by the 3D Conv layers.
        """
        super().__init__(config, logger)
        self._series_depth = series_depth

    def _dynamic_padding(self, tensor):
        # Input shape: [Batch, Sequence, Height, Width, Channels]
        shape = tf.shape(tensor)
        current_sequence_length = shape[1]

        # Calculate how many frames are missing
        padding_needed = tf.maximum(0, self._series_depth - current_sequence_length)

        # Pad only the 'Sequence' dimension (index 1) at the end
        # Padding format: [[before, after], [before, after], ...]
        paddings = [[0, 0], [0, padding_needed], [0, 0], [0, 0], [0, 0]]
        
        padded_tensors = tf.pad(tensor, paddings, mode='CONSTANT', constant_values=0)

        padded_tensors.set_shape([None, self._series_depth, 7, 7, 1280])

        # Clip the sequence if it exceeds target_depth and fix the static shape
        return padded_tensors[:, :self._series_depth, :, :, :]


    def _pad_and_fix_shape(self, x):
        # Prepare static dimensions
        input_cfg = self._config['model_3d']['input_shape']
        height, width, channels = input_cfg[0], input_cfg[1], input_cfg[2]

        # Apply the padding
        x_padded = self._dynamic_padding(x)

        # Instant forcing of the static shape
        x_fixed = tf.ensure_shape(x_padded, [None, self._series_depth, height, width, channels])

        return x_fixed


    def build(self, x, suffix=""):
        """
        Builds the 3D aggregation network architecture using purely symbolic
        TensorFlow operations to ensure graph-level compatibility.
        """
        # 1. Prepare static dimensions 
        input_cfg = self._config['model_3d']['input_shape']
        height, width, channels = input_cfg[0], input_cfg[1], input_cfg[2]

        # 2. Use the custom layer instead of Lambda
        # This replaces _pad_and_fix_shape and _dynamic_padding
        x_lay = TemporalPaddingLayer(
            target_depth=self._series_depth,
            height=height,
            width=width,
            channels=channels,
            name=f"temporal_padding_fixed_{suffix}"
        )(x)

        x_lay = layers.Reshape((self._series_depth, height, width, channels), name=f"shape_anchor_{suffix}")(x_lay)

        # 3. 3D Convolutional blocks
        filters = self._config.get('filters', 64)

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
        # Retrieve the configuration from the parent class (often empty if the superclass does not implement it)
        config = super().get_config()
        
        # ALL the arguments required by __init__ are included here
        config.update({
            "series_depth": self._series_depth
        })
        return config
