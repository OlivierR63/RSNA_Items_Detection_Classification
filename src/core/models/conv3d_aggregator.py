# -*- coding: utf-8 -*-

import tf_keras
from src.core.models.temporal_padding_layer import TemporalPaddingLayer
from src.core.utils.logger import get_current_logger
from src.config.config_loader import ConfigLoader


class Conv3DAggregator(tf_keras.layers.Layer):
    def __init__(self, backbone_2d_output_shape, series_depth, logger=None, **kwargs):
        """
        Initializes the 3D aggregator layer.

        This constructor extracts hyperparameters from the global configuration and
        pre-instantiates all necessary Keras layers (Conv3D, BatchNormalization,
        Reshape, and Pooling). Pre-instantiating layers here is mandatory to
        support static graph execution (run_eagerly=False) and weight sharing.

        Args:
            backbone_2d_output_shape (tuple): The shape of the feature maps produced
                by the 2D backbone, used to calculate spatial and channel dimensions.
            logger (logging.Logger): Logger instance for tracking the layer
                initialization and build status.
            series_depth (int): The fixed temporal/depth dimension (number of slices)
                required for the 3D volume.
            **kwargs: Standard Keras layer keyword arguments (e.g., 'name').
        """
        super().__init__(**kwargs)
        self._series_depth = series_depth
        self._backbone2d_output_shape = backbone_2d_output_shape
        self._config = ConfigLoader().get()
        self._logger = logger or get_current_logger()

        # 1. Extracting configuration parameters
        model_3d_cfg = self._config['models']['head_3d']
        self.filters = model_3d_cfg.get('filters', 64)
        height, width, channels = self._backbone2d_output_shape[-3:]

        # 2. LAYER INSTANTIATION (Tools)
        # No "suffix" is defined here as the instance is unique to this layer
        self.padding_layer = TemporalPaddingLayer(
            target_depth=self._series_depth,
            height=height,
            width=width,
            channels=channels
        )

        self.reshape_layer = tf_keras.layers.Reshape(
            (self._series_depth, height, width, channels)
        )

        self.conv3d_1_local = tf_keras.layers.Conv3D(
            filters=self.filters,
            kernel_size=(3, 3, 3),
            activation='relu',
            padding='same',
            name='conv3D_local'
        )

        self.conv3d_1_dilated = tf_keras.layers.Conv3D(
            filters=self.filters,
            kernel_size=(3, 3, 3),
            dilation_rate=(2, 1, 1),
            activation='relu',
            padding='same',
            name='conv3D_dilated'
        )
        self.bn_1 = tf_keras.layers.BatchNormalization()
        self.spatial_dropout = tf_keras.layers.SpatialDropout3D(0.2)
        self.concat = tf_keras.layers.Concatenate()

        self.conv3d_2 = tf_keras.layers.Conv3D(
            filters=self.filters * 2,
            kernel_size=(3, 3, 3),
            activation='relu',
            padding='same'
        )
        self.bn_2 = tf_keras.layers.BatchNormalization()

        self.global_pool = tf_keras.layers.GlobalAveragePooling3D()

    def call(self, x, training=None):
        """
        Executes the 3D aggregation logic using pre-instantiated layers.
        The argument 'training' is automatically injected by Keras.
        """
        # We just call the layers created in the function __init__
        x_lay = self.padding_layer(x)
        x_lay = self.reshape_layer(x_lay)

        x_local = self.conv3d_1_local(x_lay)
        x_dilated = self.conv3d_1_dilated(x_lay)

        # Both "local" and "dilated" views are merged
        x_lay = self.concat([x_local, x_dilated])
        x_lay = self.bn_1(x_lay)
        x_lay = self.spatial_dropout(x_lay, training=training)

        x_lay = self.conv3d_2(x_lay)
        x_lay = self.bn_2(x_lay)

        x_lay = self.global_pool(x_lay)

        return x_lay

    def get_config(self):
        config = super().get_config()
        config.update({
            "config": self._config,
            "backbone_2d_output_shape": self._backbone2d_output_shape,
            "series_depth": self._series_depth
        })
        return config
