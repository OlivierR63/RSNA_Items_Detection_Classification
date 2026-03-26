# -*- coding: utf-8 -*-

from keras import layers
from src.core.models.temporal_padding_layer import TemporalPaddingLayer


class Conv3DAggregator(layers.Layer):
    def __init__(self, config, backbone_2d_output_shape, logger, series_depth, **kwargs):
        """
        Initializes the 3D aggregator layer.

        This constructor extracts hyperparameters from the global configuration and
        pre-instantiates all necessary Keras layers (Conv3D, BatchNormalization,
        Reshape, and Pooling). Pre-instantiating layers here is mandatory to
        support static graph execution (run_eagerly=False) and weight sharing.

        Args:
            config (dict): Global configuration dictionary containing 'models/head_3d'
                hyperparameters (e.g., filter counts).
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
        self._config = config
        self._logger = logger

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

        self.reshape_layer = layers.Reshape(
            (self._series_depth, height, width, channels)
        )

        self.conv3d_1 = layers.Conv3D(
            filters=self.filters,
            kernel_size=(3, 3, 3),
            activation='relu',
            padding='same'
        )
        self.bn_1 = layers.BatchNormalization()

        self.conv3d_2 = layers.Conv3D(
            filters=self.filters * 2,
            kernel_size=(3, 3, 3),
            activation='relu',
            padding='same'
        )
        self.bn_2 = layers.BatchNormalization()

        self.global_pool = layers.GlobalAveragePooling3D()

    def call(self, x, suffix=""):
        """
        Executes the 3D aggregation logic using pre-instantiated layers.
        """
        # We just call the layers created in the function __init__
        x_lay = self.padding_layer(x)
        x_lay = self.reshape_layer(x_lay)

        x_lay = self.conv3d_1(x_lay)
        x_lay = self.bn_1(x_lay)

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
