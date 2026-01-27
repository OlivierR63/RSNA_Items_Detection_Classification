# coding: utf-8

import tensorflow as tf
from tensorflow.keras import layers

class BaseAggregator:
    def __init__(self, config, logger):
        self._config = config
        self._logger = logger

    def build(self, x):
        raise NotImplementedError("Each aggregator shall implement its own build() method")

    def get_config(self):
        """Returns the base configuration for serialization."""
        return {
            "config": self._config,
            "logger": None
        }