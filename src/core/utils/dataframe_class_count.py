# coding: utf-8

import json
from pathlib import Path
from typing import Dict

import tensorflow as tf

from src.config.config_loader import ConfigLoader
from src.core.utils.singleton_meta import SingletonMeta


class DataFrameClassCount(metaclass=SingletonMeta):
    def __init__(self):
        # Use the actual attribute name for initialization check
        if hasattr(self, '_cache'):
            return

        # Access the existing ConfigLoader singleton instance
        cache_path = ConfigLoader().get_value('paths').get('tfrecord_metadata_cache')

        # Resolve the full path to the cache file
        self._cache = Path(cache_path).resolve()/"cache.json"

        # Load and cache the data in memory during the first instantiation
        self._severity_labels_counts = self._get()

        # Handle class imbalance by calculating balancing weights dynamically later
        self._balancing_weights = None

    def _get(self) -> Dict[str, int]:
        """
        Reads the cache file and returns the number of items linked with each
        severity label.
        """
        if not self._cache.exists():
            return {}

        try:
            with self._cache.open('r', encoding='utf_8') as f:
                # Load the full cache dictionary
                cache_data: Dict[str, int] = json.load(f) or {}

            # Extract only the relevant counts dictionary
            return cache_data.get('values_counts', {})

        except (json.JSONDecodeError, IOError):
            # return an empty dict if the file is corrupted or unreadable
            return {}

    def _calculate_balancing_weights(self) -> tf.Tensor:
        """
        Calculates class weights to compensate for dataset imbalance.

        The weights are computed using the inverse frequency strategy:
        Weight = Total_Samples / (Number_of_Classes * Class_Frequency)

        This ensures that underrepresented classes (like 'Severe') receive higher
        importance during training, preventing the model from being biased toward
        the majority class.

        Args:
            None

        Returns:
            tf.Tensor: A 1D float32 tensor containing the balancing weight for
                each class, sorted by class ID.
        """

        total = sum(self._severity_labels_counts.values())
        num_classes = len(self._severity_labels_counts)

        # Create a tensor of weights
        weights = []
        for class_id in range(num_classes):
            count = self._severity_labels_counts.get(str(class_id), 1)  # Skip division by 0
            weight = total / (num_classes*count)
            weights.append(weight)

        return tf.constant(weights, dtype=tf.float32)

    def get_balancing_weights(self):
        """
        Returns the cached severity class weights.
        """
        return self._balancing_weights

    def set_balancing_weights(self):
        """
        Sets the balancing weights for severity classes.
        This method calculates the balancing weights based on the current
        severity label counts and updates the internal state.
        """
        # Force reloading the dict from the updated cache file on disk
        self._severity_labels_counts = self._get()
        self._balancing_weights = self._calculate_balancing_weights()

    def is_balancing_weights_valid(self) -> bool:
        """
        Checks if the internal balancing weights are valid and non-empty.
        """
        if self._balancing_weights is None:
            return False

        if self._balancing_weights.shape[0] == 0:
            return False

        return True
