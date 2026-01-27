# coding: utf-8

import tensorflow as tf
from typing import Optional, Dict, Tuple
from abc import ABC, abstractmethod  # Imported ABC for class inheritance
from pathlib import Path
import logging


class DicomTFRecordDataset(ABC):
    """
        Base TensorFlow Dataset class designed to load DICOM data from TFRecords.

        This abstract class defines the standard workflow for generating TFRecords
        (if they don't exist) and creating an optimized tf.data.Dataset pipeline.
        Subclasses must implement the abstract methods for data-specific logic.
    """

    def __init__(self, config: dict, logger: Optional[logging.Logger] = None) -> None:
        """
        Initializes the dataset handler and sets up file paths.

        Args:
            config (dict): The configuration dictionary containing application settings,
                           including "output_dir".
            logger : Optional logger instance. If None, creates a new one
        """
        self._config = config

        # Define the directory where the TFRecord files shall be stored.
        self._tfrecord_dir = Path(config["tfrecord_dir"])

        # Define the pattern to match all TFRecord files in the directory.
        self._tfrecord_pattern = str(self._tfrecord_dir / "*.tfrecord")

        self._logger = (
                        logger if logger is not None and isinstance(logger, logging.Logger)
                        else logging.getLogger(self.__class__.__name__)
                       )

    @abstractmethod
    def build_tf_dataset_pipeline(self, batch_size: int = 8) -> tf.data.Dataset:
        """
        Creates an optimized TensorFlow Dataset pipeline from the TFRecord files.

        The pipeline uses parallel file reading (interleave), shuffling, batching,
        and prefetching for high-performance input feeding.

        Args:
            batch_size (int): The number of elements to combine into a single batch.
                              Defaults to 8.

        Returns:
            tf.data.Dataset: An optimized Dataset object.
        """
        pass
