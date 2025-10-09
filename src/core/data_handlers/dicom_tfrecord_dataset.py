# coding: utf-8


import tensorflow as tf
from typing import Dict, Tuple
from abc import ABC, abstractmethod # Imported ABC for class inheritance
from pathlib import Path

class DicomTFRecordDataset(ABC):
    """
    Base TensorFlow Dataset class designed to load DICOM data from TFRecords.

    This abstract class defines the standard workflow for generating TFRecords 
    (if they don't exist) and creating an optimized tf.data.Dataset pipeline. 
    Subclasses must implement the abstract methods for data-specific logic.
    """


    def __init__(self, config: dict) -> None:
        """
        Initializes the dataset handler and sets up file paths.

        Args:
            config (dict): The configuration dictionary containing application settings, 
                           including "output_dir".
        """
        self._config = config
        
        # Define the directory where TFRecord files will be stored.
        self._tfrecord_dir = Path(config["output_dir"]) / "tfrecords"
        
        # Define the pattern to match all TFRecord files in the directory.
        self._tfrecord_pattern = str(Path(config["output_dir"]) / "tfrecords" / "*.tfrecord")
        
        # Automatically generate the TFRecord files upon initialization if they are missing.
        self.generate_tfrecord_files()


    @abstractmethod
    def generate_tfrecord_files(self) -> None:
        """
        Abstract method. Subclasses MUST implement the logic to convert raw data 
        (e.g., DICOM files) into TFRecord files and save them to disk.
        """
        pass


    @abstractmethod
    def _parse_tfrecord(self, example_proto: tf.Tensor) -> Tuple[tf.Tensor, Dict]:
        """
        Abstract method. Subclasses MUST implement the logic to deserialize a single 
        serialized tf.train.Example string into an image tensor and a metadata dictionary.

        Args:
            example_proto (tf.Tensor): A scalar string Tensor (the serialized example).

        Returns:
            Tuple[tf.Tensor, Dict]: The processed image tensor and the metadata dictionary.
        """
        pass


    def create_tf_dataset(self, batch_size: int = 8) -> tf.data.Dataset:
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
        # 1. List all TFRecord files and shuffle the file order for better data mixing.
        tfrecord_files = tf.data.Dataset.list_files(self._tfrecord_pattern, shuffle=True)

        # 2. Use 'interleave' to read records from multiple files simultaneously and 
        #    apply the parsing function. This prevents I/O blocking.
        dataset = tfrecord_files.interleave(
            # For each file path (x), create a TFRecordDataset and map the parsing function.
            lambda x: tf.data.TFRecordDataset(x).map(
                self._parse_tfrecord,
                # Use AUTOTUNE to dynamically determine optimal parallelism.
                num_parallel_calls=tf.data.AUTOTUNE
            ),
            # Use AUTOTUNE for setting the number of concurrent readers.
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # 3. Shuffle and optimize the dataset.
        
        # Shuffle the elements in the dataset using a large buffer.
        dataset = dataset.shuffle(buffer_size=1000)
        
        # Combine individual elements into batches.
        dataset = dataset.batch(batch_size)
        
        # Prefetch the next batch while the current batch is being processed by the model.
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset