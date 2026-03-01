# coding: utf-8

import tensorflow as tf
import logging
from typing import Tuple, Optional, Dict
from src.projects.lumbar_spine.tfrecord_files_manager import TFRecordFilesManager
from src.core.utils.dataset_utils import parse_tfrecord_single_element, process_study_multi_series, format_for_model
from pathlib import Path


class LumbarDicomTFRecordDataset():
    """
        TensorFlow Dataset for loading DICOM TFRecords.
    """

    def __init__(
        self,
        config: dict,
        logger: Optional[logging.Logger] = None,
        series_depth: int = 1
    ) -> None:

        """
        Initializes the dataset loader with study-specific constraints.

        Args:
            config (dict): Global configuration dictionary containing:
                - 'max_records': Number of anatomical levels to analyze (e.g., 25).
                - 'dataset_buffer_size_mb': Memory buffer for TFRecord reading.
                - 'nb_cores': CPU cores to allocate for parallel processing.
                - 'model_2d/img_shape': Target dimensions for image rescaling.

            logger (logging.Logger, optional): Custom logger instance. 
                Defaults to a class-level logger if None.

            series_depth (int): Number of slices per 3D volume (e.g., 15). 
                Determines the depth dimension of the reconstructed tensors.
        """

        self._logger = (
            logger if logger is not None and isinstance(logger, logging.Logger)
                   else logging.getLogger(self.__class__.__name__)
        )
        self._config = config
        self._MAX_RECORDS = config['max_records']
        self._series_depth = series_depth
        self._logger.info("Initializing LumbarDicomTfRecordDataset")

    def generate_tfrecord_dataset(
        self,
        tfrecord_list,
        batch_size
    ) -> tf.data.Dataset:

        """
        Constructs a specialized tf.data.Dataset pipeline for 3D lumbar spine analysis.

        This pipeline orchestrates the transformation of raw 2D DICOM frames stored in 
        TFRecords into structured multi-input 3D volumes (T1, T2, Axial).

        The pipeline executes the following sequence:
        1. Interleaves TFRecord files to stream individual study records.
        2. Parses TFRecord Protobuf examples into image tensors and raw metadata.
        3. Batches records by study size (3 * series_depth) to group all frames for 
           the three required MRI series.
        4. Reconstructs, sorts, and validates 3D volumes to ensure spatial consistency 
           across Sagittal T1, Sagittal T2, and Axial T2 series.
        5. Formats the processed volumes and labels into a multi-input dictionary 
           signature compatible with the model's architecture.

        Args:
            - tfrecord_list (list): List of file paths to the TFRecord files.
            - batch_size (int): Number of studies (patients) per training batch.
            - logger (logging.Logger, optional): Logger for tracking pipeline initialization.

        Returns:
            tf.data.Dataset: A dataset yielding tuples of (inputs_dict, targets_dict), 
                             where inputs_dict contains the 3D volumes for each series.
        """
        
        # Extract records from files (Flattening the nested structure)
        BYTES_PER_MIB = 1024*1024
        buffer_mb = int(self._config.get('dataset_buffer_size_mb', 100))
        buffer_size_bytes= buffer_mb * BYTES_PER_MIB # in MiB

        tfrecord_files_manager = TFRecordFilesManager(self._config, self._logger)
        tfrecord_files_manager.set_series_depth(self._series_depth)

        total_slices_per_patient =3*self._series_depth

        # 1. Main Pipeline: Iterate through the list of TFRecord files
        dataset = tf.data.Dataset.from_tensor_slices(tfrecord_list)

        # 2. Shuffle the file list to avoid overfitting to file order.
        dataset = dataset.shuffle(buffer_size=len(tfrecord_list))

        # 3. Infinite repeat for training
        dataset = dataset.repeat()
        
        # 4. Interleave multiple TFRecord files to overlap I/O and preprocessing latency
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, buffer_size=buffer_size_bytes),
            cycle_length=2,         # Read multiple TFRecord files at once
            block_length=total_slices_per_patient, # Read the entire patient before switching
            num_parallel_calls=tf.data.AUTOTUNE,  # Parallelize the IO
            deterministic=False      # Keep order predictable
        )

        # 5. Parse individual TFRecord elements (serialized frames)
        dataset = dataset.map(parse_tfrecord_single_element, num_parallel_calls=tf.data.AUTOTUNE)

        # 6. Skip corrupted records automatically
        dataset = dataset.ignore_errors() 
        
        # 7. Group all records found in the file (1 file = 1 patient)
        # We don't drop remainder here to catch all available frames
        dataset = dataset.batch(total_slices_per_patient)
        
        # 8. Reconstruct 3D volumes (sorting and padding)
        # This is the most CPU-intensive step, distributed across cores
        dataset = dataset.map(lambda images, metadata, labels: process_study_multi_series(images, metadata, labels),
                        num_parallel_calls=tf.data.AUTOTUNE)

        # 9. Final formatting into model-ready dictionary
        # Formatting inside interleave allows for immediate cleanup of intermediate tensors
        dataset = dataset.map(lambda volumes, study_id, label: format_for_model(volumes, study_id, label),
                        num_parallel_calls=tf.data.AUTOTUNE)
        
        # 10. Batching several studies (patients) together. 
        # Note: Set batch_size carefully based on GPU VRAM (3D volumes are heavy).
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        # 11. Prefetch the next batch in the background to hide latency
        dataset = dataset.prefetch(1)

        return dataset
