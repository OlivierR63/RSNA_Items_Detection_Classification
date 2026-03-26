# coding: utf-8

import tensorflow as tf
import logging
from typing import Optional, List
from src.core.utils.dataset_utils import (
    parse_tfrecord_single_element,
    process_study_multi_series,
    format_for_model
)


class LumbarDicomTFRecordDataset():
    """
        TensorFlow Dataset for loading DICOM TFRecords.
    """

    def __init__(
        self,
        config: dict,
        logger: Optional[logging.Logger] = None,
        series_depth: int = 1,
    ) -> None:

        """
        Initializes the dataset loader with study-specific constraints.

        Args:
            config (dict): Global configuration dictionary containing:
                - 'max_records': Number of anatomical levels to analyze (e.g., 25).
                - 'dataset_buffer_size_mb': Memory buffer for TFRecord reading..
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
        self._MAX_RECORDS = config['data_specs']['max_records_per_frame']
        self._series_depth = series_depth

        # We define the epoch tracker here.
        # Using a tf.Variable is mandatory to allow the TF Graph to pick up
        # changes during training without rebuilding the whole dataset.
        self._current_epoch_var = tf.Variable(
            0,
            dtype=tf.int64,
            trainable=False,
            name="dataset_epoch_counter"
        )

        self._logger.info("LumbarDicomTFRecordDataset initialized")

    def generate_tfrecord_dataset(
        self,
        tfrecord_list: List[str],
        batch_size: int,
        is_training: bool = True
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
            - config (dict): project's settings
            - is_training (bool): flag on the current mode : training (True) or validation (False)

        Returns:
            tf.data.Dataset: A dataset yielding tuples of (inputs_dict, targets_dict),
                             where inputs_dict contains the 3D volumes for each series.
        """

        # Check the list of TFRecord file is not empty
        nb_tfrecord_files = len(tfrecord_list)

        if nb_tfrecord_files == 0:
            self._logger.warning("The TFRecord file list is empty. Returning an empty dataset.")
            # Returns an empty dataset with the correct structure/signature
            return tf.data.Dataset.from_tensor_slices([])

        # Extract records from files (Flattening the nested structure)
        BYTES_PER_MIB = 1024*1024
        buffer_mb = int(self._config['data_specs'].get('dataset_buffer_size_mb', 100))
        buffer_size_bytes = buffer_mb * BYTES_PER_MIB  # in MiB

        total_slices_per_patient = 3*self._series_depth

        steering_cfg = self._config.get('dataset_steering')

        interleave_cfg = steering_cfg.get('interleave')
        cycle_length = interleave_cfg.get('parallel_files')
        block_length = interleave_cfg.get('block_per_file') * total_slices_per_patient
        deterministic = interleave_cfg.get('deterministic')

        num_parallel_calls = (
            tf.data.AUTOTUNE if steering_cfg.get('num_parallel_calls') == -1
            else steering_cfg.get('num_parallel_calls')
        )
        group_studies = steering_cfg.get('group_studies') * total_slices_per_patient
        prefetch_batches = steering_cfg.get('prefetch_batches')
        use_cache = steering_cfg.get('use_cache')

        # 1. Main Pipeline: Iterate through the list of TFRecord files
        dataset = tf.data.Dataset.from_tensor_slices(tfrecord_list)

        if use_cache:
            dataset = dataset.cache()
            self._logger.info("Dataset caching ENABLED")
        else:
            self._logger.info("Dataset caching DISABLED")

        # 2. Shuffle the file list to avoid overfitting to file order.
        dataset = dataset.shuffle(buffer_size=len(tfrecord_list))

        # 4. Interleave multiple TFRecord files to overlap I/O and preprocessing latency
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, buffer_size=buffer_size_bytes),
            cycle_length=cycle_length,         # Read multiple TFRecord files at once
            block_length=block_length,  # Read the entire patient before switching
            num_parallel_calls=num_parallel_calls,  # Parallelize the IO
            deterministic=deterministic     # Keep order predictable
        )

        # 5. Parse individual TFRecord elements (serialized frames)
        dataset = dataset.map(
            lambda x: parse_tfrecord_single_element(x, self._current_epoch_var, self._config),
            num_parallel_calls=num_parallel_calls
        )

        # 6. Skip corrupted records automatically
        dataset = dataset.ignore_errors()

        # 7. Group all records found in the file (1 file = 1 patient)
        # We don't drop remainder here to catch all available frames
        dataset = dataset.batch(group_studies)

        # 8. Reconstruct 3D volumes (sorting and padding)
        # This is the most CPU-intensive step, distributed across cores
        dataset = dataset.map(
            lambda images, metadata, labels: process_study_multi_series(
                images, metadata, labels, config=self._config, is_training=is_training
            ),
            num_parallel_calls=num_parallel_calls
        )

        # 9. Final formatting into model-ready dictionary
        # Formatting inside interleave allows for immediate cleanup of intermediate tensors
        dataset = dataset.map(
            lambda volumes, study_id, label: format_for_model(
                volumes, study_id, label, self._config
            ),
            num_parallel_calls=num_parallel_calls
        )

        # 10. Batching several studies (patients) together.
        # Note: Set batch_size carefully based on GPU VRAM (3D volumes are heavy).
        dataset = dataset.batch(batch_size, drop_remainder=True)

        # 11. Ensure the dataset provides an infinite stream of batches
        dataset = dataset.repeat()

        # 12. Prefetch the next batch in the background to hide latency
        dataset = dataset.prefetch(prefetch_batches)

        return dataset
