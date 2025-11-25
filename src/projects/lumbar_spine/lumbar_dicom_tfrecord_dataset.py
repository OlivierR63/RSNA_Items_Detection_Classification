# coding: utf-8

import tensorflow as tf
from typing import Dict, Tuple, List, Optional
from src.core.data_handlers.dicom_tfrecord_dataset import DicomTFRecordDataset
from src.projects.lumbar_spine.csv_metadata import CSVMetadata
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
import struct
import io
from src.core.utils.logger import log_method
import logging


class LumbarDicomTFRecordDataset(DicomTFRecordDataset):
    """
        TensorFlow Dataset for loading DICOM TFRecords.
    """

    def __init__(self, config: dict, logger: Optional[logging.Logger] = None) -> None:
        # Initialize the logger first (before calling super() to log initialization)
        self._MAX_RECORDS = 25
        self._MAX_RECORDS_FLAT = self._MAX_RECORDS * 4
        self.logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing LumbarDicomTfRecordDataset")

        super().__init__(config, logger=self.logger)  # Pass the logger to parent class

    def max_records_flat(self):
        return self._MAX_RECORDS_FLAT

    def _py_func_map_wrapper(self, tfrecord_proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Wraps the _parse_tfrecord method using tf.py_function.

        This is necessary to execute Python code (like Python logging and
        try/except blocks) reliably within the TensorFlow graph structure.
        It allows custom Python error handling during the map operation.

        The output types (Tout) must match the return types of _parse_tfrecord.
        Assuming: (image: tf.float32, metadata_string: tf.string)
        """
        return tf.py_function(
            func=self._parse_tfrecord,
            inp=[tfrecord_proto],
            # Tout must match the dtypes returned by _parse_tfrecord
            Tout=[tf.float32, tf.string]
        )

    @log_method()
    def build_tf_dataset_pipeline(self, batch_size: int = 8, *,
                                  logger: Optional[logging.Logger] = None) -> tf.data.Dataset:
        """
        Creates an optimized TensorFlow Dataset for training or evaluation by reading
        and processing TFRecord files asynchronously.

        The pipeline is constructed to maximize I/O and processing throughput
        using interleave, shuffle, and prefetch mechanisms.

        Args:
            batch_size (int): The number of elements to combine into a single batch.
                              Defaults to 8.
            logger: Automatically injected logger (optional).

        Returns:
            tf.data.Dataset: An optimized Dataset where each element is a tuple
                             (image_tensor, metadata_dict).
        """

        logger = logger or self.logger
        logger.info(f"Creating TF Dataset with batch_size={batch_size}",
                    extra={"action": "create_dataset", "batch_size": batch_size})

        try:
            # 1. List all TFRecord files matching the pattern (e.g., 'data/*.tfrecord').
            # Shuffling the file names helps ensure better data mixing across epochs.
            tfrecord_files = tf.data.Dataset.list_files(self._tfrecord_pattern, shuffle=True)
            logger.info(f"Found {len(tfrecord_files)} TFRecord files",
                        extra={"file_count": len(tfrecord_files)})

            # 2. Use "interleave" to read data from multiple files in parallel.
            # This prevents I/O bottlenecks by having multiple readers working concurrently.
            dataset = tfrecord_files.interleave(

                # The map function reads a single TFRecord file and applies the parsing function.
                lambda x: tf.data.TFRecordDataset(x).map(
                    self._py_func_map_wrapper,
                    # Use AUTOTUNE to determine the optimal number of parallel processing threads
                    num_parallel_calls=tf.data.AUTOTUNE
                ),
                # Interleave also uses AUTOTUNE
                # for setting the number of concurrent calls (readers).
                num_parallel_calls=tf.data.AUTOTUNE
            )

            # 3. Apply standard Dataset optimizations.

            # Shuffle the processed elements in memory using a buffer.
            dataset = dataset.shuffle(buffer_size=1000)

            # Combine individual elements into batches for efficient model processing.
            dataset = dataset.batch(batch_size)

            # Prefetch data: The input pipeline fetches the next batch while the model
            # is processing the current batch, preventing GPU/TPU starvation.
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            logger.info("Dataset pipeline created successfully",
                        extra={"status": "success"})

            return dataset

        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            raise

    @log_method()
    def _generate_tfrecord_files(self, *, logger: Optional[logging.Logger] = None) -> None:
        """
        Generates TFRecord files from DICOM images and associated metadata.

        This function performs the following steps:
        1. Creates the necessary output directory for TFRecord files.
        2. Loads and merges metadata from CSV files as specified in the configuration.
        3. Encodes categorical metadata fields (condition, level, series_description, severity,...)
           into numerical values for compatibility with machine learning pipelines.
        4. Converts DICOM files to TFRecord format (if no existing TFRecord files are found
            in the output directory).

        Args:
            logger: Automatically injected logger (optional).

        Notes:
            - TFRecord files are only generated if the output directory is empty.
            - The DICOM root directory and output directory are specified in the configuration.
        """

        logger = logger or self.logger
        logger.info("Starting generate_tfrecord_file", extra={"action": "generate_tf_records"})

        try:
            # 1. Prepare the directories
            self._tfrecord_dir.mkdir(parents=True, exist_ok=True)

            # 2. Load and merge metadata
            metadata_handler = CSVMetadata(
                                            logger=logger,
                                            root_dir=self._config["root_dir"],
                                            **self._config["csv_files"]
                                            )
            logger.info("Loaded metadata from CSV files",
                        extra={"csv_files": list(self._config["csv_files"].keys())})

            # 3. Encode categorical metadata
            #    Fields affected : condition, level, series_description, severity
            encoded_metadata_df = self._encode_dataframe(metadata_handler.merged)
            logger.info("Encoded categorical metadata", extra={"action": "encode_metadata"})

            # 4. Convert DICOM files to TFRecords (if needed)
            if not list(self._tfrecord_dir.glob("*.tfrecord")):
                logger.info("  Creating TFRecord files...")

                self._convert_dicom_to_tfrecords(
                    study_dir=self._config["dicom_study_dir"],
                    metadata_df=encoded_metadata_df,
                    tfrecord_dir=str(self._tfrecord_dir)
                )
                logger.info("DICOM to TFRecord conversion completed.",
                            extra={"status": "success"})
            else:
                logger.info("Existing TFRecords found. Skipping conversion.",
                            extra={"status": "skipped"})

        except Exception as e:
            logger.error(f"Error generating TFRecords: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            raise

    def _py_deserialize_and_flatten(self, metadata_bytes_tensor: tf.Tensor) -> List[tf.Tensor]:
        """
        Deserializes and flattens metadata bytes into a list of TensorFlow tensors.

        This method takes a serialized metadata byte string (typically from a TFRecord)
        and converts it into a structured list of TensorFlow tensors. The metadata is first
        deserialized into a dictionary, then split into header tensors (scalar metadata fields)
        and a flattened tensor for records (annotations or points of interest).

        The records are padded to a fixed size (self._MAX_RECORDS_FLAT) to ensure
        consistent tensor shapes for batch processing in TensorFlow pipelines.

        Args:
            metadata_bytes_tensor (tf.Tensor): A TensorFlow tensor containing the serialized
                metadata as a byte string. This tensor is typically extracted from a TFRecord
                and represents the 'metadata' field.

        Returns:
            list[tf.Tensor]: A list of 7 TensorFlow tensors:
                - The first 6 tensors are scalar `tf.int32` tensors representing:
                    0: study_id (int)
                    1: series_id (int)
                    2: instance_number (int)
                    3: description (int, encoded category)
                    4: condition (int, encoded category)
                    5: nb_records (int, number of records/annotations)
                - The 7th tensor is a 1D `tf.float32` tensor of shape (100,) representing:
                    6: Flattened and padded records (level, severity, x, y) for up to
                       self._MAX_RECORDS records.
                       Each record is represented by 4 values, and the tensor is padded with zeros
                       to ensure a fixed size of 4 * self._MAX_RECORDS elements
                       (self._MAX_RECORDS records * 4 values).

        Raises:
            ValueError: If the deserialized metadata structure is invalid or inconsistent.
            Exception: If the deserialization process fails (e.g., due to corrupted metadata bytes).
                      The exception will be logged and propagated to the caller.

        Notes:
            - This method is designed to be called within a `tf.py_function` context, allowing
              the use of Python code inside a TensorFlow graph.
            - The `.numpy()` call is required to convert the input tensor to a NumPy array for
              deserialization, as TensorFlow operations are not available in this context.
            - The padding ensures that the output tensor for records always has a fixed size,
              which is necessary for batching in TensorFlow datasets.
        """
        try:
            # We MUST call .numpy() here, inside the py_function boundary.
            deserialized_dict = self._deserialize_metadata(metadata_bytes_tensor.numpy())

            required_keys = ['study_id', 'series_id',
                             'instance_number', 'description', 'condition', 'nb_records']
            for key in required_keys:
                if key not in deserialized_dict:
                    raise ValueError(f"Missing required key in deserialized metadata: {key}")

            # 1. Extract header Tensors
            header_tensors = [
                    tf.constant(deserialized_dict.get('study_id', 0), dtype=tf.int32),
                    tf.constant(deserialized_dict.get('series_id', 0), dtype=tf.int32),
                    tf.constant(deserialized_dict.get('instance_number', 0), dtype=tf.int32),
                    tf.constant(deserialized_dict.get('description', 0), dtype=tf.int32),
                    tf.constant(deserialized_dict.get('condition', 0), dtype=tf.int32),
                    tf.constant(deserialized_dict.get('nb_records', 0), dtype=tf.int32)
                ]

            # 2. Flatten records into a 1D list of values:
            # [level1, severity1, x1, y1, level2, severity2, x2, y2, ...]
            # If no records, create a dummy 1x4 list of zeros.
            records_list = deserialized_dict.get('records', [])
            flattened_records = [val for rec in records_list for val in rec] or [-1.0] * 4

            # Pad the flattened list to the self._MAX_RECORDS_FLAT size
            # ( = self._MAX_RECORDS * 4 elements)
            # Pad with a safe value (0.0)
            padded_records = (flattened_records
                              + [0.0] * (self._MAX_RECORDS_FLAT - len(flattened_records)))
            records_tensor = tf.constant(padded_records, dtype=tf.float32)
            result = header_tensors + [records_tensor]

            return result

        except Exception as e:
            self.logger.error(f"Error in _py_deserialize_and_flatten: {str(e)}", exc_info=True)
            # Retourne une structure par défaut
            return [
                    tf.constant(0, dtype=tf.int32),  # study_id
                    tf.constant(0, dtype=tf.int32),  # series_id
                    tf.constant(0, dtype=tf.int32),  # instance_number
                    tf.constant(0, dtype=tf.int32),  # description
                    tf.constant(0, dtype=tf.int32),  # condition
                    tf.constant(0, dtype=tf.int32),  # nb_records
                    tf.constant([0.0] * 100, dtype=tf.float32)  # records
                    ]

    @log_method()
    def _parse_tfrecord(
                        self,
                        example_proto: tf.Tensor, *,
                        logger: Optional[logging.Logger] = None
                         ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Parses a single TFRecord entry, extracting and processing the image and metadata.

        This function is typically used within a tf.data pipeline to transform
        raw TFRecord strings into usable tensors and structured data.

        Args:
            example_proto (tf.Tensor): A scalar string Tensor representing one
                                       serialized tf.train.Example protocol buffer.
            logger: Automatically injected logger (optional).

        Returns:
            Tuple[tf.Tensor, Dict]: A tuple containing the processed image tensor
                                    and a dictionary of the deserialized metadata.
        """

        try:
            # Define the structure of the features stored in the TFRecord.
            feature_description = {
                "image": tf.io.FixedLenFeature([], tf.string),
                "metadata": tf.io.FixedLenFeature([], tf.string)
            }

            # Parse the scalar protocol buffer string into a dictionary of Tensors.
            example = tf.io.parse_single_example(example_proto, feature_description)

            # --- 1. Deserialize and Reshape the Image Tensor (Pure TF) ---
            image = tf.io.parse_tensor(example["image"], out_type=tf.uint16)
            # NOTE: Using tf.uint16, the original Dicom type normalization
            # to float32 happens later.

            # Reshape the 1D Tensor back into its original 4D shape
            # (e.g., [Depth, Height, Width, Channels]).
            # The shape is assumed to be fixed for this dataset.
            image = tf.reshape(image, [64, 64, 64, 1])

            # --- 2. Deserialize the Metadata (using tf.py_function) ---
            metadata_bytes = example["metadata"]

            # Define the output types for the metadata deserializer:
            # [0: study_id (int), 1: series_id (int), 2: instance_number (int),
            # 3: description (int), 4: condition (int), 5: nb_records (int),
            # 6: records (list of tuples -> will be tf.float32/tf.int32)]

            # We call the deserializer via tf.py_function, which can execute Python code.
            # It must return a consistent set of Tensors, not a dict.

            # The metadata will be returned as individual scalar Tensors
            # and one combined Tensor for records.
            # We need the inner function to return a flat list/tuple of Tensors.

            # To handle complex structures (the list of records), we must wrap the deserializer
            # and parse the records list inside the tf.py_function output, returning fixed-size
            # Tensors, which is difficult for a variable list of records.

            # The cleaner solution: Return only the most complex part (records)
            # as one flattened tensor and re-structure it later in the TF graph.

            # Define the output signature for tf.py_function
            # 6 scalar headers (tf.int32) + 1 padded records tensor (tf.float32)
            output_types = [tf.int32] * 6 + [tf.float32]

            # Call the py_function
            deserialized_tensors = tf.py_function(
                self._py_deserialize_and_flatten,
                inp=[metadata_bytes],
                Tout=output_types
            )

            # Assign the results back to named Tensors
            (
                study_id_t, series_id_t, instance_number_t, description_t,
                condition_t, nb_records_t, padded_records_t
                ) = deserialized_tensors

            # Reshape the padded records tensor to (self._MAX_RECORDS, 4)
            padded_records_t = tf.reshape(padded_records_t, [self._MAX_RECORDS, 4])

            # --- 3. Normalize the Image (Pure TF) ---
            # The image is currently uint16. Convert to float32 and normalize.
            image = tf.cast(image, dtype=tf.float32)
            # Simple normalization: scaling pixel values to the range [0, 1].
            image = image / tf.reduce_max(image)

            # --- 4. Return Processed Data ---
            return_object = (image, {
                "study_id": study_id_t,
                "series_id": series_id_t,
                "instance_number": instance_number_t,
                "description": description_t,
                "condition": condition_t,
                "nb_records": nb_records_t,
                # The records are now a (self._MAX_RECORDS, 4) float32 tensor
                "records": padded_records_t
            })

            return return_object

        except Exception as e:
            logger.error(f"Error parsing TFRecord: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            # Return a safe dummy structure to avoid pipeline crash
            dummy_image = tf.zeros([64, 64, 64, 1], dtype=tf.float32)
            dummy_records = tf.zeros([self._MAX_RECORDS, 4], dtype=tf.float32)

            return dummy_image, {
                "study_id": tf.constant(0, dtype=tf.int32),
                "series_id": tf.constant(0, dtype=tf.int32),
                "instance_number": tf.constant(0, dtype=tf.int32),
                "description": tf.constant(0, dtype=tf.int32),
                "condition": tf.constant(0, dtype=tf.int32),
                "nb_records": tf.constant(0, dtype=tf.int32),
                "records": dummy_records
            }

    @log_method()
    def _convert_dicom_to_tfrecords(self, study_dir: str, metadata_df: pd.DataFrame,
                                    tfrecord_dir: str, *,
                                    logger: Optional[logging.Logger] = None) -> None:
        """
        Converts DICOM files stored in a hierarchical directory structure into
        TensorFlow TFRecord files, generating one TFRecord file per study.

        Args:
            - study_dir (str): The path to the directory containing study subfolders.
            - metadata_df (pd.DataFrame): A DataFrame containing pre-processed metadata.
            - tfrecord_dir (str): The directory where the resulting TFRecord files will be saved.
            - logger: Automatically injected logger (optional).

        Returns:
            None: The function saves files to disk but returns nothing.
        """
        logger = logger or self.logger
        logger.info("Starting DICOM to TFRecord conversion",
                    extra={"action": "convert_dicom", "dicom_study_dir": study_dir})

        try:
            # Ensure the destination directory for TFRecord files exists and return its Path.
            tfrecord_path = self._setup_tfrecord_directory(tfrecord_dir)

            # Iterate over all items (expected study directories) in the root study path.
            for study_path in tqdm(list(Path(study_dir).iterdir())):
                if not study_path.is_dir():
                    msg_warning = (
                                    f"Skipping non-directory item {study_path} "
                                    f"in study folder {study_dir}"
                                   )
                    logger.warning(msg_warning)
                    continue

                study_metadata_df = metadata_df[metadata_df['study_id'] == int(study_path.name)]

                if study_metadata_df.empty:
                    msg_warning = (
                                    f"No metadata found for study {study_path.name}. "
                                    f"Skipping TFRecord generation for this study."
                                   )
                    logger.warning(msg_warning)

                    logger.warning(
                                    "This study will not be considered "
                                    "during training or evaluation."
                                   )

                    logger.warning(
                                    "This may be due to missing "
                                    "or inconsistent records in the CSV files."
                                   )

                    msg_warning = (
                                    "Please check the CSV files "
                                    "and ensure they contain the right records."
                                   )
                    logger.warning(msg_warning)
                    continue

                self._process_study(study_path, study_metadata_df, tfrecord_path)

            logger.info("DICOM to TFRecord conversion completed successfully",
                        extra={"status": "success"})

        except Exception as e:
            logger.error(f"Error during DICOM conversion: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            raise

    def _setup_tfrecord_directory(self, tfrecord_dir: str) -> Path:
        """
            Setup the TFRecord directory, creating it if it doesn't exist.
        """
        tfrecord_path = Path(tfrecord_dir)
        tfrecord_path.mkdir(parents=True, exist_ok=True)
        return tfrecord_path

    @log_method()
    def _process_study(self, study_path: Path, metadata_df: pd.DataFrame,
                       tfrecord_dir: Path, logger: Optional[logging.Logger] = None) -> None:
        """
            Iterates over the series directories within a given study directory.

            Each series directory is checked and then processed by `_process_single_series_instance`
            to extract DICOM data and write it as features into the TFRecord file
            designated for the study. Non-directory items are skipped and logged.

            Args:
                - study_path: The path to the root directory of the study.
                - metadata_df: DataFrame containing metadata relevant to the study.
                - tfrecord_dir: The target directory for the output TFRecord file.
                                This directory is supposed to already exist. No specific checking
                                is done here.
                - logger: Logger instance for logging warnings.
        """
        logger = logger or self.logger

        study_id = study_path.name

        logger.info(f"Starting processing study {study_id}",
                    extra={"action": "process_study", "study_dir": study_path})

        tfrecord_path = tfrecord_dir / f"{study_id}.tfrecord"

        nb_skipped_series = 0

        with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
            for series_path in study_path.iterdir():
                if not self._process_single_series_instance(series_path, metadata_df, writer):
                    nb_skipped_series += 1
                    continue

        if nb_skipped_series > 0:
            logger.warning(
                f"Study {study_id} processed with {nb_skipped_series} skipped series "
                "due to missing metadata or aborted TFRecord file generation.",
                extra={"status": "partial_success", "skipped_series": nb_skipped_series}
            )
        else:
            logger.info(
                f"Study {study_id} processing completed successfully.",
                extra={"status": "success"}
            )

    def _process_single_series_instance(
                                        self,
                                        series_path: Path,
                                        metadata_df: pd.DataFrame,
                                        writer: tf.io.TFRecordWriter,
                                        logger: Optional[logging.Logger] = None
                                       ) -> None:
        """
            Processes a single series directory within a study.
            This method checks if the provided series path is a directory.
            If it is, it filters the metadata DataFrame for entries corresponding
            to the series ID. If matching metadata is found, it delegates the
            processing of the series to `_process_series`. Non-directory items
            are skipped with a warning.

            Args:
                - series_path: The path to the series directory.
                - metadata_df: DataFrame containing metadata relevant to the study.
                - writer: The active TFRecordWriter instance for the current study.
                - logger: Logger instance for logging warnings.


            Returns:
                bool: True if the series was processed (meaning it was a valid directory with
                      metadata, and `_process_series` was called). False if the series was
                      skipped (non-directory item, missing metadata, or complete processing failure
                      in `_process_series`).
        """
        logger = logger or self.logger

        study_path = series_path.parent
        study_id = study_path.name

        if not series_path.is_dir():
            msg_warning = (
                            f"\nSkipping non-directory item: {series_path} "
                            f"in study: {study_path}"
                           )
            logger.warning(msg_warning)
            return False

        series_id = int(series_path.name)
        series_metadata_df = metadata_df[metadata_df['series_id'] == series_id]

        if series_metadata_df.empty:
            warning_message = (
                f"Skipping series {series_path.name} in study {study_id}: "
                "No matching metadata found.\n"
                "-> Consequence: This series will not be considered "
                "during training or evaluation.\n"
                "-> Root Cause: This may be due to missing "
                "or inconsistent records in the CSV files.\n"
                "-> Action: Please check the CSV files "
                "and ensure they contain the right records."
            )
            logger.warning(warning_message,
                           extra={"status": "metadata_missing",
                                  "series_dir": series_path.name,
                                  "study_id": study_id})
            return False

        is_successful = not self._process_series(series_path, series_metadata_df, writer)

        return is_successful

    @log_method()
    def _process_series(
                            self, series_path: Path,
                            metadata_df: pd.DataFrame,
                            writer: tf.io.TFRecordWriter,
                            logger: Optional[logging.Logger] = None
                        ) -> None:
        """
            Processes all DICOM files within a single series directory.

            For each DICOM file (`.dcm`), this method filters the provided metadata
            DataFrame using the file's `instance_number` (stem). If matching metadata
            is found, the process continues by delegating file reading, metadata
            serialization to `_process_dicom_file`, and feature writing to
            `_write_tfrecord_example`. Files without corresponding metadata are
            skipped, and a warning is logged.

            Args:
                - series_path: The path to the series directory.
                - metadata_df: DataFrame containing metadata specific to this series.
                - writer: The active TFRecordWriter instance for the current study.

            Returns:
                bool: True if the processing of the series resulted in a *complete failure*
                        (i.e., zero successful TFRecord examples were written).
                      False otherwis (meaning full or partial success).
        """
        logger = logger or self.logger

        logger.info(f"Starting processing series {series_path.name}",
                    extra={"action": "process_series", "series_dir": series_path})

        nb_skipped_files = 0
        nb_failed_process = 0
        nb_success_file = 0

        for dicom_path in series_path.glob("*.dcm"):
            (metadata_ok, process_initiated_and_aborted) = self._process_single_dicom_instance(
                                                                dicom_path, metadata_df, writer
            )

            if metadata_ok is False:
                nb_skipped_files += 1
                continue

            if process_initiated_and_aborted is True:
                nb_failed_process += 1
                continue

            nb_success_file += 1

        full_success: bool = (nb_skipped_files == 0 and nb_failed_process == 0)
        partial_success: bool = (nb_success_file > 0 and not full_success)
        complete_failure: bool = (nb_success_file == 0)

        if complete_failure:
            logger.error(
                f"Series {series_path.name} processing failed: "
                f"All files were skipped or failed during processing.",
                extra={"status": "failed"}
            )

        if partial_success:
            total_unprocessed = nb_skipped_files + nb_failed_process
            logger.warning(
                f"Series {series_path.name} partially processed ({nb_success_file} success). "
                f"Skipped {total_unprocessed} files in total: "
                f"({nb_skipped_files} due to missing metadata; "
                f"{nb_failed_process} due to processing errors).",
                extra={"status": "partial_success",
                       "skipped_metadata": nb_skipped_files,
                       "failed_processing": nb_failed_process}
            )

        if full_success:
            logger.info(
                f"Series {series_path.name} processing completed successfully.",
                extra={"status": "success"}
            )

        return complete_failure

    @log_method()
    def _process_single_dicom_instance(
                                        self,
                                        dicom_path: Path,
                                        metadata_df: pd.DataFrame,
                                        writer: tf.io.TFRecordWriter,
                                        logger: Optional[logging.Logger] = None
                                       ) -> bool:
        """
            Processes a single DICOM file within a series.

            This method filters the provided metadata DataFrame using the DICOM file's
            `instance_number` (derived from the file name). If matching metadata is found,
            it delegates the processing of the DICOM file and metadata serialization to
            `_process_dicom_file`, and writes the resulting features to the TFRecord
            using `_write_tfrecord_example`. If no matching metadata is found, it logs
            a warning and returns False.

            Args:
                - dicom_path: The path to the DICOM file.
                - metadata_df: DataFrame containing metadata specific to the series.
                - writer: The active TFRecordWriter instance for the current study.
                - series_path: The path to the parent series directory (for logging context).
                - logger: Optional logger instance.

            Returns:
                Tuple[bool, bool]: A tuple `(metadata_ok, process_aborted)`.
                    - `metadata_ok`: True if matching metadata was found for the DICOM file.
                      False otherwise (file was skipped).
                    - `process_aborted`: True if metadata was found, but the processing
                      (`_process_dicom_file` or `_write_tfrecord_example`) failed
                      due to an exception. False otherwise (success or skipped due to metadata).
        """
        logger = logger or self.logger
        series_path = dicom_path.parent

        dicom_metadata_df = metadata_df[metadata_df['instance_number'] == int(dicom_path.stem)]
        if dicom_metadata_df.empty:
            # Consolidated Warning: Use newlines (\n) to separate the points
            # while maintaining a single log event.
            warning_message = (
                f"Skipping DICOM file {dicom_path.name} in series {series_path.name}. "
                "No matching metadata found.\n"
                "-> Consequence: This file will not be considered during training or evaluation.\n"
                "-> Action: Please check the CSV files for missing or inconsistent records."
            )
            logger.warning(warning_message)

            metadata_OK = False
            process_initiated_and_aborted = False
            return metadata_OK, process_initiated_and_aborted

        try:
            # Process the DICOM file and its metadata
            img_bytes, serialized_metadata = self._process_dicom_file(dicom_path, dicom_metadata_df)

            # Write the result to the TFRecord
            self._write_tfrecord_example(img_bytes, serialized_metadata, writer)

            metadata_OK = True
            process_initiated_and_aborted = False

            return metadata_OK, process_initiated_and_aborted

        except Exception:
            metadata_OK = True
            process_initiated_and_aborted = True

            return metadata_OK, process_initiated_and_aborted

    @log_method()
    def _process_dicom_file(
                                self,
                                dicom_path: Path,
                                metadata_df: pd.DataFrame,
                                logger: Optional[logging.Logger] = None
                            ) -> Tuple[bytes, bytes]:
        """
            Process a single DICOM file and return serialized image and metadata.

            Args:
                - dicom_path: The path to the DICOM file.
                - metadata_df: DataFrame containing metadata for this instance.
                - logger: Optional logger instance.

            Returns:
                Tuple[bytes, bytes]: Serialized image tensor and serialized metadata.

            Raises:
                Exception: If reading, converting, or serializing the DICOM image fails.
        """
        logger = logger or self.logger

        logger.info(f"Processing DICOM file {dicom_path}",
                    extra={"action": "process_dicom_file", "dicom_file": dicom_path})

        try:
            img = sitk.ReadImage(str(dicom_path))
            img_array = sitk.GetArrayFromImage(img)
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint16)
            img_bytes = tf.io.serialize_tensor(img_tensor).numpy()

            serialized_metadata = self._serialize_metadata(metadata_df)

            logger.info(f"Successfully processed DICOM file {dicom_path}",
                        extra={"status": "success"})

            return img_bytes, serialized_metadata

        except Exception as e:

            logger.error(
                f"Error during DICOM file read/conversion/serialization for {dicom_path.name}: "
                f"{str(e)}",
                exc_info=True,  # Includes the full stack trace upon failure
                extra={"status": "failed", "error_type": "DicomProcessingError"}
            )
            raise

    @log_method()
    def _write_tfrecord_example(
                                    self, img_bytes: bytes,
                                    serialized_metadata: bytes,
                                    writer: tf.io.TFRecordWriter,
                                    logger: Optional[logging.Logger] = None
                                ) -> None:
        """
            Write a single TFRecord example to the writer.

            Args:
                - img_bytes: Serialized image tensor bytes.
                - serialized_metadata: Serialized metadata bytes.
                - writer: The active TFRecordWriter instance.

            Returns:
                None: The method writes the example directly to the provided writer.

            Raises:
                Exception: If an error occurs during the creation or writing 
                           of the TFRecord example.
        """
        logger = logger or self.logger

        feature = {
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
            "metadata": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[serialized_metadata])
            )
        }
        try:
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        except Exception as e:
            logger.error(f"Error writing TFRecord example: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            raise

    @log_method()
    def _get_metadata_for_file(self, file_path: str, metadata_df: pd.DataFrame, *,
                               logger: Optional[logging.Logger] = None) -> bytes:
        """
            Returns the serialized metadata extracted from a dataframe and associated
            with a given DICOM file.

            This method derives the necessary identifiers (study ID, series ID, instance number)
            from the file path structure and uses these to query and serialize the relevant
            records from the comprehensive metadata DataFrame.

            Args:
                file_path (str): The full path to the DICOM file (used to derive IDs).
                metadata_df (pd.DataFrame): The main DataFrame containing all metadata records.
                logger: Automatically injected logger (optional).

            Returns:
                bytes: The compact byte sequence representing the serialized metadata
                       for the specific image. **Returns an empty byte sequence (b'') if**
                       **the input DataFrame is None or if an exception occurs during processing.**
        """

        logger = logger or self.logger
        logger.info("Starting retrieving metadata from CSV files")

        try:
            if metadata_df is None:
                # NOTE: Returning bytes instead of a dict, as the caller
                # (_convert_dicom_to_tfrecords) expects serialized bytes for the TFRecord.
                return b''

            # Call the core serialization function, passing the extracted IDs
            # and the full DataFrame.
            # This will filter the DataFrame and create the compact byte sequence.
            serialized_metadata = self._serialize_metadata(metadata_df)

            # Return the byte sequence.
            return serialized_metadata

            logger.info("CSV metadata retrieval completed successfully",
                        extra={"status": "success"})

        except Exception as e:
            logger.error(f"Error in function _get_metadata_for_file() : {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            return b''

    @log_method()
    def _serialize_metadata(self,
                            data_df: pd.DataFrame, *,
                            logger: Optional[logging.Logger] = None) -> bytes:
        """
            Serializes metadata records (header + payload) for a single DICOM instance
            into a structured binary format.

            This function is responsible for taking a subset of the main DataFrame
            corresponding to a specific DICOM image (defined by study_id, series_id,
            and instance_number) and converting its associated metadata (e.g.,
            lesion data) into a compact binary representation suitable for TFRecords.

            The structure of the resulting bytes object is:
            [Header (15 bytes)] + [Payload (8 bytes * N records)]

            Args:
                records_df (pd.DataFrame): A DataFrame containing all records
                    (e.g., all lesions/annotations) associated with a single
                    instance_number, series_id, and study_id.
                logger (Logger): The logger instance used for logging status and errors.

            Returns:
                bytes: The complete serialized metadata block (header + payload).

            Raises:
                Exception: If null values are detected in the `records_df`.
                    **Logging Change**: If nulls are found, the full error message is
                    logged via three separate calls to `logger.error` before
                    the exception is raised, which includes the list of affected columns.
                ValueError: If the input `records_df` is empty, or if the number of
                    records exceeds the maximum allowed limit (`self._MAX_RECORDS`).
                RuntimeError: If any underlying serialization step fails.
        """
        logger = logger or self.logger
        logger.info("Starting function _serialize_metadata")

        if data_df.isnull().any().any():
            # Get columns that contain null values
            null_columns = data_df.columns[data_df.isnull().any()].tolist()

            msg_error_part1 = "Null values detected in data_df before serialization."
            msg_error_part2 = f"Columns affected: {null_columns}."
            msg_error_part3 = "Serialization might fail or produce corrupted records."

            logger.error(msg_error_part1)
            logger.error(msg_error_part2)
            logger.error(msg_error_part3)

            # Build the full message to raise the exception
            full_msg_error = f"{msg_error_part1} {msg_error_part2} {msg_error_part3}"
            raise ValueError(full_msg_error)

        try:

            header_bytes = self._serialize_header(data_df)
            payload_bytes = self._serialize_payload(data_df)

            metadata_bytes = header_bytes + payload_bytes

            msg_info = "Function _serialize_metadata completed successfully"
            logger.info(msg_info, extra={"status": "success"})
            return metadata_bytes

        except Exception as e:
            logger.error(f"Error in function _serialize_metadata(): {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            raise

    def _serialize_header(self, records_df: pd.DataFrame) -> bytes:
        """
            Serializes the header part of the metadata.

            Raises:
                ValueError: If records_df is empty or None, or if the number of records
                            exceeds the defined limit (self._MAX_RECORDS) or if global
                            properties (like study_id) are not unique.
        """
        # Consolidated check for empty or None DataFrame
        if records_df is None or records_df.empty:
            raise ValueError("Cannot serialize header: Input DataFrame is None or empty.")

        # VALIDATION: Ensure NO Null Values in Header Columns ---
        HEADER_COLS = [
                        'study_id',
                        'series_id',
                        'instance_number',
                        'condition',
                        'series_description'
                        ]
        if records_df[HEADER_COLS].isnull().any().any():
            # Isolate the header columns (HEADER_COLS)
            header_df = records_df[HEADER_COLS]

            # Determine which header columns contain null values
            header_null_mask = header_df.isnull().any()

            # Retrieve the names of the concerned columns as a list
            null_cols = header_df.columns[header_null_mask].tolist()

            raise ValueError(
                                f"Cannot serialize header: "
                                f"Null values detected in critical header columns: {null_cols}. "
                                "Serialization requires all header values to be non-null integers."
                            )

        # The header requires a single, unique value for these identifiers,
        # as the DataFrame should represent records for one specific DICOM file.
        for col in ['study_id', 'series_id', 'instance_number', 'condition', 'series_description']:
            if len(records_df[col].unique()) != 1:
                raise ValueError(
                                    f"Cannot serialize header: "
                                    f"Column '{col}' must contain exactly one unique value, "
                                    f"but found {len(records_df[col].unique())}."
                                )

        study_id_str = str(records_df['study_id'].unique()[0])
        series_id_str = str(records_df['series_id'].unique()[0])
        instance_number_str = str(records_df['instance_number'].unique()[0])

        # Extract global properties (must be unique for the filtered records)
        condition = int(records_df['condition'].unique()[0])
        description = int(records_df['series_description'].unique()[0])

        # Serialize study_id and series_id as 5-byte unsigned integers (big-endian)
        study_id_bytes = int(study_id_str).to_bytes(5, byteorder='big', signed=False)
        series_id_bytes = int(series_id_str).to_bytes(5, byteorder='big', signed=False)

        # Serialize instance_number as a 2-byte unsigned short (max 9999)
        instance_number_bytes = int(instance_number_str).to_bytes(2, byteorder='big', signed=False)

        # Serialize description and condition as 1-byte unsigned characters
        description_bytes = description.to_bytes(1, byteorder='big', signed=False)
        condition_bytes = condition.to_bytes(1, byteorder='big', signed=False)

        # Serialize the number of records (max self._MAX_RECORDS) as a 1-byte unsigned character
        nb_records = len(records_df)
        if nb_records > self._MAX_RECORDS:
            raise ValueError(f"The number of records exceeds the limit of {self._MAX_RECORDS}.")

        nb_records_bytes = nb_records.to_bytes(1, byteorder='big', signed=False)

        serialized_bytes = (
                                study_id_bytes + series_id_bytes + instance_number_bytes
                                + description_bytes + condition_bytes + nb_records_bytes
                            )

        return serialized_bytes

    def _serialize_payload(self, records_df: pd.DataFrame) -> bytes:
        """
            Serializes the payload part of the metadata by iterating over individual
            records (rows) in the DataFrame.

            Each record is converted into an 8-byte binary sequence containing the
            level, severity, and scaled x/y coordinates.

            Raises:
                ValueError: If records_df is empty or None, or if any critical payload
                            columns contain null values (NaN).
        """

        # Consolidated check for empty or None DataFrame
        if records_df is None or records_df.empty:
            raise ValueError("Cannot serialize payload: Input DataFrame is None or empty.")

        # Ensure NO Null Values in Payload Columns ---
        PAYLOAD_COLS = ['level', 'severity', 'x', 'y']
        if records_df[PAYLOAD_COLS].isnull().any().any():
            # Isolate the columns of interest for the payload
            payload_df = records_df[PAYLOAD_COLS]

            # Determine which of these columns contain at least one null value
            null_mask = payload_df.isnull().any()

            # Get the names of the columns where the mask is True
            null_cols = payload_df.columns[null_mask].tolist()

            raise ValueError(
                "Cannot serialize payload: "
                f"Null values detected in critical payload columns: {null_cols}. "
                f"Serialization requires all payload values to be non-null."
            )

        payload_bytes = b''

        for row in records_df.itertuples():

            level = row.level
            severity = row.severity

            # Convert float coordinates to integers by scaling (multiplication by 100)
            # This conversion is guaranteed to be robust because null values (NaN)
            # have been explicitly checked and rejected at the beginning of this function.
            x = round(float(row.x) * 100)
            y = round(float(row.y) * 100)

            # Serialize level and severity as 1-byte unsigned characters
            level_bytes = struct.pack('=B', level)
            severity_bytes = struct.pack('=B', severity)

            # Serialize scaled x and y as 3-byte unsigned integers (big-endian)
            x_bytes = x.to_bytes(3, byteorder='big', signed=False)
            y_bytes = y.to_bytes(3, byteorder='big', signed=False)

            # Each record provide an additional 8-bytes long payload
            payload_bytes += level_bytes + severity_bytes + x_bytes + y_bytes

        return payload_bytes

    @log_method()
    def _deserialize_metadata(self, metadata_bytes: bytes, *,
                              logger: Optional[logging.Logger] = None) -> Dict:
        """
            Deserializes a compact byte sequence back into structured metadata components.
            This function is the inverse of _serialize_metadata.
            Args:
                metadata_bytes: The byte sequence containing the serialized metadata.
                logger: Automatically injected logger (optional).
            Returns:
                dict: A dictionary containing the deserialized header values and a list of tuples
                      (level, severity, x, y) for the individual records.
            Raises:
                ValueError: If the input is not a byte sequence
                            or if the buffer length is insufficient.
        """
        if not isinstance(metadata_bytes, bytes):
            raise ValueError("Input must be a byte sequence")

        if not metadata_bytes:
            raise ValueError("Input byte sequence is empty")

        MINIMUM_BUFFER_LENGTH = 31  # Requested minimal length (header + 1 record)
        if len(metadata_bytes) < MINIMUM_BUFFER_LENGTH:
            msg_error = ("Invalid buffer length: expected at least ",
                         f"{MINIMUM_BUFFER_LENGTH} bytes, got {len(metadata_bytes)}")
            raise struct.error(msg_error)

        try:
            buffer = io.BytesIO(metadata_bytes)
            header = self._deserialize_header(buffer)
            records = self._deserialize_records(buffer, header['nb_records'])
            return {**header, 'records': records}

        except Exception as e:
            logger = logger or self.logger
            logger.error(f"Error in function _deserialize_metadata: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})

            raise Exception(f"Error deserializing metadata: {str(e)}")

    def _deserialize_header(self, buffer: io.BytesIO) -> Dict:
        """
            Deserializes the header part of the metadata.
        """
        study_id = int.from_bytes(buffer.read(5), byteorder='big', signed=False)
        series_id = int.from_bytes(buffer.read(5), byteorder='big', signed=False)
        instance_number = int.from_bytes(buffer.read(2), byteorder='big', signed=False)
        description = int.from_bytes(buffer.read(1), byteorder='big', signed=False)
        condition = int.from_bytes(buffer.read(1), byteorder='big', signed=False)
        nb_records = int.from_bytes(buffer.read(1), byteorder='big', signed=False)

        return {
            'study_id': study_id,
            'series_id': series_id,
            'instance_number': instance_number,
            'description': description,
            'condition': condition,
            'nb_records': nb_records
        }

    def _deserialize_records(self, buffer: io.BytesIO, nb_records: int) -> List[Tuple]:
        """
            Deserializes the records part of the metadata.
        """
        records = []

        for _ in range(nb_records):
            level = int.from_bytes(buffer.read(1), byteorder='big', signed=False)
            severity = int.from_bytes(buffer.read(1), byteorder='big', signed=False)

            x_scaled = int.from_bytes(buffer.read(3), byteorder='big', signed=False)
            x = x_scaled / 100.0  # Rescale back to float

            y_scaled = int.from_bytes(buffer.read(3), byteorder='big', signed=False)
            y = y_scaled / 100.0  # Rescale back to float

            records.append((level, severity, x, y))

        return records

    @log_method()
    def _encode_dataframe(self, metadata_df: pd.DataFrame, *,
                          logger: Optional[logging.Logger] = None) -> pd.DataFrame:
        """
            Converts categorical textual metadata fields in a DataFrame to numerical values.
            This process is essential for preparing data for serialization
            or machine learning models,replacing descriptive strings with compact integers.

            Args:
                metadata_df: The DataFrame containing metadata to be converted.
                            Example column values: {
                                                        "condition": "Spinal Canal Stenosis",
                                                        "level": "L1-L2", ...
                                                    }

            Returns:
                pd.DataFrame: The DataFrame with the specified columns converted to integer values.
        """
        logger = logger or self.logger
        logger.info("Starting function _encode_dataframe")

        try:
            # Define columns to encode and their corresponding mapping variables
            columns_to_encode = ["condition", "level", "series_description", "severity"]

            # Create mappings for each column
            mappings = self._create_mappings(metadata_df, columns_to_encode)

            # Apply encoding to each column
            metadata_df = self._apply_encodings(metadata_df, columns_to_encode, mappings)

            msg_info = "Function _encode_dataframe completed successfully"
            logger.info(msg_info, extra={"status": "success"})
            return metadata_df

        except Exception as e:
            logger.error(f"Error in function _encode_dataframe: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            raise

    def _create_mappings(self, metadata_df: pd.DataFrame,
                         columns_to_encode: Dict[str, str]) -> Dict[str, Dict]:
        """
            Creates mapping dictionaries for each categorical column.

            Args:
                metadata_df: The DataFrame containing metadata to be converted.
                columns_to_encode: Dictionary mapping column names to their mapping variable names.

            Returns:
                Dict[str, Dict]: A dictionary of mapping dictionaries for each column.
        """
        mappings = {}

        for column in columns_to_encode:
            values = metadata_df[column].unique().tolist()
            mappings[column] = self._create_string_to_int_mapper(values).mapping

        return mappings

    def _apply_encodings(self, metadata_df: pd.DataFrame,
                         columns_to_encode: List[str],
                         mappings: Dict[str, Dict]) -> pd.DataFrame:
        """
            Applies the encoding mappings to each specified column in the DataFrame.

            Args:
                metadata_df: The DataFrame containing metadata to be converted.
                columns_to_encode: Dictionary mapping column names to their mapping variable names.
                mappings: Dictionary of mapping dictionaries for each column.

            Returns:
                pd.DataFrame: The DataFrame with the specified columns converted to integer values.
        """
        for column in columns_to_encode:
            metadata_df[column] = metadata_df[column].map(mappings[column]).fillna(-1).astype(int)

        return metadata_df

    @log_method()
    def _create_string_to_int_mapper(self, strings_lst: list, *,
                                     logger: Optional[logging.Logger] = None) -> callable:
        """
            Creates a mapping function between strings and integers.

            This is a factory function that generates a callable (the mapper)
            capable of converting predefined category strings into their corresponding
            integer indices (starting from 0).

            Args:
                strings_lst (list): A list of strings to be mapped (e.g., ["Normal", "Stenosis"])
                                The order of the strings defines their integer value.

            Returns:
                callable: A function that maps an input string to an integer.
                          The resulting function includes 'mapping' and 'reverse_mapping
                          dictionaries as attributes.
        """

        logger = logger or self.logger
        logger.info("Starting function _create_string_to_int_mapper")

        try:
            # Create the primary dictionary {string: integer} using enumeration.
            # The integer (i) corresponds to the string's index in the list.
            mapping = {key: idx for idx, key in enumerate(strings_lst)}

            # Create the optional reverse dictionary {integer: string}
            # for inspection/deserialization.
            reverse_mapping = {idx: key for idx, key in enumerate(strings_lst)}

            def mapper(key: str) -> int:
                """Maps an input string to its corresponding integer index."""
                # Use .get() to return the mapped integer.
                # Returns -1 if the input string is not found (unknown category).
                return mapping.get(key, -1)

            # Attach the mapping dictionaries as attributes to the mapper function.
            # This allows users to inspect or reverse the mapping later.
            mapper.mapping = mapping
            mapper.reverse_mapping = reverse_mapping

            # Return the callable function
            logger.info("Function _create_string_to_int_mapper completed successfully",
                        extra={"status": "success"})
            return mapper

        except Exception as e:
            error_msg = f"Error in function _create_string_to_int_mapper : {str(e)}"
            logger.error(error_msg, exc_info=True, extra={"status": "failed", "error": str(e)})
            raise
