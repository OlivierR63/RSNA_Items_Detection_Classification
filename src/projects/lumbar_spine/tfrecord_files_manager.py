# coding: utf-8

from argparse import ONE_OR_MORE
from typing import Dict, Tuple, List, Optional
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import inspect
from src.core.utils.logger import log_method
from src.core.utils.dataset_utils import calculate_max_series_depth
from src.projects.lumbar_spine.csv_metadata_handler import CSVMetadataHandler
from pathlib import Path
from tqdm import tqdm


# ----------------------- Helper functions -------------------------------
def _int64_feature(value: int) -> tf.train.Feature:
    """
    Returns an int64_list from a single integer.
    Used for IDs, instance numbers, and dimensions.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value: list[int]) -> tf.train.Feature:
    """
    Returns an int64_list from a list of integers.
    Used for shape/format tuples like [Width, Height].
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bool_feature(value: bool) -> tf.train.Feature:
    """
    Returns an int64_feature from a boolean.
    The boolean is cast to int (0 or 1) for serialization.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _float_list_feature(value: list[float]) -> tf.train.Feature:
    """
    Returns a float_list from a list of floats.
    Used for the flattened 'records' containing coordinates and severity.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """
    Returns a bytes_list from a raw bytes string.
    Used for storing the raw pixel data of the DICOM image.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# -------------------------------------------------------------------------

class EmptyDirectoryError(Exception):
    """
    Exception raised when a required directory is empty.
    """
    pass


class SeriesProcessingError(Exception):
    """
    Exception raised when some series failed to be processed
    """
    pass


class TFRecordFilesManager:
    """
    Manager responsible for orchestrating the conversion of medical imaging studies 
    from DICOM format to TensorFlow Records (TFRecords).

    This class handles directory navigation, pixel data normalization at the series level, 
    and the alignment of medical metadata with anatomical labels to produce 
    standardized inputs for deep learning models.
    """

    def __init__(
        self,
        config: dict,
        logger: logging.Logger
    ) -> None:

        """
        Initializes the TFRecordFilesManager with configuration and logging.

        Args:
            config (Any): Configuration object containing project paths, 
                processing parameters, and constants like MAX_RECORDS.
            logger (Optional[logging.Logger]): Logger instance for tracking 
                the conversion process. If None, a default logger will be used.
        """

        class_name = self.__class__.__name__
        path_str = config.get('tfrecord_dir', None)

        if path_str is None:
            error_msg = f"{class_name} initialization failed: the key 'tfrecord_dir'' is missing in the configuration file"
            raise ValueError(error_msg)

        self._tfrecord_dir = Path(path_str)
        self._config = config
        self._logger = logger

        self._MAX_RECORDS = self._config.get('max_records', 0)
        if self._MAX_RECORDS <= 0:
            error_msg = (
                f"{class_name} initialization failed: the 'max_record' key is missing "
                f"or invalid (value: {self._MAX_RECORDS}) in the configuration file"
            )
            
            raise ValueError(error_msg)

        # This attribute will be set later, when the instance method self.generate_tfrecord_files is called
        self._max_series_depth = calculate_max_series_depth(logger=self._logger)

    @log_method()
    def generate_tfrecord_files(self, *, logger: Optional[logging.Logger] = None) -> None:
        """
        Generates TFRecord files from DICOM images and associated metadata.

        This function performs the following steps:
        1. Creates the necessary output directory for TFRecord files.
        2. Loads and merges metadata from CSV files as specified in the configuration.
        3. Encodes categorical metadata fields into numerical values for compatibility 
           with machine learning pipelines. Key fields include:
           - Observed Pathology (formerly "condition") and spinal level, both condensed in a single
             feature called "condition_level".
           - Series Description
           - Severity
        4. Converts DICOM files to TFRecord format (if no existing TFRecord files are found
            in the output directory).

        Args:
            logger: Automatically injected logger (optional).

        Notes:
            - TFRecord files are only generated if the output directory is empty.
            - The DICOM root directory and output directory are specified in the configuration.
        """

        logger = logger or self._logger
        logger.info("Starting generate_tfrecord_file", extra={"action": "generate_tf_records"})

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        try:
            # Convert DICOM files to TFRecords (if needed)
            if not list(self._tfrecord_dir.glob("*.tfrecord")):

                # 1. Prepare the directories
                self._tfrecord_dir.mkdir(parents=True, exist_ok=True)

                # 2. Load and merge metadata
                metadata_handler = CSVMetadataHandler(
                    config = self._config,
                    logger=logger,
                    root_dir=self._config["root_dir"],
                    dicom_studies_dir = self._config['dicom_studies_dir'],
                    **self._config["csv_files"]
                )

                metadata_df = metadata_handler.generate_metadata_dataframe()
                logger.info("Loaded metadata from CSV files",
                            extra={"csv_files": list(self._config["csv_files"].keys())})

                logger.info("  Creating TFRecord files...")

                self._convert_dicom_to_tfrecords(
                    studies_dir=self._config["dicom_studies_dir"],
                    metadata_df=metadata_df,
                    tfrecord_dir=str(self._tfrecord_dir)
                )
                logger.info("DICOM to TFRecord conversion completed.",
                            extra={"status": "success"})

                #self._max_series_depth = calculate_max_series_depth(self._logger)

            else:
                #if self._max_series_depth is None:
                    #self._max_series_depth = calculate_max_series_depth(self._logger)

                logger.info("Existing TFRecords found. Skipping conversion.",
                            extra={"status": "skipped"})

        except Exception as e:
            error_msg = (
                f"function {class_name}.{func_name} failed."
                f"Error generating TFRecords: {str(e)}"
            )

            logger.error(
                error_msg,
                exc_info=True,
                extra={"status": "failed", "error": str(e)}
            )
            raise

    @log_method()
    def _convert_dicom_to_tfrecords(self, studies_dir: str, metadata_df: pd.DataFrame,
                                    tfrecord_dir: str, *,
                                    logger: Optional[logging.Logger] = None) -> None:
        """
        Converts DICOM files stored in a hierarchical directory structure into
        TensorFlow TFRecord files, generating one TFRecord file per study.

        Args:
            - studies_dir (str): The full path to the root directory containing study subdirectories.
            - metadata_df (pd.DataFrame): A DataFrame containing pre-processed metadata.
            - tfrecord_dir (str): The directory where the resulting TFRecord files will be saved.
            - logger: Automatically injected logger (optional).

        Returns:
            None: The function saves files to disk but returns nothing.

        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        logger = logger or self._logger
        logger.info(f"Starting {class_name}.{func_name}",
                    extra={"action": "convert_dicom", "dicom_studies_dir": studies_dir})

        # Case empty root directory
        if not any(Path(studies_dir).iterdir()):
            error_msg = f"No studies found in {studies_dir}. Process stops immediately."
            logger.critical(error_msg, extra={"status": "failure"})
            raise EmptyDirectoryError(error_msg)

        try:
            # Ensure the destination directory for TFRecord files exists and return its Path.
            tfrecord_path = self._setup_tfrecord_directory(tfrecord_dir)

            # Iterate over all items (expected study directories) in the root studies_dir.
            for study_full_path in tqdm(list(Path(studies_dir).iterdir())):

                # Case study_full_path is not a directory
                if not study_full_path.is_dir():
                    msg_warning = (
                        f"Skipping non-directory item {study_full_path} "
                        f"in root directory {studies_dir}"
                    )
                    logger.warning(msg_warning)
                    continue

                # Case directory study_full_path is empty
                if not any(study_full_path.iterdir()):
                    msg_warning = (
                        f"Skipping empty directory {study_full_path} in root directory {studies_dir}"
                    )
                    logger.warning(msg_warning, extra={"status": "Skipped studies"})
                    continue

                study_metadata_df = metadata_df[metadata_df['study_id'] == int(study_full_path.name)]

                # Case missing metadata 
                if study_metadata_df.empty:
                    msg_warning = (
                        f"Skipping study {study_full_path.name} due to missing metadata. "
                        "This study will not be considered during training or evaluation "
                        "and the relevant TFRecord file will not be generated."
                        "Possible cause: missing or inconsistent records in the CSV files. "
                        "Required action: Please check the CSV files and ensure they contain the right records."
                    )
                    logger.warning(msg_warning)
                    continue

                self._process_study(study_full_path, study_metadata_df, tfrecord_path)

            logger.info("DICOM to TFRecord conversion completed successfully",
                        extra={"status": "success"})

        except Exception as e:
            error_msg = (
                f"function {class_name}.{func_name} failed."
                f"Error during DICOM conversion: {str(e)}"
            )

            logger.error(
                error_msg,
                exc_info=True,
                extra={"status": "failed", "error": str(e)}
            )
            raise

    def _setup_tfrecord_directory(
        self,
        tfrecord_dir: str
    ) -> Path:
        """
            Setup the TFRecord directory, creating it if it doesn't exist.
        """
        tfrecord_path = Path(tfrecord_dir)
        tfrecord_path.mkdir(parents=True, exist_ok=True)
        return tfrecord_path

    @log_method()
    def _process_study(
        self,
        study_path: Path,
        metadata_df: pd.DataFrame,
        tfrecord_dir: Path,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Orchestrates the processing of all series within a study and saves them into a single TFRecord.

        Args:
            - study_path (Path): Path to the study directory.
            - metadata_df (pd.DataFrame): DataFrame containing metadata for the study.
            - tfrecord_dir (Path): Destination directory for the generated TFRecord file.
            - logger (Optional[logging.Logger]): Logger instance for status reporting.

        Raises:
            - EmptyDirectoryError: If no items are found in the study directory.
            - SeriesProcessingError: If one or more series fail to process correctly.

        Return: None
        """
        
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        
        logger = logger or self._logger
        logger.info(f"Starting {class_name}.{func_name}",
                    extra={"action": "process_study", "study_dir": study_path})

        study_id = study_path.name

        tfrecord_path = tfrecord_dir / f"{study_id}.tfrecord"

        nb_skipped_series = 0
        columns = ['study_id', 'series_id', 'description', 'actual_file_format']
        input_features_df = metadata_df[columns].drop_duplicates()
        labels_df = metadata_df[['condition_level', 'severity', 'x', 'y']].drop_duplicates()

        try:
            series_list = list(study_path.iterdir())
            nb_series = len(series_list)

            if (nb_series) == 0:
                error_msg = (
                    f"Issue in function {class_name}.{func_name}. "
                    f"Study {study_path.name} is empty."
                )
                logger.error(
                    error_msg,
                    extra={"status": "failure", "nb_series": nb_series, "skipped_series": nb_series}
                )

                raise EmptyDirectoryError(error_msg)

            # Case nb_series > 0
            with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
                for series_path in series_list:
                    # Case series_full_path is not a directory
                    if not series_path.is_dir():
                        msg_error = (
                            f"Skipping non-directory item {series_path.name} "
                            f"in study directory {study_path.name}"
                        )
                        logger.error(msg_error)
                        continue

                    nb_success, _, _ = self._process_single_series_instance(
                        series_path,
                        input_features_df,
                        labels_df,
                        writer
                    )

                    if nb_success == 0:
                        nb_skipped_series += 1
                        continue


            if nb_skipped_series > 0 :
                error_msg = (f"Issue in function {class_name}.{func_name}."
                    f"Study {study_id} processed with {nb_skipped_series} skipped series "
                    "due to missing metadata or aborted TFRecord file generation."
                )
                logger.error(
                    error_msg,
                    extra={"status": "failure", "nb_series": nb_series, "skipped_series": nb_skipped_series}
                )

                raise SeriesProcessingError(error_msg)

            else: # last case : nb_series > 0 and nb_skipped_series == 0
                logger.info(
                    f"Function {class_name}.{func_name}: study {study_id} processing completed successfully.",
                    extra={"status": "success", "nb_series": nb_series, "skipped_series": nb_skipped_series}
                )

        except Exception as e:
            # Log the specific study failure before re-raising
            logger.error(
                f"Failed to process study {study_id}: {str(e)}",
                exc_info=True,
                extra={"study_id": study_id, "status": "failed"}
            )
            raise  # Keep propagating the error


    @log_method()
    def _process_single_series_instance(
        self,
        series_path: Path,
        input_features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        writer: tf.io.TFRecordWriter,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[int, int, int]:

        """
        Coordinates the processing of a single series by validating its directory 
        structure and metadata availability.

        This method performs preliminary checks: it verifies that the series path is 
        a valid directory and ensures that corresponding metadata exists in the 
        provided DataFrame. If successful, it delegates the heavy lifting to 
        `_process_series`.

        Args:
            series_path (Path): Path to the specific series directory.
            input_features_df (pd.DataFrame): DataFrame containing input features 
                (study_id, series_id, description).
            labels_df (pd.DataFrame): DataFrame containing ground truth labels 
                (coordinates and severity).
            writer (tf.io.TFRecordWriter): The active TFRecord writer for the study.
            logger (Optional[logging.Logger]): Logger for tracking warnings and errors.

        Returns:
            Tuple[int, int, int]: A tuple containing:
                - nb_success: Number of successfully processed DICOM instances.
                - nb_failures: Number of instances that failed during processing.
                - nb_files: Total number of DICOM files found in the series.

        Raises:
            Exception: Re-raises any exception encountered during `_process_series` 
                after logging the failure.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        
        logger = logger or self._logger
        logger.info(f"Starting {class_name}.{func_name}",
                    extra={"action": "_process_single_series_instance", "series_path": series_path})

        study_path = series_path.parent
        study_id = study_path.name

        if not series_path.is_dir():
            msg_warning = (
                f"Issue in function {func_name}.{class_name}: "
                f"\nSkipping non-directory item: {series_path} "
                f"in study: {study_path}"
            )
            logger.warning(msg_warning)
            return 0, 0, 0

        series_id = int(series_path.name)
        series_metadata_df = input_features_df[input_features_df['series_id'] == series_id]

        if series_metadata_df.empty:
            warning_message = (
                f"Issue in function {func_name}.{class_name}: "
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
            dicom_files_list = list(series_path.glob("*.dcm"))
            nb_files = len(dicom_files_list)
            return 0, 0, nb_files

        try:
            nb_success, nb_failures, nb_dcm_files = self._process_series(series_path, input_features_df, labels_df, writer)
            return nb_success, nb_failures, nb_dcm_files

        except Exception as e:
            # Log the failure of the specific series instance before re-raising
            logger.error(
                f"Failed to process series instance {series_id} in study {study_id}: {str(e)}",
                exc_info=True,
                extra={"study_id": study_id, "series_id": series_id, "status": "failed"}
            )
            raise  # Keep propagating the error

    @log_method()
    def _process_series(
        self,
        series_path: Path,
        input_features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        writer: tf.io.TFRecordWriter,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[int, int, int]:

        """
        Processes all DICOM instances within a single series directory.

        This method first computes global statistics for the series (min/max pixel values) 
        to ensure consistent normalization across all instances. It then iterates through 
        each DICOM file, delegating the extraction and serialization to 
        `_process_single_dicom_instance`.

        Args:
             - series_path (Path): Path to the series directory containing DICOM files.
             - input_features_df (pd.DataFrame): DataFrame containing input features metadata.
             - labels_df (pd.DataFrame): DataFrame containing target labels and coordinates.
             - writer (tf.io.TFRecordWriter): The active TFRecord writer for the current study.
             - logger (Optional[logging.Logger]): Logger instance for status reporting.

        Returns:
            Tuple[int, int, int]: A tuple containing:
                - nb_success: Number of DICOM files successfully serialized.
                - nb_failures: Number of files that encountered errors during processing.
                - nb_dicom_files: Total number of DICOM files discovered in the directory.

        Raises:
            Exception: Propagates any critical error encountered during series-level 
                operations (e.g., stats computation) after logging.
                
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger

        info_msg = (
            f"Starting function {class_name}.{func_name} "
            f"for series {series_path.name}"
        )

        logger.info(
           info_msg,
           extra={"action": "process_series", "series_dir": series_path}
        )

        nb_failures = 0
        nb_success = 0

        try:
            # Define the min / max pixel values of the series
            series_min, series_max = self._get_series_stats(series_path)
        
            dicom_files_list = list(series_path.glob("*.dcm"))
            nb_dicom_files = len(dicom_files_list)

            if nb_dicom_files != self._max_series_depth:
                dcm_files_instances_numbers_list = [int(Path(dcm_file).stem) for dcm_file in dicom_files_list]
                max_inst = np.max(dcm_files_instances_numbers_list)
                min_inst = np.min(dcm_files_instances_numbers_list)

                # Create a list of objects (Path or int)
                final_elements_list = dicom_files_list.copy()
                target = self._max_series_depth - nb_dicom_files

                # Step 1: Fill internal gaps (In-between existing instances)
                for instance in range(min_inst + 1, max_inst):
                    if instance not in dcm_files_instances_numbers_list:
                        final_elements_list.append(instance) # Add instance number as sent
                        target -=1

                    if 0 == target: break

                # Step 2: Fill external gaps (Head and tail)
                if target > 0:
                    nb_inserted_files = min(max_inst -1, int(target) // 2)
                    nb_appended_files = target - nb_inserted_files

                    # Insert at the beginning (Head)
                    for idx in range(nb_inserted_files):
                        # Using negative or specific range to avoid collision if needed
                        final_elements_list.insert(0, min_inst - 1 - idx)

                    # Add at the end (tail)
                    for idx in range(nb_appended_files):
                        final_elements_list.append(max_inst + 1 + idx)

                # Replace the original list with the padded / extended one
                dicom_files_list = final_elements_list

            for item in  dicom_files_list:
                series_path = series_path.resolve()
                if isinstance(item, Path):
                    # Real DICOM file
                    instance_num = int(item.stem)
                    is_padding = False
                else:
                    # It is a padding instance : the item is the instance number
                    instance_num = item
                    is_padding = True

                process_completed: bool = self._process_single_dicom_instance(
                    series_path,
                    series_min,
                    series_max,
                    input_features_df,
                    labels_df,
                    writer,
                    instance_num=instance_num,
                    is_padding = is_padding
                )

                if process_completed is False:
                    nb_failures += 1
                    continue

                nb_success += 1

            full_success: bool = (nb_failures == 0)
            partial_success: bool = (nb_success > 0 and not full_success)
            complete_failure: bool = (nb_success == 0)

            if complete_failure:
                logger.error(
                    f"Series {series_path.name} processing failed: "
                    f"All files failed during processing.",
                    exc_info = True,
                    extra={"status": "failed"}
                )

            if partial_success:
                logger.warning(
                    f"Series {series_path.name} partially completed:"
                    f"    - ({nb_success} were processed with success). "
                    f"    - {nb_failures} files were skipped due to processing errors).",
                    extra={"status": "partial_success",
                           "failed_processing": nb_failures}
                )

            if full_success:
                logger.info(
                    f"Series {series_path.name} processing completed successfully.",
                    extra={"status": "success"}
                )

            return nb_success, nb_failures, nb_dicom_files

        except Exception as e:
            # Log the failure of the specific series before re-raising
            series_id = series_path.name
            study_id = series_path.parent.name

            logger.error(
                f"Failed to process series instance {series_id} in study {study_id}: {str(e)}",
                exc_info=True,
                extra={"study_id": study_id, "series_id": series_id, "status": "failed"}
            )
            raise  # Keep propagating the error

    @log_method()
    def _get_series_stats(
        self,
        series_path: Path,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[int, int]:

        """
        Calculates the minimum and maximum pixel intensity values across an entire DICOM series.

        This method iterates through all DICOM files in the specified directory to determine 
        the global dynamic range of the volume. These statistics are essential for 
        consistent normalization during the TFRecord serialization process.

        Args:
            series_path (Path): Path to the directory containing the DICOM series.
            logger (Optional[logging.Logger]): Logger instance for error reporting.

        Returns:
            Tuple[int, int]: A tuple containing the (global_min, global_max) pixel values 
                represented as integers.

        Raises:
            FileNotFoundError: If no DICOM files are discovered in the provided path.
            Exception: Propagates any error encountered during file reading or 
                pixel array extraction.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger

        try: 
            if isinstance(series_path, (str, Path)):
                dicom_paths = list(Path(series_path).glob("*.dcm"))

                if not dicom_paths:
                    error_msg = (
                        f"Error in function {class_name}.{func_name}"
                        f"No DICOM files found in {series_path}"
                    )
                    raise FileNotFoundError(error_msg)

            global_min = float('inf')
            global_max = float('-inf')

            for path in dicom_paths:
                ds = pydicom.dcmread(path)
                arr = ds.pixel_array

                current_min = np.min(arr)
                current_max = np.max(arr)

                if current_min < global_min: global_min = current_min
                if current_max > global_max: global_max = current_max
            
            return int(global_min), int(global_max)

        except Exception as e:
            error_msg = (
                f"Function {class_name}.{func_name} failed: "
                f"{e}"
            )
            logger.error(error_msg, exc_info = True)
            raise

    @log_method()
    def _process_single_dicom_instance(
        self,
        series_path: Path,
        series_min: int,
        series_max: int,
        input_features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        writer: tf.io.TFRecordWriter,
        instance_num: int,
        is_padding: bool=False,
        logger: Optional[logging.Logger] = None
    ) -> bool:

        """
        Handles the end-to-end processing of an individual DICOM instance or padding frame.

        Args:
             - series_path (Path): Directory path containing the series' DICOM files.
             - series_min (int): Global minimum pixel value of the series.
             - series_max (int): Global maximum pixel value of the series.
             - input_features_df (pd.DataFrame): Metadata for the study/series.
             - labels_df (pd.DataFrame): Ground truth labels and coordinates.
             - writer (tf.io.TFRecordWriter): Active writer for TFRecord export.
             - instance_num (int): The specific instance number to process/simulate.
             - is_padding (bool): If True, generates a 1x1 dummy pixel instead of reading a file.
             - logger (Optional[logging.Logger]): Logger instance.

        Returns:
            bool: True if successful, False otherwise.
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        logger = logger or self._logger

        # Construct the specific file path only if it's a real DICOM
        current_file_path = series_path / f"{instance_num}.dcm" if not is_padding else None

        try:
            if is_padding:
                # -->> Route to a specialized method for dummy feature generation << --
                # This method should create a 1x1 pixel with appropriate metadata
                features = self._generate_padding_features(
                    series_path,
                    instance_num,
                    series_min,
                    series_max,
                    input_features_df,
                    labels_df
                )
            else:
                # Standard processing for real DICOM files
                features = self._process_dicom_file_with_metadata(
                    current_file_path,
                    series_min,
                    series_max,
                    input_features_df,
                    labels_df
                )

            # Serialize and write to TFRecord
            writer.write(features.SerializeToString())
            return True

        except Exception as e:
            # Use instance_num in the log since current_file_path might be None
            context_info = f"Instance: {instance_num} (Padding: {is_padding})"
            error_msg = (
                f"Function {class_name}.{func_name} failed: "
                f"Error in _process_single_dicom_instance. {context_info}. - {e}"
            )
            logger.error(
                error_msg,
                exc_info=True,
                extra={"status": "failed", "error_type": "DicomProcessingError", "instance": instance_num}
            )
            return False

    def _generate_padding_features(
        self,
        series_path: Path,
        instance_num: int,
        series_min: int,
        series_max: int,
        input_features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        logger: Optional[logging.Logger] = None
    ) -> tf.train.Example:
        """
        Generates a 1x1 dummy pixel feature set for padding.
    
        This mimics the output of _process_dicom_file_with_metadata but 
        without reading any file from disk.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger

        info_msg = (
            f"Function {class_name}.{func_name} started "
            f"for processing padding instance {instance_num} in series {series_path}"
        )

        logger.info(
            info_msg,
            extra={
                "action": "generate_padding_features",
                "series": {str(series_path)},
                "instance": {instance_num}
            }
        )

        try:
            # 1. Basic Identifiers
            series_id = int(series_path.stem)
            study_id = int(series_path.parent.stem)

            # 2. Extract metadata from input_features_df for this series
            # We take the first match as metadata (description, etc.) is series-consistent
            series_meta = input_features_df[input_features_df['series_id'] == series_id].iloc[0]
            description_code = int(series_meta['description'])

            # 3. Create the 1x1 dummy pixel (uint16 to match DICOM depth)
            # We use 0 as the "null" value.
            dummy_pixel = np.zeros((1, 1, 1), dtype=np.uint16).tobytes()

            # 4. Build the Feature Dictionary
            # Ensure keys match exactly those in your real DICOM processing
            features_dict = self._prepare_tf_features(
                    is_padding=True,
                    image_bytes=dummy_pixel,
                    study_id=study_id,
                    series_id=series_id,
                    series_min=series_min,
                    series_max=series_max,
                    instance_id=instance_num,
                    img_height=1,
                    img_width=1,
                    description=description_code,
                    labels_df=labels_df,
                    nb_max_records=self._MAX_RECORDS
                )
            
            features = tf.train.Example(features=tf.train.Features(feature=features_dict))

            info_msg = (
                f"Function {class_name}.{func_name}: "
                f"Successfully processed padding instance {instance_num} in series {series_path}"
            )

            logger.info(
                info_msg,
                extra={"status": "success"}
            )

            return features

        except Exception as e:
            error_msg = (
                f"Function {class_name}.{func_name} failed: "
                f"The padding instance generation for {series_path}/{instance_num}.dcm failed. "
                f"{str(e)}"
            )
            logger.error(
                error_msg,
                exc_info=True,  # Includes the full stack trace upon failure
                extra={"status": "failed", "error_type": "DicomProcessingError"}
            )
            raise

    @log_method()
    def _process_dicom_file_with_metadata(
        self,
        dicom_path: Path,
        series_min: int,
        series_max: int,
        input_features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        logger: Optional[logging.Logger] = None
    ) -> tf.train.Example:

        """
        Processes a single DICOM file to extract pixel data and associated metadata.

        This method performs the core data transformation:
        1. Reads the DICOM image using SimpleITK and converts it to a raw byte stream.
        2. Extracts and validates hierarchical IDs (Study, Series, Instance).
        3. Retrieves the specific series description from the provided metadata.
        4. Compiles all data into a structured `tf.train.Example` via `_prepare_tf_features`.

        Args:
            - dicom_path (Path): Path to the DICOM file to be processed.
            - series_min (int): Minimum pixel intensity value for the entire series.
            - series_max (int): Maximum pixel intensity value for the entire series.
            - input_features_df (pd.DataFrame): DataFrame containing series-level descriptions.
            - labels_df (pd.DataFrame): DataFrame containing ground truth labels and coordinates.
            - logger (Optional[logging.Logger]): Logger instance for status and error tracking.

        Returns:
            tf.train.Example: A serialized TensorFlow Example containing image bytes, 
                spatial dimensions, and comprehensive metadata.

        Raises:
            ValueError: If the series description cannot be found in the metadata.
            Exception: Propagates any error occurring during image reading, 
                array conversion, or feature preparation.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger

        info_msg = (
            f"Function {class_name}.{func_name} started "
            f"for processing DICOM file {dicom_path}"
        )
        logger.info(info_msg,
                    extra={"action": "process_dicom_file_with_metadata", "dicom_file": dicom_path})

        # Metadata serialization
        instance_id = int(dicom_path.stem)
        series_id = int(dicom_path.parent.name)
        study_id = int(dicom_path.parent.parent.name)

        try:
            # Retrieve DICOM image array
            img = sitk.ReadImage(str(dicom_path))
            img_array:np.ndarray = sitk.GetArrayFromImage(img)

            # SimpleITK considers each image as a 3D volume. Each DICOM file is loaded
            # as a volume with one slice only ==> format (Depth, Height, Width), where Depth = 1
            # Note: The channel dimension is absent by default if the number of channels is 1,
            # which is normally the rule with DICOM files.

            # The first dimension is useless and must be removed
            img_array = np.squeeze(img_array)

            # Handle Channels (Enforce Rank 3: [H, W, C])
            if img_array.ndim == 2:
                # Add channel dimension if missing
                img_array = img_array[:,:,np.newaxis]

            elif img_array.ndim == 3:
                # Verify that the last dimension is indeed the channel count
                # and not a leftover depth dimension
                if img_array.shape[2] != img.GetNumberOfComponentsPerPixel():
                    # This could happen if squeeze didn't work as expected
                    # or if the axes are transposed.
                    raise ValueError(f"Inconsistent channel data in {dicom_path}")

            else:
                error_msg = (
                    f"Unsupported image: rank {img_array.ndim} "
                    f"(expected 2 before expansion or 3) in {dicom_path}"
                )
                raise ValueError(error_msg)
            
            # Scale the image to the target size :
            bool_var_1 = (input_features_df['series_id'] == series_id)
            bool_var_2 = (input_features_df['study_id'] == study_id)
            mask = bool_var_1 & bool_var_2

            relevant_data = input_features_df.loc[mask, ['actual_file_format', 'description']]
            if not relevant_data.empty:
                # Note: target_format is formatted as follows: (Width, Height, Depth)
                # This format derives from the command sitk.ImageFileReader().GetSize(),
                # that was called by function CSVMetadataHandler._get_file_format().
                # This format is the reverse order of the data returned by
                # sitk.GetArrayFromImage(img), which is also used in this function.
                target_format =  relevant_data['actual_file_format'].iloc[0]
                description = int(relevant_data['description'].iloc[0])

            else:
                raise ValueError(f"No matching data in file {class_name}.{func_name}")

            img_padded = self._pad_image_to_expected_format(img_array, target_format, dicom_path)

            img_uint16 = img_padded.astype(np.uint16)
            img_bytes = img_uint16.tobytes()

            features_dict = self._prepare_tf_features(
                is_padding=False,
                image_bytes=img_bytes,
                study_id=study_id,
                series_id=series_id,
                series_min=series_min,
                series_max=series_max,
                instance_id=instance_id,
                img_height=target_format[1],
                img_width=target_format[0],
                description=description,
                labels_df=labels_df,
                nb_max_records=self._MAX_RECORDS
            )
            
            features = tf.train.Example(features=tf.train.Features(feature=features_dict))

            logger.info(f"Function {class_name}.{func_name}: Successfully processed DICOM file {dicom_path}",
                        extra={"status": "success"})

            return features

        except Exception as e:
            error_msg = (
                f"Function {class_name}.{func_name} failed: "
                f"Error in _process_dicom_file_with_metadata: The DICOM file read/conversion/serialization process for {dicom_path.name} failed. "
                f"{str(e)}"
            )
            logger.error(
                error_msg,
                exc_info=True,  # Includes the full stack trace upon failure
                extra={"status": "failed", "error_type": "DicomProcessingError"}
            )
            raise

    def _pad_image_to_expected_format(self, img_array, file_format, dicom_path):
        """
        Apply centered zero-padding to reach the target resolution.
        file_format: (Width, Height, Channels)
        """

        # 1. Extract current dimensions from the first two axes (Height, Width)
        current_height, current_width, _ = img_array.shape

        # 2. Padding calculation for Height (using index 1 of file_format)
        pad_height_before = (file_format[1] - current_height) // 2
        pad_height_after = file_format[1] - current_height - pad_height_before
        
        # 3. Padding calculation for Width (using index 0 of file_format)
        pad_width_before = (file_format[0] - current_width) // 2
        pad_width_after = file_format[0] - current_width - pad_width_before

        # 4. Define padding based on array dimensionality
        if img_array.ndim == 3:
            padding = (
                (pad_height_before, pad_height_after),      # Vertical padding
                (pad_width_before, pad_width_after),         # Horizontal padding
                (0, 0)                       # No padding for channels
            )
        else:
            raise ValueError(f"Unsupported dicom file format {file_format} in {dicom_path}")

        # 5. Apply padding with constant zeros (black)
        img_padded = np.pad(img_array, padding, mode='constant', constant_values=0)

        return img_padded

    def _prepare_tf_features(
        self,
        is_padding,
        image_bytes: bytes,
        study_id: int,
        series_id: int,
        series_min:int,
        series_max: int,
        instance_id: int,
        img_height: int,
        img_width: int,
        description: int,
        labels_df: pd.DataFrame,
        nb_max_records: int,
        logger: Optional[logging.Logger] = None
    ) -> dict:

        """
        Constructs a feature dictionary for a TensorFlow Example protobuf.

        This method handles the mapping of image data and metadata into TensorFlow-compatible 
        feature types. A critical part of this process is the normalization of the 
        labels: it ensures that the 'records' vector always contains exactly `nb_max_records` 
        levels (ordered from 0 to 24), filling missing data with zeros to maintain a 
        fixed-length input for the model.

        Args:
            - is_padding (bool): True if the image is a padding one (dummy image). False otherwise.
            - image_bytes (bytes): Raw bytes of the DICOM image pixels.
            - study_id (int): Unique identifier for the study.
            - series_id (int): Unique identifier for the series.
            - series_min (int): Minimum pixel intensity in the series.
            - series_max (int): Maximum pixel intensity in the series.
            - instance_id (int): Instance number of the DICOM file.
            - img_height (int): Height of the image in pixels.
            - img_width (int): Width of the image in pixels.
            - description (int): Categorical description ID of the series.
            - labels_df (pd.DataFrame): DataFrame containing condition levels, severity, and coordinates.
            - nb_max_records (int): Expected number of condition levels (e.g., 25).
            - logger (Optional[logging.Logger]): Logger instance for execution tracking.

        Returns:
            dict: A dictionary mapping feature names to `tf.train.Feature` objects.

        Raises:
            Exception: Propagates any error encountered during DataFrame manipulation 
                or feature conversion.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger
    
        try:

            # 1. Ensure condition_level is the index and we have all levels from 0 to 24 (nb_max_records - 1)
            full_range = range(nb_max_records)

            # 1. Set 'condition_level' as index to allow label-based reindexing
            # 2. Reindex with full_range (0-24) to ensure all levels exist and are sorted
            # 3. Fill missing levels with 0.0 to maintain a fixed-size, predictable structure
            prepared_df = (
                labels_df.set_index('condition_level')
                .reindex(full_range, fill_value=0.0)
                .reset_index()
            )

            # 4. Now the loop is simple and guaranteed to have 25 iterations in the right order
            records_flat = []
            for _, row in prepared_df.iterrows():
                records_flat.extend([
                    float(row['condition_level']),
                    float(row['severity']),
                    float(row['x']),
                    float(row['y'])
                ])

            # 5. Build the feature dictionary
            return {
                'image': _bytes_feature(image_bytes),
                'is_padding': _bool_feature(is_padding),
                'file_format': _int64_list_feature([img_width, img_height]),
                'study_id': _int64_feature(study_id),
                'series_id': _int64_feature(series_id),
                'series_min': _int64_feature(series_min),
                'series_max': _int64_feature(series_max),
                'instance_number': _int64_feature(instance_id),
                'img_height': _int64_feature(img_height),
                'img_width': _int64_feature(img_width),
                'description': _int64_feature(description),
                'records': _float_list_feature(records_flat),
                'nb_records': _int64_feature(len(labels_df))
            }

        except Exception as e:
            error_msg = (
                f"Function {class_name}.{func_name} failed: "
                f"{e}"
            )
            logger.error(error_msg, exc_info = True)
            raise

    def get_max_series_depth(self) -> int:
        """
        Retrieves the maximum allowed number of DICOM instances per series.

        This limit is used during the processing phase to ensure memory stability 
        and consistent input shapes across different medical series.

        Returns:
            int: The current maximum depth threshold.
        """
        return self._max_series_depth

    def set_series_depth(self, depth: int) -> None:
        """
        Updates the maximum allowed number of DICOM instances per series.

        Args:
            depth (int): The new depth limit to be applied for future 
                processing tasks.
        """
        self._max_series_depth = depth
