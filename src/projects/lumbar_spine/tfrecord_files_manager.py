# coding: utf-8
from typing import Tuple, Optional
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import inspect
from src.core.utils.logger import log_method
from src.projects.lumbar_spine.csv_metadata_handler import CSVMetadataHandler
from pathlib import Path
from tqdm import tqdm
from typing import List, Union


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
    Returns an int32_feature from a boolean.
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

        path_str = config['paths']['tfrecord']

        self._tfrecord_dir = Path(path_str)
        self._config = config
        self._logger = logger

        self._max_records = self._config['data_specs']['max_records_per_frame']

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

        paths_cfg = self._config['paths']
        dicom_studies_dir = paths_cfg['dicom_studies']

        try:
            # Convert DICOM files to TFRecords (if needed)
            if not list(self._tfrecord_dir.glob("*.tfrecord")):

                # 1. Prepare the directories
                self._tfrecord_dir.mkdir(parents=True, exist_ok=True)

                # 2. Load and merge metadata
                metadata_handler = CSVMetadataHandler(
                    config=self._config,
                    logger=logger,
                    dicom_studies_dir=dicom_studies_dir,
                    **paths_cfg["csv"]
                )

                metadata_df = metadata_handler.generate_metadata_dataframe()
                logger.info("Loaded metadata from CSV files",
                            extra={"csv": list(paths_cfg["csv"].keys())})

                logger.info("  Creating TFRecord files...")

                self._convert_dicom_to_tfrecords(
                    studies_dir=dicom_studies_dir,
                    metadata_df=metadata_df,
                    tfrecord_dir=str(self._tfrecord_dir)
                )
                logger.info("DICOM to TFRecord conversion completed.",
                            extra={"status": "success"})
            else:
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
            - studies_dir (str): The full path to the root directory containing study subfolders.
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
                        f"Skipping empty directory {study_full_path} "
                        f"in root directory {studies_dir}"
                    )
                    logger.warning(msg_warning, extra={"status": "Skipped studies"})
                    continue

                study_metadata_df = (
                    metadata_df[metadata_df['study_id'] == int(study_full_path.name)]
                )

                # Case missing metadata
                if study_metadata_df.empty:
                    msg_warning = (
                        f"Skipping study {study_full_path.name} due to missing metadata. "
                        "This study will not be considered during training or evaluation "
                        "and the relevant TFRecord file will not be generated."
                        "Possible cause: missing or inconsistent records in the CSV files. "
                        "Required action: Please check the CSV files "
                        "and ensure they contain the right records."
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
        Orchestrates the processing of all series within a study
        and saves them into a single TFRecord.

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
        columns = [
            'study_id', 'series_id', 'series_description', 'instance_number', 'actual_file_format'
        ]
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

                    nb_success, _, _, _ = self._process_single_series_instance(
                        series_path,
                        input_features_df,
                        labels_df,
                        writer
                    )

                    if nb_success == 0:
                        nb_skipped_series += 1
                        continue

            if nb_skipped_series > 0:
                error_msg = (
                    f"Issue in function {class_name}.{func_name}."
                    f"Study {study_id} processed with {nb_skipped_series} skipped series "
                    "due to missing metadata or aborted TFRecord file generation."
                )
                logger.error(
                    error_msg,
                    extra={
                        "status": "failure",
                        "nb_series": nb_series,
                        "skipped_series": nb_skipped_series
                    }
                )

                raise SeriesProcessingError(error_msg)

            else:  # last case : nb_series > 0 and nb_skipped_series == 0
                logger.info(
                    f"Function {class_name}.{func_name}: "
                    f"study {study_id} processing completed successfully.",
                    extra={
                        "status": "success",
                        "nb_series": nb_series,
                        "skipped_series": nb_skipped_series
                    }
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
    ) -> Tuple[int, int, int, int]:

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

            logger.warning(
                warning_message,
                extra={
                    "status": "metadata_missing",
                    "series_dir": series_path.name,
                    "study_id": study_id
                }
            )

            dicom_files_list = list(series_path.glob("*.dcm"))
            nb_files = len(dicom_files_list)
            return 0, 0, nb_files, 0

        try:
            nb_success, nb_failures, nb_dcm_files, nb_padding_instances = (
                self._process_series(series_path, input_features_df, labels_df, writer)
            )
            return nb_success, nb_failures, nb_dcm_files, nb_padding_instances

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
    ) -> Tuple[int, int, int, int]:

        """
        Orchestrates series processing by handling stats, padding, and iteration.
        """
        logger = logger or self._logger
        series_id, study_id = series_path.name, series_path.parent.name

        try:
            # 1. Stats & File Discovery
            series_min, series_max = self._get_series_stats(series_path)
            dicom_files = sorted(list(series_path.glob("*.dcm")), key=lambda x: int(x.stem))

            # 2. Get Normalized Sequence (Real files + Padding indices)
            full_sequence = self._plan_series_sequence(dicom_files)

            # 3. Execution Loop
            nb_success, nb_failures = 0, 0
            for item in full_sequence:
                if isinstance(item, Path):
                    instance_num, is_padding = (int(item.stem), False)
                else:
                    instance_num, is_padding = (item, True)

                success = self._process_single_dicom_instance(
                    series_path.resolve(), series_min, series_max,
                    input_features_df, labels_df, writer,
                    instance_num=instance_num, is_padding=is_padding
                )
                if success:
                    nb_success += 1
                else:
                    nb_failures += 1

            self._log_series_status(series_id, nb_success, nb_failures, logger)

            nb_dicom = len(dicom_files)
            nb_padding = len(full_sequence) - nb_dicom
            return nb_success, nb_failures, nb_dicom, nb_padding

        except Exception as e:
            logger.error(
                f"Failed series {series_id} (Study {study_id}): {e}",
                exc_info=True,
                extra={"study_id": study_id, "series_id": series_id, "status": "failed"}
            )
            raise

    @log_method()
    def _plan_series_sequence(
        self,
        dicom_files: List[Path],
        logger: Optional[logging.Logger] = None
    ) -> List[Union[Path, int]]:
        """
        Calculates the final sequence of instances to reach series_depth.
        Returns a list containing Paths (real files) and ints (padding instances).
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger

        try:
            depth_target = self._config["series_depth"]
            nb_files = len(dicom_files)

            if nb_files == depth_target:
                return dicom_files

            indices = [int(f.stem) for f in dicom_files]
            min_i, max_i = (min(indices), max(indices)) if indices else (0, 0)

            # Step 1: Real files + Internal Gaps
            sequence = dicom_files.copy()
            current_target = depth_target - nb_files

            for inst in range(min_i + 1, max_i):
                if current_target <= 0:
                    break
                if inst not in indices:
                    sequence.append(inst)
                    current_target -= 1

            # Step 2: External Gaps (Head & Tail)
            if current_target > 0:
                head_count = current_target // 2
                tail_count = current_target - head_count

                for i in range(head_count):
                    sequence.insert(0, min_i - 1 - i)
                for i in range(tail_count):
                    sequence.append(max_i + 1 + i)

            return sequence

        except Exception as e:
            error_msg = (
                f"Function {class_name}.{func_name} failed: "
                f"{e}"
            )
            logger.error(error_msg, exc_info=True)
            raise

    def _log_series_status(
        self,
        series_id: str,
        nb_success: int,
        nb_failures: int,
        logger: logging.Logger
    ) -> None:
        """
        Handles final logging for a series based on success/failure counts.
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        full_success = (nb_failures == 0)
        partial_success = (nb_success > 0 and not full_success)
        complete_failure = (nb_success == 0)

        if complete_failure:
            logger.error(
                f"Series {series_id} processing failed: All files failed.",
                extra={"status": "failed"}
            )

        elif partial_success:
            logger.warning(
                f"Series {series_id} partially completed: "
                f"({nb_success} success, {nb_failures} skipped).",
                extra={"status": "partial_success", "failed_count": nb_failures}
            )

        elif full_success:
            logger.info(
                f"Function {class_name}.{func_name}: "
                f"Series {series_id} processing completed successfully.",
                extra={"status": "success"}
            )

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

                if current_min < global_min:
                    global_min = current_min

                if current_max > global_max:
                    global_max = current_max

            return int(global_min), int(global_max)

        except Exception as e:
            error_msg = (
                f"Function {class_name}.{func_name} failed: "
                f"{e}"
            )
            logger.error(error_msg, exc_info=True)
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
        is_padding: bool = False,
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
                extra={
                    "status": "failed",
                    "error_type": "DicomProcessingError",
                    "instance": instance_num
                }
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
            description_code = int(series_meta['series_description'])

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
                    nb_max_records=self._max_records
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
        Orchestrates image loading, padding, and TF serialization.

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

        instance_num = dicom_path.name
        series_path = dicom_path.parent

        info_msg = (
            f"Function {class_name}.{func_name} started "
            f"for processing image instance {instance_num} in series {series_path}"
        )
        logger.info(info_msg)

        try:
            # 1. Load and normalize array [H, W, C]
            img_array, num_components = self._load_normalized_dicom(dicom_path, logger)

            # 2. Get target formatting metadata
            target_w, target_h, description = self._get_target_metadata(
                dicom_path,
                input_features_df,
                logger
            )

            # 3. Apply padding if necessary
            img_array = self._apply_center_padding(
                img_array,
                target_w,
                target_h,
                logger
            )

            # 4. Serialize to TF Example
            img_bytes = img_array.astype(np.uint16).tobytes()

            features_dict = self._prepare_tf_features(
                is_padding=False,
                image_bytes=img_bytes,
                study_id=int(dicom_path.parent.parent.name),
                series_id=int(dicom_path.parent.name),
                series_min=series_min,
                series_max=series_max,
                instance_id=int(dicom_path.stem),
                img_height=target_h,
                img_width=target_w,
                description=description,
                labels_df=labels_df,
                nb_max_records=self._max_records
            )

            info_msg = (
                f"Function {class_name}.{func_name}: "
                f"successfully processed image instance {instance_num} in series {series_path}"
            )

            logger.info(
                info_msg,
                extra={"status": "success"}
            )

            return tf.train.Example(features=tf.train.Features(feature=features_dict))

        except Exception as e:
            error_msg = f"Function {class_name}.{func_name} failed for {dicom_path.name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def _load_normalized_dicom(
        self,
        dicom_path: Path,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[np.ndarray, int]:

        """
        Reads DICOM and ensures [H, W, C] format.
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger

        try:
            img = sitk.ReadImage(str(dicom_path))
            img_array = sitk.GetArrayFromImage(img)

            # Remove depth dimension (Depth=1 for single DICOM)
            img_array = np.squeeze(img_array, axis=0)

            if img_array.ndim == 2:
                img_array = img_array[:, :, np.newaxis]

            if img_array.shape[2] != img.GetNumberOfComponentsPerPixel():
                raise ValueError(f"Channel mismatch in {dicom_path}")

            return img_array, img.GetNumberOfComponentsPerPixel()

        except Exception as e:
            error_msg = (
                f"Function {class_name}.{func_name} failed: "
                f"{e}"
            )
            logger.error(error_msg, exc_info=True)
            raise

    def _get_target_metadata(
        self,
        dicom_path: Path,
        input_features_df: pd.DataFrame,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[int, int, int]:
        """
        Retrieves the target dimensions and series description from the metadata DataFrame.

        Returns:
            Tuple[int, int, int]: (target_width, target_height, description)
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger
        try:
            series_id = int(dicom_path.parent.name)
            study_id = int(dicom_path.parent.parent.name)

            # Filter metadata for the specific series and study
            mask_1 = (input_features_df['series_id'] == series_id)
            mask_2 = (input_features_df['study_id'] == study_id)
            mask = mask_1 & mask_2

            relevant_data = input_features_df.loc[
                mask,
                ['actual_file_format', 'series_description']
            ]

            if relevant_data.empty:
                raise ValueError(f"No matching metadata found for DICOM: {dicom_path}")

            # Calculate target format (square based on max dimension)
            target_format_df = relevant_data['actual_file_format'].unique()
            all_dimensions = [dim for fmt in target_format_df for dim in fmt[:2]]
            max_val = max(all_dimensions)

            # target_format is (Width, Height)
            target_w, target_h = max_val, max_val
            description = int(relevant_data['series_description'].iloc[0])

            return target_w, target_h, description

        except Exception as e:
            error_msg = (
                f"Function {class_name}.{func_name} failed: "
                f"{e}"
            )
            logger.error(error_msg, exc_info=True)
            raise

    def _apply_center_padding(
        self,
        img_array: np.ndarray,
        target_w: int,
        target_h: int,
        logger: logging.Logger
    ) -> np.ndarray:
        """
        Calculates and applies zero-padding to center the image.
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger

        try:
            current_h, current_w = img_array.shape[0], img_array.shape[1]

            if (current_h, current_w) == (target_h, target_w):
                return img_array

            pad_h = max(0, target_h - current_h)
            pad_w = max(0, target_w - current_w)

            if pad_h == 0 and pad_w == 0:
                return img_array

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            return np.pad(
                img_array,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0
            )

        except Exception as e:
            error_msg = (
                f"Function {class_name}.{func_name} failed: "
                f"{e}"
            )
            logger.error(error_msg, exc_info=True)
            raise

    def _prepare_tf_features(
        self,
        is_padding,
        image_bytes: bytes,
        study_id: int,
        series_id: int,
        series_min: int,
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
            - labels_df (pd.DataFrame): DataFrame containing condition levels, severity,
                                        and coordinates.
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

            # 1. Ensure condition_level is the index and we have all levels
            # from 0 to 24 (nb_max_records - 1)
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
                'series_description': _int64_feature(description),
                'records': _float_list_feature(records_flat),
                'nb_records': _int64_feature(len(labels_df))
            }

        except Exception as e:
            error_msg = (
                f"Function {class_name}.{func_name} failed: "
                f"{e}"
            )
            logger.error(error_msg, exc_info=True)
            raise

    def get_series_depth(self) -> int:
        """
        Retrieves the maximum allowed number of DICOM instances per series.

        This limit is used during the processing phase to ensure memory stability
        and consistent input shapes across different medical series.

        Returns:
            int: The current maximum depth threshold.
        """
        return self._config['series_depth']
