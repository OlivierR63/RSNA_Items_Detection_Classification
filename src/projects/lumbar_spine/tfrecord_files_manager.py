# coding: utf-8

from ast import Try
from typing import Dict, Tuple, List, Optional
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
from src.core.utils.logger import log_method
from src.projects.lumbar_spine.csv_metadata import CSVMetadata
from pathlib import Path
from tqdm import tqdm
import pydicom
import SimpleITK as sitk
from src.projects.lumbar_spine.serialization_manager import SerializationManager
import json
import inspect


class TFRecordFilesManager:
    def __init__(self, config, logger):
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
            error_msg = f"{class_name} initialization failed: the 'max_record' key is missing or wrongly defined in the configuration file"
            raise ValueError(error_msg)

        self._serialization_manager = SerializationManager(self._config, logger)

        # This attribute will be set later, when the instance method self.generate_tfrecord_files is called
        self._max_series_depth = None

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

                # 4. Create a new feature in metadata_df, as a synthesis of the features 'condition' and 'series_description'
                nb_levels = encoded_metadata_df['level'].nunique()
                encoded_metadata_df['condition_level'] = encoded_metadata_df['condition']*nb_levels + encoded_metadata_df['level'] 

                logger.info("  Creating TFRecord files...")

                self._convert_dicom_to_tfrecords(
                    study_dir=self._config["dicom_study_dir"],
                    metadata_df=encoded_metadata_df,
                    tfrecord_dir=str(self._tfrecord_dir)
                )
                logger.info("DICOM to TFRecord conversion completed.",
                            extra={"status": "success"})

                self._max_series_depth = self._calculate_max_series_depth()

            else:
                if self._max_series_depth is None:
                    self._max_series_depth = self._calculate_max_series_depth()

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

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        logger = logger or self._logger
        logger.info(f"Starting {class_name}.{func_name}",
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
                                    f"Skipping study {study_path.name} due to missing metadata. "
                                    "This study will not be considered during training or evaluation "
                                    "and the relevant TFRecord file will not be generated."
                                    "Possible cause: missing or inconsistent records in the CSV files. "
                                    "Required action: Please check the CSV files and ensure they contain the right records."
                                   )
                    logger.warning(msg_warning)
                    continue

                self._process_study(study_path, study_metadata_df, tfrecord_path)

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
        

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__
        
        logger = logger or self._logger
        logger.info(f"Starting {class_name}.{func_name}",
                    extra={"action": "process_study", "study_dir": study_path})

        study_id = study_path.name

        tfrecord_path = tfrecord_dir / f"{study_id}.tfrecord"

        nb_skipped_series = 0
        header_df = metadata_df[['study_id', 'series_id', 'series_description']].drop_duplicates()
        pathologies_df = metadata_df[['condition_level', 'severity', 'x', 'y']].drop_duplicates()

        with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
            for series_path in study_path.iterdir():
                if not self._process_single_series_instance(series_path, header_df, pathologies_df, writer):
                    nb_skipped_series += 1
                    continue

        if nb_skipped_series > 0:
            warning_msg = (f"Issue in function {class_name}.{func_name}."
                f"Study {study_id} processed with {nb_skipped_series} skipped series"
                "due to missing metadata or aborted TFRecord file generation.")
            logger.warning(
                warning_msg,
                extra={"status": "partial_success", "skipped_series": nb_skipped_series}
            )
        else:
            logger.info(
                f"Function {class_name}.{func_name}: study {study_id} processing completed successfully.",
                extra={"status": "success"}
            )

    @log_method()
    def _process_single_series_instance(
        self,
        series_path: Path,
        header_df: pd.DataFrame,
        pathologies_df: pd.DataFrame,
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
                - header_df: 
                - pathologies_df:
                - writer: The active TFRecordWriter instance for the current study.
                - logger: Logger instance for logging warnings.


            Returns:
                bool: True if the series was processed (meaning it was a valid directory with
                      metadata, and `_process_series` was called). False if the series was
                      skipped (non-directory item, missing metadata, or complete processing failure
                      in `_process_series`).
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
            return False

        series_id = int(series_path.name)
        series_metadata_df = header_df[header_df['series_id'] == series_id]

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
            return False

        is_successful = not self._process_series(series_path, header_df, pathologies_df, writer)

        return is_successful

    @log_method()
    def _process_series(
        self,
        series_path: Path,
        header_df: pd.DataFrame,
        pathologies_df: pd.DataFrame,
        writer: tf.io.TFRecordWriter,
        logger: Optional[logging.Logger] = None
    ) -> bool:
        """
            Processes all DICOM files within a single series directory.

            The process delegates both file reading and  metadata
            serialization to `_process_dicom_file`, and feature writing to
            `_write_tfrecord_example`.

            Args:
                - series_path: The path to the series directory.
                - header_df:
                - pathologies_df:
                - writer: The active TFRecordWriter instance for the current study.

            Returns:
                bool: True if the processing of the series resulted in a *complete failure*
                        (i.e., zero successful TFRecord examples were written).
                      False otherwise (meaning full or partial success).
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

        nb_failed_process = 0
        nb_success_file = 0

        # Define the min / max pixel values of the series
        series_min, series_max = self._get_series_stats(series_path)

        for dicom_path in series_path.glob("*.dcm"):

            process_aborted: bool = self._process_single_dicom_instance(
                dicom_path,
                series_min,
                series_max,
                header_df,
                pathologies_df,
                writer
            )

            if process_aborted is True:
                nb_failed_process += 1
                continue

            nb_success_file += 1

        full_success: bool = (nb_failed_process == 0)
        partial_success: bool = (nb_success_file > 0 and not full_success)
        complete_failure: bool = (nb_success_file == 0)

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
                f"    - ({nb_success_file} were processed with success). "
                f"    - {nb_failed_process} files were skipped due to processing errors).",
                extra={"status": "partial_success",
                       "failed_processing": nb_failed_process}
            )

        if full_success:
            logger.info(
                f"Series {series_path.name} processing completed successfully.",
                extra={"status": "success"}
            )

        return complete_failure

    @log_method()
    def _get_series_stats(
        self,
        series_path: Path,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[int, int]:

        """
            Calculate the min et max pixel values on the whole volume of dicom files in a series.
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
                    raise FileNotFoundError()

            global_min = float('inf')
            global_max = float(-'inf')

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
        dicom_path: Path,
        series_min: int,
        series_max: int,
        header_df: pd.DataFrame,
        pathologies_df: pd.DataFrame,
        writer: tf.io.TFRecordWriter,
        logger: Optional[logging.Logger] = None
    ) -> bool:
        """
            Processes a single DICOM file within a series.

            This method delegates the processing of the DICOM file and metadata serialization to
            `_process_dicom_file`, and writes the resulting features to the TFRecord
            using `_write_tfrecord_example`.

            Args:
                - dicom_path: The path to the DICOM file.
                - series_min: The lowest pixel value in the series
                - series_max: The highest pixel value in the series
                - header_df: DataFrame containing header metadata
                - pathologies_df: DataFrame containing ground truth labels and pathologies
                - writer: The active TFRecordWriter instance for the current study
                - logger: Optional logger instance

            Returns:
            - `process_aborted` (bool): True if the processing
               (`_process_dicom_file` or `_write_tfrecord_example`) failed
               due to an exception. False otherwise (success or skipped due to metadata).
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger

        try:
            # Process the DICOM file and its metadata
            img_bytes, serialized_metadata = self._process_dicom_file_with_metadata(dicom_path, series_min, series_max, header_df, pathologies_df)

            # Write the result to the TFRecord
            logger.info(f"writer = {writer}")
            self._write_tfrecord_example(img_bytes, serialized_metadata, writer)

            process_aborted = False
            return process_aborted

        except Exception:
            error_msg = (
                f"Function {class_name}.{func_name} failed: "
                f"Error in _process_single_dicom_instance. Process failed. File: {dicom_path}"
            )
            logger.error(
                error_msg,
                exc_info=True,
                extra={"status": "failed", "error_type": "DicomProcessingError"}
            )
            process_aborted = True
            return process_aborted

    @log_method()
    def _process_dicom_file_with_metadata(
        self,
        dicom_path: Path,
        series_min: int,
        series_max: int,
        header_df: pd.DataFrame,
        pathologies_df: pd.DataFrame,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[bytes, int, int, bytes]:
        """
            Process a single DICOM file and return serialized image and metadata.

            Args:
                - dicom_path: The path to the considered DICOM file.
                - series_min: The lowest pixel value in the series
                - series_max: The highest pixel value in the series
                - header_df: DataFrame containing header metadata
                - pathologies_df: DataFrame containing ground truth labels and pathologies
                - logger: Optional logger instance.

            Returns:
                Tuple[bytes, bytes]: Serialized image tensor and serialized metadata.

            Raises:
                Exception: If reading, converting, or serializing the DICOM image fails.
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger

        info_msg = (
            f"Function {class_name}.{func_name} started "
            f"for processing DICOM file {dicom_path}"
        )
        logger.info(info_msg,
                    extra={"action": "process_dicom_file", "dicom_file": dicom_path})

        try:
            # image processing
            img = sitk.ReadImage(str(dicom_path))
            img_array:np.ndarray = sitk.GetArrayFromImage(img)
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint16)
            img_bytes = tf.io.serialize_tensor(img_tensor).numpy()

            height, width = img_array.shape[-2:]

            # Metadata serialization
            instance_id = int(dicom_path.stem)
            series_id = int(dicom_path.parent.name)
            study_id = int(header_df['study_id'].unique())

            matches = header_df.loc[header_df['series_id'] == series_id, 'series_description'].unique()

            if len(matches) > 0:
                description = int(matches[0]) # It is essential to convert an np.int64 variable into an int, for serialization purpose
            else:
                err_msg = (
                    f"Function {class_name}.{func_name} failed: "
                    f"no description found related with study {study_id} and series {series_id}"
                )
                raise ValueError(err_msg)

            serialized_metadata = SerializationManager.serialize_metadata(
                study_id,
                series_id,
                series_min,
                series_max,
                instance_id,
                height,
                width,
                description,
                pathologies_df
            )

            logger.info(f"Function _process_dicom_file_with_metadata: Successfully processed DICOM file {dicom_path}",
                        extra={"status": "success"})

            return img_bytes, serialized_metadata

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

    @log_method()
    def _write_tfrecord_example(
        self,
        img_bytes: bytes,
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
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger
        logger.info(f"Starting function {class_name}.{func_name}")

        try:
            feature = {
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                "metadata": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[serialized_metadata])
                )
            }
        
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        except Exception as e:
            error_msg = (
                f"Function {class_name}.{func_name} failed. "
                f"Error writing TFRecord example: {str(e)}"
            )
            logger.error(
                error_msg,
                exc_info=True,
                extra={"status": "failed", "error": str(e)}
            )
            raise

    @log_method()
    def _encode_dataframe(
        self,
        metadata_df: pd.DataFrame,
        *,
        logger: Optional[logging.Logger] = None
    ) -> pd.DataFrame:

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
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger
        logger.info(f"Starting function {class_name}.{func_name}")

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
            error_msg = (
                f"Error in function {class_name}.{func_name}: {str(e)}"
            )

            logger.error(
                error_msg,
                exc_info=True,
                extra={"status": "failed", "error": str(e)}
            )
            raise

    def _create_mappings(
        self,
        metadata_df: pd.DataFrame,
        columns_to_encode: Dict[str, str],
        logger: Optional[logging.Logger] = None
    ) -> Dict[str, Dict]:

        """
            Creates mapping dictionaries for each categorical column.

            Args:
                metadata_df: The DataFrame containing metadata to be converted.
                columns_to_encode: Dictionary mapping column names to their mapping variable names.

            Returns:
                Dict[str, Dict]: A dictionary of mapping dictionaries for each column.
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger

        mappings = {}

        try: 
            for column in columns_to_encode:
                if column not in metadata_df.columns:
                    self._logger.warning(f"Column '{column}' not found in DataFrame. Skipping.")
                    continue

                # Sort values to ensure mapping [0, 1, 2...] is always the same
                values = sorted(metadata_df[column].dropna().unique().tolist())
                
                # We assume self._create_string_to_int_mapper returns an object with a .mapping attribute
                mapper = self._create_string_to_int_mapper(values)
                mappings[column] = mapper.mapping

                self._logger.info(f"Created mapping for '{column}': {len(values)} categories found.")

            return mappings

        except Exception as e:
            error_msg = f"Error in {self.__class__.__name__}.{func_name} while processing column '{column}': str({e})"
            self._logger.error(
                error_msg,
                exc_info=True
            )
            raise


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
    def _create_string_to_int_mapper(
        self,
        observed_pathologies_lst: list,
        *,
        logger: Optional[logging.Logger] = None
    ) -> callable:
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

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger
        logger.info(f"Starting function {class_name}.{func_name}")

        try:
            # Create the primary dictionary {string: integer} using enumeration.
            # The integer (i) corresponds to the string's index + 1 in the list.
            mapping = {key: idx for idx, key in enumerate(observed_pathologies_lst)}

            # Create the optional reverse dictionary {integer: string}
            # for inspection/deserialization.
            reverse_mapping = {idx: key for idx, key in enumerate(observed_pathologies_lst)}

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
            logger.info(f"Function {class_name}.{func_name} completed successfully",
                        extra={"status": "success"})
            return mapper

        except Exception as e:
            error_msg = f"Error in function {class_name}.{func_name} : {str(e)}"
            logger.error(
                error_msg,
                exc_info=True,
                extra={"status": "failed", "error": str(e)}
            )
            raise

    def _process_study_multi_series(
        self, 
        images: tf.Tensor, 
        meta: Dict[str, tf.Tensor], 
        labels: Dict[str, tf.Tensor],
        logger: Optional[logging.Logger] = None
    ) -> Tuple[Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
                     Tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
                     Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
                     ], 
               tf.Tensor, 
               Dict[str, tf.Tensor]
            ]:
        """
        Orchestrates the conversion of a raw study-level frame collection into a 
        structured multi-modal input for the neural network.
    
        This method ensures the model receives exactly three imaging series 
        (Sagittal T1, Sagittal T2, and Axial T2) by:
        1. Filtering frames by anatomical description.
        2. Resolving duplicates by selecting the most complete series for each plane.
        3. Standardizing the depth of each series through symmetric padding.
        4. Reducing redundant study-level metadata and labels to single vectors.

        Args:
            images (tf.Tensor): Flattened batch of all images in the study (N, H, W, C).
            meta (dict): Dictionary of metadata tensors (each of length N).
            labels (dict): Dictionary of label tensors (study-wide diagnostics).

        Returns:
            tuple: (study_data_triplet, study_id, reduced_labels)
                   - study_data_triplet: ((img, sid, desc)_t1, (img, sid, desc)_t2, (img, sid, desc)_ax)
                   - study_id: The common identifier for the study.
                   - reduced_labels: Compacted labels for the entire study.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger

        try:
            # --- 1. Define Search Targets (Integer Codes) ---
            # Based on the internal mapping: 0: "Sagittal T1", 1: "Sagittal T2", 2: "Axial T2"
            # Using constants ensures Graph-compatibility during execution.
            t1_code = tf.constant(0, dtype=tf.int32)
            t2_code = tf.constant(1, dtype=tf.int32)
            ax_code = tf.constant(2, dtype=tf.int32)

            # --- 2. Process Individual Branches ---
            # We extract the 3-tuple (Padded_Images, Selected_Series_ID, Description_Code) for each plane.
            # This handles series selection and black-frame padding internally.
            res_t1 = self._process_single_description(images, meta, t1_code)
            res_t2 = self._process_single_description(images, meta, t2_code)
            res_ax = self._process_single_description(images, meta, ax_code)

            # --- 3. Extract Global Study Context ---
            # Since all frames in the window belong to the same study (guaranteed by _get_group_key),
            # we take the first element as the representative study_id.
            study_id = meta['study_id'][0]

            # --- 4. Label Consolidation ---
            # Diagnostic labels (records) are identical for all frames within a study.
            # We reduce them to a single scalar/vector to avoid redundant dimensions (64, -> 1,).
            reduced_labels = {
                k: v[0] if v.shape.rank is not None and v.shape.rank > 0 else v 
                for k, v in labels.items()
            }

            # Return the structured data required by LumbarDicomTFRecordDataset._format_for_model :
            return (res_t1, res_t2, res_ax), study_id, reduced_labels

        except Exception as e:
            error_msg = f"Error in function {class_name}.{func_name} : {str(e)}"
            logger.error(
                error_msg,
                exc_info=True,
                extra={"status": "failed", "error": str(e)}
            )
            raise

    def _process_single_description(
        self, 
        all_images: tf.Tensor, 
        all_meta: Dict[str, tf.Tensor], 
        target_desc_tensor: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Filters, selects the best series, and pads images for a specific anatomical view.

        The process follows these steps:
        1. Mask all images matching the target description code.
        2. If multiple series match, select the one with the most frames.
        3. Sort frames by instance number to ensure correct anatomical order.
        4. Apply symmetric padding or cropping to reach the fixed target depth.

        Args:
            all_images (tf.Tensor): Flattened batch of images (N, H, W, C).
            all_meta (Dict[str, tf.Tensor]): Metadata containing 'description', 
                                             'series_id', and 'instance_number'.
            target_desc_tensor (tf.Tensor): The pathology/view code to filter for.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: 
                - Padded Volume: (max_series_depth, H, W, C)
                - Best Series ID: The ID of the selected series (int64).
                - Description Code: The original target code (int32).
        """
        img_cfg = self._config['model_2d']['img_shape']

        max_depth_tf = tf.constant(self._max_series_depth, dtype=tf.int32)
    
        # 1. Create the filtering mask
        desc_mask = tf.equal(all_meta['description'], target_desc_tensor)
        desc_mask = tf.cast(desc_mask, tf.bool)
    
        # Set shapes for TensorFlow Graph compatibility
        desc_mask.set_shape([None])
        all_images.set_shape([None, None, None, None])

        # Check if any image matches the description
        mask_has_data = tf.reduce_any(desc_mask)

        def process_valid_series():
            # --- 2. Filtering by Description ---
            d_images = tf.boolean_mask(all_images, desc_mask)
            d_series_ids = tf.boolean_mask(all_meta['series_id'], desc_mask)
            d_instances = tf.boolean_mask(all_meta['instance_number'], desc_mask)

            # --- 3. Best Series Selection (Highest Frame Count) ---
            unique_ids, _, counts = tf.unique_with_counts(d_series_ids)
            best_id = unique_ids[tf.argmax(counts)]

            # Isolate the chosen series
            series_mask = tf.equal(d_series_ids, best_id)
            series_mask = tf.cast(series_mask, tf.bool)
            series_mask.set_shape([None])
            d_images.set_shape([None, None, None, None])

            final_imgs = tf.boolean_mask(d_images, series_mask)
            final_insts = tf.boolean_mask(d_instances, series_mask)

            # --- 4. Spatial Sorting ---
            sort_idx = tf.argsort(final_insts)
            sorted_imgs = tf.gather(final_imgs, sort_idx)

            # --- 5. Volumetric Normalization (Padding/Cropping) ---
            # Ensure we don't exceed max_series_depth before padding
            sorted_imgs = sorted_imgs[:max_depth_tf, ...]
        
            num_f = tf.shape(sorted_imgs)[0]
            pad_needed = max_depth_tf - num_f

            pad_before = pad_needed // 2
            pad_after = pad_needed - pad_before

            padded_vol = tf.pad(
                sorted_imgs,
                [[pad_before, pad_after], [0, 0], [0, 0], [0, 0]]
            )
        
            # Final shape enforcement for the model*
            # Convert single-channel MRI to 3-channel (pseudo-RGB) to match model input requirements.
            if img_cfg[2] == 3:
                padded_vol = tf.image.grayscale_to_rgb(padded_vol)

            padded_vol.set_shape([self._max_series_depth, img_cfg[0], img_cfg[1], img_cfg[2]])
        
            return padded_vol, tf.cast(best_id, tf.int64), target_desc_tensor

        def process_empty_series():
            # Handle missing series by returning a zeroed tensor and a sentinel value (-1) for series_id.
            default_id = tf.constant(-1, dtype=tf.int64)

            empty_vol = tf.zeros(
                [max_depth_tf, img_cfg[0], img_cfg[1], img_cfg[2]], 
                dtype=all_images.dtype
            )
            return empty_vol, default_id, target_desc_tensor

        # Execute conditional logic
        return tf.cond(mask_has_data, process_valid_series, process_empty_series)

    def _py_deserialize_and_flatten(
        self,
        metadata_bytes_tensor: tf.Tensor,
        logger: Optional[logging.Logger] = None
    ) -> List[tf.Tensor]:
        """
        Deserializes and flattens metadata bytes into a list of TensorFlow tensors.

        This method takes a serialized metadata byte string (typically from a TFRecord)
        and converts it into a structured list of TensorFlow tensors. The metadata is first
        deserialized into a dictionary, then split into header tensors (scalar metadata fields)
        and a flattened tensor for records (annotations or points of interest).

        The records are padded to a fixed size (self._MAX_RECORDS * 4) to ensure
        consistent tensor shapes for batch processing in TensorFlow pipelines.

        Args:
            metadata_bytes_tensor (tf.Tensor): A TensorFlow tensor containing the serialized
                metadata as a byte string. This tensor is typically extracted from a TFRecord
                and represents the 'metadata' field.

        Returns:
            list[tf.Tensor]: A list of 10 TensorFlow tensors:
                - The first 10 tensors are scalar `tf.int32` tensors representing:
                    0: study_id (int)
                    1: series_id (int)
                    2: series_min_pixel_value (int): Min pixel value into the series
                    3: series_max_pixel_value (int): Max pixel value into the series
                    4: instance_number (int)
                    5: img_height (int): height of the image (in pixels)
                    6: img_width (int): width of the image (in pixels)
                    7: description (int, encoded category)
                    8: nb_records (int, number of records/annotations)
                - The 10th tensor is a 1D `tf.float32` tensor of shape (100,) representing:
                    9: Flattened and padded records (condition_level, severity, x, y) for up to
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
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger

        try:
            # We MUST call .numpy() here, inside the py_function boundary.
            metadata_np = metadata_bytes_tensor.numpy()
            deserialized_dict=SerializationManager.deserialize_metadata(metadata_np)

            required_keys = ['study_id', 'series_id',
                             'instance_number', 'description', 'nb_records']

            for key in required_keys:
                if key not in deserialized_dict:
                    error_msg = (f"function {class_name}.{func_name} failed."
                                 f"Missing required key in deserialized metadata: {key}")
                    logger.critical(error_msg)
                    raise ValueError(error_msg)

            # 1. Extract header Tensors
            header_list = [
                int(deserialized_dict.get('study_id', 0)),
                int(deserialized_dict.get('series_id', 0)),
                int(deserialized_dict.get('series_min_pixel_value', 0)),
                int(deserialized_dict.get('series_max_pixel_value', 0)),
                int(deserialized_dict.get('instance_number', 0)),
                int(deserialized_dict.get('img_height', 0)),
                int(deserialized_dict.get('img_width', 0)),
                int(deserialized_dict.get('description', 0)),
                int(deserialized_dict.get('nb_records', 0))
            ]

            # 2. Flatten records into a 1D list of values:
            # [level1, severity1, x1, y1, level2, severity2, x2, y2, ...]
            # If no records, create a dummy 1x4 list of zeros.
            records_list = deserialized_dict.get('records', [])

            # Ensure that the list is sorted by condition_level value
            records_list.sort(key=lambda x:x[0])

            # Fill the gaps between 2 non consecutive records.
            flattened_records_list = []
            lookup_dict = {rec[0]: rec for rec in records_list}

            for idx in range(self._MAX_RECORDS):
                if idx in lookup_dict:
                    flattened_records_list.extend([float(v) for v in lookup_dict[idx]])
                else:
                    # Insert a "filling" tuple
                    flattened_records_list.extend([float(idx), 0.0, 0.0, 0.0])

            # Pad the flattened list to the right size
            # ( = self._MAX_RECORDS * 4 elements)
            # Pad with a safe value (0.0)
            current_length = len(flattened_records_list)
            max_records_flat = self._MAX_RECORDS * 4

            if current_length > max_records_flat :
                error_msg = (f"Function {class_name}.{func_name} failed. "
                             f"CORRUPTED data : Too many records ({current_length/4}"
                             f"for study {deserialized_dict.get('study_id')}. "
                             f"Expected max: {self._MAX_RECORDS}.")
                logger.critical(error_msg)
                raise ValueError(error_msg)

            # Appy padding if the current length is below the required
            # flat size
            nb_padded_records = max_records_flat - len(flattened_records_list)
            padded_records = (flattened_records_list + [0.0] * nb_padded_records)

            # Final shape vaidation before TensorFlow injection
            # This is fail-safe to prevent runtime shape mismatches in the GPU
            if len(padded_records) != max_records_flat:
                error_msg = (f"Function {class_name}.{func_name} failed. "
                             f"STRUCTURE ERROR: Final length {len(padded_records)} "
                             f"is inconsistent with the model (expected {max_records_flat}).")
                logger.critical(error_msg)
                raise RuntimeError(error_msg)

            # Convert to NumPy array. 
            # tf.py_function will automatically cast this to tf.float32 
            # based on the Tout signature
            records_np = np.array(padded_records, dtype=np.float32)
            
            # Combine header values (Python int) and record array
            return header_list + [records_np]

        except Exception as e:
            error_msg = f"Function {class_name}.{func_name} failed: {str(e)}"
            logger.error(
                error_msg,
                exc_info=True
            )
            
            # Returns a default structure
            return [0]*9 + [np.zeros(self._MAX_RECORDS*4, dtype=np.float32)]


    @log_method()
    def _parse_tfrecord_single_element(
        self,
        serialized_record_tf: tf.Tensor,
        *,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    
        """
        Parses and pre-processes a single DICOM frame from a TFRecord.

        Operations performed:
        1. Metadata Extraction: Deserializes study/series info and pathology records 
           using a python-wrapped helper (py_function).
        2. Coordinate Normalization: Scales (x, y) coordinates from absolute pixel 
           values to relative [0, 1] range based on original image dimensions.
        3. Image Reshaping: Resizes raw DICOM frames to model-specific dimensions 
           (e.g., 256x256) and adds channel dimension.
        4. Intensity Normalization: Applies Min-Max scaling using series-level 
           extrema and maps values to a configured target range (e.g., [-1, 1]).

        Args:
            serialized_record_tf: Scalar string Tensor (tf.train.Example).
            logger: Optional logging instance.

        Returns:
            Tuple containing:
            - normalized_image_tf (tf.float32): Processed 3D-ready image frame.
            - metadata (dict): Core identifiers (study_id, series_id, etc.).
            - labels (dict): Dictionary with 'records' containing normalized 
              pathology data [condition, severity, x_norm, y_norm].
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger
    
        new_height, new_width, channels = self._config['model_2d']['img_shape']

        try:
            # Define the structure of the features stored in the TFRecord.
            feature_description = {
                "image": tf.io.FixedLenFeature([], tf.string),
                "metadata": tf.io.FixedLenFeature([], tf.string)
            }

            # Parse the scalar protocol buffer string into a dictionary of Tensors.
            parsed_features_tf: Dict[str, tf.Tensor] = tf.io.parse_single_example(serialized_record_tf, feature_description)

            # --- 1. Deserialize the Metadata (using tf.py_function) ---
            metadata_bytes_tf: tf.Tensor = parsed_features_tf["metadata"]

            # Define the output types for the metadata deserializer:
            # [0: study_id (int), 1: series_id (int), 2: instance_number (int),
            # 3: description (int), 4: nb_records (int),
            # 5: records (list of tuples -> will be tf.float32/tf.int32)]

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
            # 9 scalar headers (tf.int32) + 1 padded records tensor (tf.float32)
            output_types = [tf.int32] * 9 + [tf.float32]

            # Call the py_function

            deserialized_tensors = tf.py_function(
                self._py_deserialize_and_flatten,
                inp=[metadata_bytes_tf],
                Tout=output_types
            )

            # Assign the results back to named Tensors
            # As a reminder: condition = observed pathology.
            (study_id_t, series_id_t, series_min_t, series_max_t,
             instance_number_t, img_height_t, img_width_t,
             description_t, nb_records_t, padded_records_t) = deserialized_tensors

            # Secure the shapes (essential)
            study_id_t.set_shape([]) # Scalar
            series_id_t.set_shape([])
            series_min_t.set_shape([])
            series_max_t.set_shape([])
            instance_number_t.set_shape([])
            img_height_t.set_shape([])
            img_width_t.set_shape([])
            description_t.set_shape([])
            nb_records_t.set_shape([])
            padded_records_t.set_shape([self._MAX_RECORDS * 4])

            # --- 2. Deserialize and Reshape the Image Tensor (Pure TF) ---
            image_tf: tf.Tensor = tf.io.parse_tensor(parsed_features_tf["image"], out_type=tf.uint16)
            # NOTE: Using tf.uint16, the original Dicom type normalization
            # to float32 happens later.

            # Crucial stage: the image shape is unknown so far.
            # It must be set before applying the "resize" command.
            # A new dimension is added for the channel (grayscale = 1)
            height_t = tf.cast(img_height_t, tf.int32)
            width_t = tf.cast(img_width_t, tf.int32)
            image_tf = tf.reshape(image_tf, [height_t, width_t, 1])

            # Reshape the 1D Tensor back into its original 4D shape
            # (e.g., [Depth, Height, Width, Channels]).
            # The shape is assumed to be fixed for this dataset.
            image_tf = tf.image.resize(image_tf, [new_height, new_width])

            # Reshape the padded records tensor to (self._MAX_RECORDS, 4)
            padded_records_t = tf.reshape(padded_records_t, [self._MAX_RECORDS, 4])

            # Extract categorical metadata : (conditon_level, severity)
            # We keep columns 0 and 1
            categorical_data = padded_records_t[:, :2]

            # Extract and normalize x and y coordinates:
            # index 2 is x, index 3 is y.
            coords_raw = padded_records_t[:, 2:]

            # Denominator creation from the original dimensions, which were extracted from 
            # the TFRecord. 
            # Note: stack in [width, height] order to match [x, y] coordinates.
            img_dims = tf.cast(tf.stack([img_width_t, img_height_t]), tf.float32)

            # Element-wise division: x/width and y/height
            normalized_coords = coords_raw / img_dims

            # Final tensor reconstruction (condition, severity, x_norm, y_norm)
            final_records_t = tf.concat([categorical_data, normalized_coords], axis=1)

            # --- 3. Normalize the Image (Pure TF) ---
            # The image is currently uint16. Convert to float32 and normalize.
            image_tf = tf.cast(image_tf, dtype=tf.float32)

            normalized_image_tf = self._normalize_image(image_tf, series_min_t, series_max_t)

            # --- 4. Precision Reduction for Memory Optimization ---
            # Cast to float16 immediately after normalization to reduce memory footprint by 50%.
            # This is critical for keeping the windowing buffer (group_by_window) within 
            # the available RAM limits (especially on machines with high memory pressure).
            normalized_image_tf = tf.cast(normalized_image_tf, dtype=tf.float16)

            metadata = {
                "study_id": study_id_t,
                "series_id": series_id_t,
                "instance_number": instance_number_t,
                "description": description_t
            }

            labels = {
                # The records are now a (self._MAX_RECORDS, 4) float32 tensor
                "records": final_records_t
            }

            # --- 4. Return Processed Data ---
            return_object = (normalized_image_tf, metadata, labels)

            return return_object

        except Exception as e:
            error_msg = (
                f"Function {class_name}.{func_name} failed: "
                f"Error parsing TFRecord: {str(e)}"
            )
            logger.error(
                error_msg,
                exc_info=True,
                extra={"status": "failed", "error": str(e)}
            )

            # Return a safe dummy structure to avoid pipeline crash
            dummy_image = tf.zeros([new_height, new_width, channels], dtype=tf.float32)
            dummy_records = tf.zeros([self._MAX_RECORDS, 4], dtype=tf.float32)

            dummy_metadata = {
                "study_id": tf.constant(0, dtype=tf.int32),
                "series_id": tf.constant(0, dtype=tf.int32),
                "instance_number": tf.constant(0, dtype=tf.int32),
                "description": tf.constant(0, dtype=tf.int32)
            }

            dummy_label = {
                # The records are now a (self._MAX_RECORDS, 4) float32 tensor
                "records": dummy_records
            }

            return (dummy_image, dummy_metadata, dummy_label)

    @log_method()
    def _normalize_image(
        self,
        image_tf: tf.Tensor,
        series_min_t: tf.Tensor,
        series_max_t: tf.Tensor,
        logger=None
    ) -> tf.Tensor:
        """
        Normalizes the image intensity based on series-level extrema and config scaling.
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger

        try:
            model_2d_config = self._config.get('model_2d', None)

            if not model_2d_config:
                error_msg = "Function normalize_error failed: key data 'model_2d' is missing in the configuration file"
                logger.error(error_msg)
                raise ValueError(error_msg)

            min_scaling_value = model_2d_config['min_scaling_value']
            max_scaling_value = model_2d_config['max_scaling_value']

            # Cast series-level stats to float32 for tensor arithmetic.
            s_min = tf.cast(series_min_t, tf.float32)
            s_max = tf.cast(series_max_t, tf.float32)

            # Apply Min-Max normalization using series extrema.
            # Add a small epsilon to denominator to avoid division by zero.
            denom = tf.maximum(s_max - s_min, 1e-8)
            image_tf = (image_tf - s_min) / denom

            # Rescale from [0, 1] to the target range defined in config (e.g., [-1, 1]).
            image_tf = image_tf * (max_scaling_value - min_scaling_value) + min_scaling_value

            return image_tf

        except Exception as e:
            error_msg = f"Critical error in function {class_name}.{func_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    @log_method()
    def _calculate_max_series_depth(self, logger=None) -> int:
        """
        Calculates the maximum number of slices per series using high-performance 
        TF dataset reduction and a localized metadata cache.
    
        Args:
            logger:
        
        Returns:
            int: The maximum number of records (depth) found across all series.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger

        cache_file = self._tfrecord_dir / "depth_metadata_cache.json"
        tfrecord_files = list(self._tfrecord_dir.glob("*.tfrecord"))
        file_count = len(tfrecord_files)
    
        if not tfrecord_files:

            return 0

        # 1. Smart Cache Management
        if cache_file.exists():
            try:
                with cache_file.open('r') as f:
                    cache_data = json.load(f)
                    # Invalidate cache if the number of files has changed
                    if cache_data.get('file_count') == file_count:
                        logger.info(f"Metadata cache hit. Loading max_depth = {cache_data['max_depth']}.")
                        return cache_data['max_depth']
            except Exception:
                pass
        
        try:
            # 2. Optimized TensorFlow Pipeline
            self._logger.info(f"Scanning {file_count} TFRecords for depth calculation. It might take some time.")
            file_patterns = [str(f) for f in tfrecord_files]

            def fast_parse(serial_example):
                """Minimal parsing: only extract the metadata feature to save RAM/CPU."""
                features = {'metadata': tf.io.FixedLenFeature([], tf.string)}
                return tf.io.parse_single_example(serial_example, features)

            # Create the optimized dataset pipeline
            dataset = tf.data.TFRecordDataset(file_patterns).map(
                fast_parse, 
                num_parallel_calls=tf.data.AUTOTUNE
            )

            series_counts_dict = {}
            series_study_dict = {}
    
            # 3. High-speed Counting Loop
            # We iterate only over the metadata blobs using a numpy iterator
            for metadata_blob in dataset.map(lambda x: x['metadata']).as_numpy_iterator():
                study_id = int.from_bytes(metadata_blob[0:5], byteorder='big', signed=False)
                series_id = int.from_bytes(metadata_blob[5:10], byteorder='big', signed=False)
                series_counts_dict[series_id] = series_counts_dict.get(series_id, 0) + 1
                series_study_dict[series_id] = study_id

            if not series_counts_dict:
                return 0

            max_series = max(series_counts_dict, key = series_counts_dict.get)
            max_depth = series_counts_dict[max_series]
            max_study = series_study_dict[max_series]


            # 4. Save cache with validation metadata
            with cache_file.open('w') as f:
                json.dump({
                    "max_depth": int(max_depth),
                    "study_id": max_study,
                    "series_id": max_series,
                    "file_count": file_count,
                    "created_at": str(pd.Timestamp.now())
                }, f)
            
            logger.info(f"Scanning completed. At most {max_depth} files were found per series. Study = {study_id} and series = {series_id}")

            return int(max_depth)

        except Exception as e:
            self._logger.error(f"Function {class_name}.{func_name} failed: str{e}")

    def get_max_series_depth(self):
        return self._max_series_depth

    def set_series_depth(self, depth):
        self._max_series_depth = depth
