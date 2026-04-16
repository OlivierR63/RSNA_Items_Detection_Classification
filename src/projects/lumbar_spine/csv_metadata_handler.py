# coding: utf-8

from tqdm import tqdm
import pydicom
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
import SimpleITK as sitk
from typing import Optional
import logging
import inspect
import ast
from pathlib import Path
from multiprocessing import Pool, cpu_count
from src.core.utils.logger import log_method


def to_tuple(val: str) -> Tuple[int, int]:
    """
    Helper function to manage str vs. tuple
    """
    return ast.literal_eval(val) if isinstance(val, str) else val


def _extract_dicom_metadata_task(
    file_path: str
) -> Optional[Tuple[int, int, int, Tuple[int, int]]]:

    """
    Worker function optimized for Windows pickling.
    Using string path for better compatibility between processes.
    """
    try:
        path_obj = Path(file_path)

        # Fast extraction from path string
        instance_number = int(path_obj.stem)
        series_id = int(path_obj.parent.name)
        study_id = int(path_obj.parent.parent.name)

        # LIGHTWEIGHT READ: stop_before_pixels is vital for HDD performance
        # to minimize the amount of data pulled from the disk.
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)

        return (study_id, series_id, instance_number, (int(ds.Columns), int(ds.Rows)))

    except Exception:
        return None


class CSVMetadataHandler:
    """
    Handles the loading, merging, and preprocessing of metadata from three
    separate CSV files (series descriptions, label coordinates, and training labels).

    The primary goal is to create a single, comprehensive DataFrame used later
    for serialization and data lookup.
    """

    def __init__(
        self,
        dicom_studies_dir: str,
        series_description: str,
        label_coordinates: str,
        instances_series_format: str,
        train: str,
        config: dict,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initializes the CSV metadata handler and orchestrates data loading.

        This handler centralizes access to medical metadata by cross-referencing
        DICOM structures with labels and ensuring dimensional consistency across series.

        Args:
            dicom_studies_dir: Root directory path containing the hierarchical
                DICOM studies (StudyID/SeriesID/InstanceID.dcm).
            series_description: Path to the CSV mapping SeriesID to their
                anatomical views (e.g., Sagittal T1, Axial T2).
            label_coordinates: Path to the CSV containing ground truth landmark
                coordinates (x, y) for various spinal conditions.
            instances_series_format: Path to the inventory CSV documenting the
                original dimensions (W, H) of all ~142,000 instances and the
                calculated reference target format for each series.
                Note: Format values are parsed from strings to tuples via ast.literal_eval.
            train: Path to the main training labels CSV containing the
                target classes (Normal/Mild/Severe) per study.
            config: Global configuration dictionary containing processing
                hyperparameters and environment paths.
            logger: Dedicated logger instance for tracking data loading
                integrity and potential CSV inconsistencies.

        Raises:
            FileNotFoundError: If any of the mandatory CSV paths are invalid.
            ValueError: If the configuration dictionary is missing critical keys.
        """
        self._logger = logger or logging.getLogger(self.__class__.__name__)

        self._logger.debug("Starting CSVMetadataHandler object initialization")

        # Load and ensure all data is treated as string initially to prevent merging errors.
        self._dicom_studies_dir = Path(dicom_studies_dir)
        self._config = config
        self._format_cache = {}
        self._paths_dict: dict[str, Path] = {}
        self._series_desc_df = None
        self._label_coords_df = None
        self._instances_series_format_df = None
        self._train_df = None

        # Setup file paths
        self._setup_paths(series_description, label_coordinates, instances_series_format, train)

        # Load dataframe and sanitize its data
        self._load_and_cleanse_data()

        self._logger.debug("CSVMetadataHandler initialization completed")

    def _setup_paths(
        self,
        series_description: str,
        label_coordinates: str,
        instances_series_format: str,
        train: str
    ) -> None:
        """
        Defines paths for raw data and the enriched version (cache)
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(f"Starting function {class_name}.{func_name}")

        raw_paths = {
            'series_description': series_description,
            'train': train,
            'label_coordinates': label_coordinates,
            'instances_series_format': instances_series_format
        }

        root_dir_cfg = self._config.get("root_dir", None)
        if root_dir_cfg is None:
            critical_msg = (
                "Fatal error: the parameter 'root_dir' "
                "is required but was not found. "
                "Please check your YAML file structure."
            )
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure"}
            )

            raise ValueError(critical_msg)

        for key, original_path in raw_paths.items():
            # If the path is relative (e.g., "train.csv"), it is joined with root_dir.
            path = Path(original_path)

            # If it is already an absolute path (e.g., "/kaggle/input/..."),
            # the / operator leaves it unchanged.
            self._paths_dict[key] = Path(root_dir_cfg) / path

        self._logger.debug(f"self._paths_dict = {self._paths_dict}")

        self._logger.debug(
            f"Function {class_name}.{func_name} completed successfully",
            extra={"status": "success"}
        )

    def _load_and_cleanse_data(self) -> None:
        """
        Orchestrates metadata loading and multi-stage data type restoration.

        This method follows a specific sequence:
        1. Loads raw CSVs, prioritizing enriched files (with dimensions) when available.
        2. Performs an initial textual standardization (null removal & string normalization).
        3. Restores numeric types (int/float) for core features and identifiers.
        4. Reconstructs complex types (tuples) for image dimensions.

        Raises:
            ValueError: If numeric columns contain non-convertible standardized strings.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(f"Starting function {class_name}.{func_name}")

        # Load core metadata files
        self._train_df = pd.read_csv(self._paths_dict['train'])
        self._series_desc_df = pd.read_csv(self._paths_dict['series_description'])

        # DEBUG: check columns of self._series_desc_df right after loading from CSV file
        self._logger.debug(
            f"features list of self._series_desc_df: {self._series_desc_df.columns}"
        )

        # Load the cache format file when it exists.
        # In the other case, explore all the dicom files and build
        # the cache file for a next time
        format_path = self._paths_dict['instances_series_format']
        if format_path.is_file():
            self._instances_series_format_df = pd.read_csv(format_path)
        else:
            self._instances_series_format_df = self._get_instances_series_format()

            output_path = self._paths_dict['instances_series_format']
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self._instances_series_format_df.to_csv(output_path, index=False)

        # Load coordinates file
        coords_path = self._paths_dict['label_coordinates']
        self._logger.info(
            f"Loading raw coordinates from: {coords_path.name}"
        )

        self._label_coords_df = pd.read_csv(coords_path)

        # DEBUG: check columns of self._label_coords_df right after loading from CSV file
        self._logger.debug(
            f"features list of self._label_coords_df: \n{self._label_coords_df.columns}"
        )

        # 1. Cleanse input data (converts everything to lowercase strings)
        self._train_df = self._filter_null_dataframe_values(self._train_df)
        self._series_desc_df = self._filter_null_dataframe_values(self._series_desc_df)
        self._label_coords_df = self._filter_null_dataframe_values(self._label_coords_df)
        self._instances_series_format_df = (
            self._filter_null_dataframe_values(self._instances_series_format_df)
        )

        # DEBUG: Visualize all the columns of self._series_desc_df and self._label_coords_df
        # after _filter_null_dataframe_values
        if self._series_desc_df is not None:
            self._logger.debug(f"self._series_desc_df columns = {self._series_desc_df.columns}")

        if self._label_coords_df is not None:
            self._logger.debug(f"self._label_coords_df columns = {self._label_coords_df.columns}")

        if self._instances_series_format_df is not None:
            self._logger.debug(
                f"self._label_coords_df columns = {self._instances_series_format_df.columns}"
            )

        # 2. Convert identifiers with error handling
        try:
            self._series_desc_df[['study_id', 'series_id']] = (
                self._series_desc_df[['study_id', 'series_id']].astype(np.int64)
            )

            int_cols = ['study_id', 'series_id', 'instance_number']
            self._label_coords_df[int_cols] = (
                self._label_coords_df[int_cols].astype(np.int64)
            )

            self._label_coords_df[['x', 'y']] = (
                self._label_coords_df[['x', 'y']].astype(float)
            )

        except ValueError as e:
            # Critical: stop pipeline if data types cannot be guaranteed
            critical_msg = (
                f"Type conversion failed: {str(e)}. "
                "Ensure CSV files do not contain non-numeric characters in ID columns."
            )
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure"}
            )
            raise  # Re-raise to stop the pipeline if data integrity is compromised

        self._logger.debug(
            f"Function {class_name}.{func_name} completed successfully",
            extra={"status": "success"}
        )

    @property
    def train_df(self) -> pd.DataFrame:
        """
            Returns the raw training label DataFrame.
        """

        self._logger.info("Accessing raw training DataFrame", extra={"action": "get_train_df"})
        return self._train_df

    @log_method()
    def generate_metadata_dataframe(
        self,
        logger: Optional[logging.Logger] = None
    ) -> pd.DataFrame:

        """
        Orchestrates the complete metadata pipeline: cleaning, scaling, merging, and encoding.

        This method acts as the central hub for metadata preparation. It processes the raw
        CSV files, handles the specific (608, 608) to (640, 640) coordinate translation,
        and performs feature engineering for the deep learning model.

        Args:
            logger (Optional[logging.Logger]): Custom logger to track progress.
                Defaults to the class's internal logger.

        Returns:
            pd.DataFrame: The final, fully encoded and merged DataFrame containing
                all features (including 'condition_level') and corrected coordinates.

        Note:
            The feature 'condition_level' is a synthesized index combining 'condition'
            and 'level'. It relies on nb_levels = nunique(). This is safe here because
            this function is strictly applied to the full reference datasets, ensuring
            the presence of all possible classes and levels during processing.
        """

        logger = logger or self._logger

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(f"Starting function {class_name}.{func_name}")

        try:
            # Perform the initial merge and preprocessing upon instantiation.
            merged_df = self._merge_metadata()

            if merged_df.empty:
                critical_msg = "Fatal error: empty DataFrame"
                logger.critical(
                    critical_msg,
                    exc_info=True,
                    extra={"status": "failure"}
                )
                raise ValueError(critical_msg)

            logger.info(
                "Metadata merged successfully",
                extra={"merged_shape": merged_df.shape}
            )

            # Encode categorical metadata
            #    Fields affected : condition, level, series_description, severity
            encoded_metadata_df = self._encode_dataframe(merged_df)
            logger.info("Encoded categorical metadata", extra={"action": "encode_metadata"})

            # Create a new feature in metadata_df, as a synthesis of the
            # features 'condition' and 'series_description'
            nb_conditions_levels = encoded_metadata_df['condition_level'].nunique()

            if nb_conditions_levels != 25:  # Assuming 25 levels is the standard
                warning_msg = (
                    f"Unexpected number of levels detected: {nb_conditions_levels}. "
                    "Verify dataset integrity."
                    f"Quit function {class_name}{func_name}"
                )
                logger.warning(warning_msg)

            else:
                self._logger.debug(
                    f"Function {class_name}.{func_name} completed successfully",
                    extra={"status": "success"}
                )

            return encoded_metadata_df

        except Exception as e:
            critical_msg = (
                f"Error initializing CSVMetadataHandler: {str(e)}"
            )
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise

    def _filter_null_dataframe_values(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Removes null values and standardizes all remaining data into lowercase strings.

        This method performs a strict textual normalization by:
        1. Dropping existing NaN/None values.
        2. Casting all remaining data points to strings.
        3. Stripping whitespace and converting to lowercase.
        4. Catching and removing any resulting 'nan' string artifacts.

        Args:
            data_df (pd.DataFrame): The DataFrame to be standardized.

        Returns:
            pd.DataFrame: A DataFrame where every element is a cleaned string.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(f"Starting function {class_name}.{func_name}")

        data_df = df.copy()

        try:
            data_df = (
                # 1. Remove actual NaN/None values
                data_df.dropna()

                # 2. Convert remaining data to strings
                .astype(str)

                # 3. Standardize text format
                .apply(lambda s: s.str.lower().str.strip())

                # 4. Handle cases with "nan" strings
                .replace('nan', pd.NA)

                # 5. Final sweep of newly created nulls
                .dropna()
            )

            self._logger.debug(
                f"Function {class_name}.{func_name} completed successfully",
                extra={"status": "success"}
            )
            return data_df

        except Exception as e:
            critical_msg = f"Fatal error in {class_name}.{func_name}: {e}"
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure"}
            )
            raise

    def _get_instances_series_format(self) -> pd.DataFrame:
        """
        Scans DICOM files and extracts dimensions.
        Save outcome in CSV file
        Optimized for Windows 10 and HDD environments.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        root_path = Path(self._dicom_studies_dir).resolve()

        if self._logger:
            self._logger.info(f"Starting scan on HDD: {root_path}")

        try:
            # 1. Step 1: Gather paths (Sequential on HDD is faster for discovery)
            all_dcm_files = [str(dcm_file) for dcm_file in root_path.rglob("*.dcm")]
            total_files = len(all_dcm_files)

            # 2. Step 2: Parallel processing with limited workers
            # IMPORTANT FOR HDD: Do not use all CPU cores.
            # Too many concurrent disk requests will cause "Disk Thrashing" on a HDD.
            # We limit to 2 or 4 workers max to keep disk I/O semi-sequential.
            num_workers = min(cpu_count(), 4)

            if self._logger:
                self._logger.info(f"Processing {total_files} files using {num_workers} workers.")

            results: List[Tuple] = []

            # 'chunksize' is increased to 50 (see yaml config file) to reduce the overhead of
            # communication between CPU and HDD.
            chunksize = self._config['system']['chunksize']
            with Pool(processes=num_workers) as pool:
                for res in tqdm(
                    pool.imap_unordered(
                        _extract_dicom_metadata_task,
                        all_dcm_files,
                        chunksize=chunksize
                    ),
                    total=total_files,
                    desc="HDD Metadata Extraction"
                ):
                    if res:
                        results.append(res)

            # 3. Assemble and Sort
            data_df = pd.DataFrame(results, columns=[
                'study_id',
                'series_id',
                'instance_number',
                'instance_format'
            ])

            data_df = data_df.sort_values(
                by=['study_id', 'series_id', 'instance_number']
            ).reset_index(drop=True)

            # 4. Add a new feature "target_series_format" to the dataframe
            data_df = self._calculate_target_series_format(data_df)

            # DEBUG : display the 500 headlines of the dataframe
            self._logger.debug(f"data_df = \n{data_df}")

            if self._logger:
                self._logger.info(f"Extraction complete. {len(data_df)} instances cataloged.")
                self._logger.debug(
                    f"Function {class_name}.{func_name} completed successfully",
                    extra={"status": "success"}
                )

            return data_df

        except Exception as e:
            critical_msg = f"Fatal error in Function {class_name}.{func_name}: {e}"
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure"}
            )
            raise

    def _calculate_target_series_format(self, df: pd.DataFrame) -> None:

        """
        Harmonizes the image dimensions within each series to a uniform square format.

        This method identifies the largest dimension (width or height) across all
        instances of a given series and defines a square target format (max_dim, max_dim).
        This ensures spatial consistency for 3D/volumetric processing and prevents
        anatomical distortion during subsequent padding or resizing steps.

        The calculated format is mapped back to the input DataFrame in a new column.

        Args:
            df (pd.DataFrame): The main metadata DataFrame containing at least
                'series_id' and 'instance_format' columns.
                Note: 'instance_format' may contain string representations of tuples
                (e.g., "(320, 320)") which are safely converted using ast.literal_eval.

        Returns:
            None: The 'target_series_format' column is added in-place to the data_df.

        Process:
            1. Scans 'instance_format' and ensures all values are Python tuples.
            2. Groups data by 'series_id' to find the global maximum dimension per series.
            3. Creates a square (N, N) reference tuple for each series.
            4. Maps these reference formats back to every instance in the DataFrame.

        Debug:
            Logs a sample (first 10 entries) of the mapping dictionary to verify
            the homogenization logic without flooding the log files.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(
            f"Starting function {class_name}.{func_name}"
        )

        data_df = df.copy()

        try:
            # Unify format per series

            # CRITICAL: Ensure instance_format is a tuple, not a string from CSV
            if isinstance(data_df['instance_format'].iloc[0], str):
                data_df['instance_format'] = data_df['instance_format'].apply(ast.literal_eval)

            # Determine the largest dimension for each series
            series_max_dim_df = data_df.groupby('series_id')['instance_format'].apply(
                lambda formats: max([dim for img_format in formats for dim in img_format])
            )

            # Create the square format tuple
            series_target = series_max_dim_df.apply(lambda m: (m, m)).to_dict()

            # Safer debug for large dataset: ony log the 10 first lines
            sample_target = dict(list(series_target.items())[:10])
            self._logger.debug(
                f"Function {class_name}.{func_name}: Sample of series_target =\n{sample_target}"
            )

            # Map back to the dataframe
            data_df['target_series_format'] = data_df['series_id'].map(series_target)

            self._logger.debug(
                    f"Function {class_name}.{func_name} completed successfully",
                    extra={"status": "success"}
                )

            return data_df

        except Exception as e:
            critical_msg = f"Fatal error in Function {class_name}.{func_name}: {e}"
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure"}
            )
            raise

    def _get_file_formats(
        self,
        record: dict
    ) -> Tuple[int, int]:
        """
        Retrieve DICOM dimensions from dictionary record
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(
            f"Starting function {class_name}.{func_name}"
        )

        try:
            study_id = int(float(record['study_id']))
            series_id = int(float(record['series_id']))
            instance = int(float(record['instance_number']))

            # Create a unique key for this specific image
            dcm_file = Path(self._dicom_studies_dir) / f"{study_id}/{series_id}/{instance}.dcm"
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(dcm_file))
            reader.ReadImageInformation()
            full_size = reader.GetSize()

            # Note : GetSize() returns a tuple (Width, Height, Depth), analog to (x, y, z)
            result = (full_size[0], full_size[1])

            self._logger.debug(
                f"Function {class_name}.{func_name} completed successfully",
                extra={"status": "success"}
            )

            return (result)  # Returns (Width, Height)

        except Exception as e:
            self._logger.critical(
                f"Error reading file format: {str(e)}",
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise

    def _merge_metadata(self) -> pd.DataFrame:
        """
        Orchestrates the complete metadata pipeline for training.

        Execution steps:
            1. Melt and clean main training labels.
            2. Merge with ground truth (x, y) coordinates.
            3. Merge with series descriptions (anatomical views).
            4. Merge with DICOM instance dimensions and target formats.
            5. Clean missing records and revise coordinates for centered padding.
            6. Normalize types for database consistency.

        Note:
            A preliminary screening session showed several DICOM file formats used
            in the training data. While series are usually consistent, 35 series
            contain mixed formats (e.g., 640x640 and 608x608 simultaneously).

            To ensure downstream consistency, all files within a series are aligned
            to the largest detected format. Coordinates are updated in the DataFrame
            to reflect the future centered padding (e.g., a +16 pixel shift for
            608 to 640 conversion).

            The physical DICOM files REMAIN UNCHANGED AT THIS STAGE. They will be
            reformatted during the final image processing pipeline.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(
            f"Starting function {class_name}.{func_name}"
        )

        try:
            with pd.option_context(
                'display.max_columns', None,   # Unlimited columns
                'display.expand_frame_repr', False,  # No multi-blocks split
                'display.max_colwidth', None   # No cell truncation
            ):
                tmp_train_1_df = self._melt_and_clean_train_df()
                debug_msg = (
                    "After calling _melt_and_clean_train_df: "
                    f"tmp_train_1_df = \n{tmp_train_1_df.head(25)}"
                )
                self._logger.debug(debug_msg)

                tmp_train_2_df = self._merge_with_label_coordinates(tmp_train_1_df)
                debug_msg = (
                    "After calling _merge_with_label_coordinates: "
                    f"tmp_train_2_df = \n{tmp_train_2_df.head(25)}"
                )
                self._logger.debug(debug_msg)

                tmp_train_3_df = self._merge_with_series_descriptions(tmp_train_2_df)
                debug_msg = (
                    "After calling _merge_with_series_descriptions: "
                    f"tmp_train_3_df sample = \n{tmp_train_3_df.head(25)}"
                )
                self._logger.debug(debug_msg)

                merged_df = self._merge_with_instances_and_series_format(tmp_train_3_df)
                debug_msg = (
                    "After calling _merge_with_instances_and_series_format: "
                    f"merged_df sample = \n{merged_df.head(25)}"
                )
                self._logger.debug(debug_msg)

                cleansed_df = self._remove_null_data(merged_df)

                # Update the coordinates in case the image is padded and enlarged
                # into the series format.
                revised_xy_df = self._revise_xy_coords_in_merged_dataframe(cleansed_df)
                debug_msg = (
                    "After calling _revise_xy_coords_in_merged_dataframe: "
                    f"revised_xy_df sample = \n{revised_xy_df.head(25)}"
                )
                self._logger.debug(debug_msg)

                # Normalize numeric types and standardize string casing
                normalized_df = self._normalize_identifier_types(revised_xy_df)
                debug_msg = (
                    "After calling _normalize_identifier_types: "
                    f"normalized_df = \n{normalized_df.head(25)}"
                )
                self._logger.debug(debug_msg)

                self._logger.debug(
                    f"Function {class_name}.{func_name} completed successfully",
                    extra={"status": "success"}
                )

            return normalized_df

        except Exception as e:
            critical_msg = (
                f"Fatal error in function {class_name}.{func_name}, while merging data: {e}"
            )
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise

    def _melt_and_clean_train_df(self) -> pd.DataFrame:
        """
        Melt and clean the training DataFrame.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(
            f"Starting function {class_name}.{func_name}"
        )

        try:
            tmp_train_df = self._train_df.melt(
                id_vars="study_id",
                var_name="condition_level",
                value_name="severity"
            )

            if tmp_train_df.empty:
                critical_msg = "Fatal error: method _melt_and_clean_train_df. Empty DataFrame"
                self._logger.critical(critical_msg, exc_info=True, extra={"status": "failure"})
                raise ValueError(critical_msg)

            initial_count = len(tmp_train_df)

            tmp_train_df = tmp_train_df.dropna()
            final_count = len(tmp_train_df)

            if final_count == 0:
                critical_msg = "Fatal error: method _melt_and_clean_train_df. Empty DataFrame"
                self._logger.error(critical_msg, exc_info=True, extra={"status": "failure"})
                raise ValueError(critical_msg)

            self._logger.info(
                f"Dropped {initial_count - final_count} NaN rows",
                extra={
                    "step": 1,
                    "initial_count": initial_count,
                    "final_count": final_count
                }
            )

            # Standardize severity text as well
            tmp_train_df['severity'] = tmp_train_df['severity'].str.lower().str.strip()

            # CRITICAL: Force types for merging
            # This ensures study_id is an integer for subsequent joins
            tmp_train_df['study_id'] = tmp_train_df['study_id'].astype(np.int64)

            # Count the number of lines with one null value or more.
            # Remove all the lines with null values
            nb_null_lines = tmp_train_df.isna().any(axis=1).sum()

            self._logger.debug(
                f"Function {class_name}.{func_name}: there are {nb_null_lines} "
                "lines with one null value or more in the DataFrame when merge "
                "is completed."
            )
            tmp_train_df = tmp_train_df.dropna()

            self._logger.debug(
                f"Function {class_name}.{func_name}: "
                "Remove null values from the dataframe"
            )

            if tmp_train_df.empty:
                critical_msg = (
                    "Merge result is empty. No matching 'study_id' and 'condition_level' "
                    "found between input data and label coordinates."
                )

                self._logger.critical(
                    critical_msg,
                    exc_info=True,
                    extra={"status": "failure"}
                )
                raise ValueError(critical_msg)

            self._logger.debug(
                f"Function {class_name}.{func_name} completed successfully",
                extra={"status": "success"}
            )

            return tmp_train_df

        except Exception as e:
            critical_msg = (
                f"Fatal error in {class_name}.{func_name} "
                f"(melting and cleaning training DataFrame): {str(e)}"
            )
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise

    def _merge_with_label_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the DataFrame with label coordinates.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(
            f"Starting function {class_name}.{func_name}"
        )
        data_df = df.copy()

        # DEBUG: Marker to visualize the shape of self._label_coords_df
        # (Only the 25 first line of the DataFrame).
        self._logger.debug(f"self._label_coords_df = \n{self._label_coords_df.head(25)}")

        try:
            self._logger.info("Merging with label coordinates...", extra={"step": 3})
            self._label_coords_df['condition_level'] = (
                self._label_coords_df['condition'] + '_' + self._label_coords_df['level']
            )

            # Create a local copy to avoid mutating self._label_coords_df permanently
            coords_prepared_df = self._label_coords_df.copy()

            coords_prepared_df['condition_level'] = (
                coords_prepared_df['condition'].str.replace(r" ", "_", regex=True) +
                '_' +
                coords_prepared_df['level'].str.replace(r"[ /]", "_", regex=True)
            )

            # DEBUG: Marker to visualize the shape of coords_prepared_df
            # (Only the 25 first line of the DataFrame).
            self._logger.debug(f"coords_prepared_df = \n{coords_prepared_df.head(25)}")

            coords_prepared_df = coords_prepared_df.drop(columns=['condition', 'level'])

            merged_df = data_df.merge(
                coords_prepared_df,
                on=["study_id", 'condition_level'],
                how='left'
            )

            # Count the number of lines with one null value or more.
            # Remove all the lines with null values
            nb_null_lines = merged_df.isna().any(axis=1).sum()

            self._logger.debug(
                f"Function {class_name}.{func_name}: there are {nb_null_lines} "
                "lines with one null value or more in the DataFrame when merge "
                "is completed."
            )
            merged_df = merged_df.dropna()

            self._logger.debug(
                f"Function {class_name}.{func_name}: "
                "Remove null values from the dataframe"
            )

            if merged_df.empty:
                critical_msg = (
                    "Merge result is empty. No matching 'study_id' and 'condition_level' "
                    "found between input data and label coordinates."
                )

                self._logger.critical(
                    critical_msg,
                    exc_info=True,
                    extra={"status": "failure"}
                )
                raise ValueError(critical_msg)

            # When no issue is found, the location is shifted outside of the
            # image frame.
            condition = (
                merged_df['severity']
                .str.lower()
                .str.contains("normal */ *mild", regex=True)
            )
            merged_df['x'] = np.where(condition, -1, merged_df['x'])
            merged_df['y'] = np.where(condition, -1, merged_df['y'])

            self._logger.info(
                f"Merged with label coordinates. Shape: {merged_df.shape}",
                extra={"step": 3, "shape": merged_df.shape}
            )

            self._logger.debug(
                f"Function {class_name}.{func_name} completed successfully",
                extra={"status": "success"}
            )

        except Exception as e:
            critical_msg = f"Fatal error in _merge_with_label_coordinates: {e}"
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise

        return merged_df

    def _merge_with_series_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the DataFrame with series descriptions.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(
            f"Starting function {class_name}.{func_name}"
        )

        data_df = df.copy()
        for col in ['study_id', 'series_id']:
            data_df[col] = data_df[col].astype(np.int64)

        try:
            self._logger.info("Merging with series descriptions...", extra={"step": 4})

            merged_df = data_df.merge(
                self._series_desc_df,
                on=['study_id', 'series_id'],
                how='left'
            )

            # Count the number of lines with one null value or more.
            # Remove all the lines with null values
            nb_null_lines = merged_df.isna().any(axis=1).sum()

            self._logger.debug(
                f"Function {class_name}.{func_name}: there are {nb_null_lines} "
                "lines with one null value or more in the DataFrame when merge "
                "is completed."
            )
            merged_df = merged_df.dropna()

            self._logger.debug(
                f"Function {class_name}.{func_name}: "
                "Remove null values from the dataframe"
            )

            if merged_df.empty:
                critical_msg = (
                    "Merge result is empty. No matching 'study_id' and 'condition_level' "
                    "found between input data and label coordinates."
                )

                self._logger.critical(
                    critical_msg,
                    exc_info=True,
                    extra={"status": "failure"}
                )
                raise ValueError(critical_msg)

            self._logger.info(
                f"Merging with series descriptions completed. Final shape: {merged_df.shape}",
                extra={"step": 4, "final_shape": merged_df.shape}
            )

            self._logger.debug(
                f"Function {class_name}.{func_name} completed successfully",
                extra={"status": "success"}
            )

        except Exception as e:
            critical_msg = f"Fatal error in _merge_with_series_descriptions: {e}"
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise

        return merged_df

    def _merge_with_instances_and_series_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches the training metadata with DICOM image dimensions.

        Adds 'instance_format' (original size) and 'target_series_format' (homogenized size)
        columns to the DataFrame. This metadata is required for future coordinate
        adjustments and image padding.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(
            f"Starting function {class_name}.{func_name}"
        )

        data_df = df.copy()
        self._logger.debug(f"data_df = {data_df.head(50).to_string()}")

        duplicate_df = self._instances_series_format_df.copy()

        for col in ['study_id', 'series_id', 'instance_number']:
            data_df[col] = data_df[col].astype(np.int64)
            duplicate_df[col] = duplicate_df[col].astype(np.int64)

        try:
            self._logger.info("Merging with instances and series formats...", extra={"step": 5})

            merged_df = data_df.merge(
                duplicate_df,
                on=['study_id', 'series_id', 'instance_number'],
                how='left'
            )

            # Count the number of lines with one null value or more.
            # Remove all the lines with null values
            nb_null_lines = merged_df.isna().any(axis=1).sum()

            self._logger.debug(
                f"Function {class_name}.{func_name}: there are {nb_null_lines} "
                "lines with one null value or more in the DataFrame when merge "
                "is completed."
            )
            merged_df = merged_df.dropna()

            self._logger.debug(
                f"Function {class_name}.{func_name}: "
                "Remove null values from the dataframe"
            )

            if merged_df.empty:
                critical_msg = (
                    "Merge result is empty. No matching 'study_id' and 'condition_level' "
                    "found between input data and label coordinates."
                )

                self._logger.critical(
                    critical_msg,
                    exc_info=True,
                    extra={"status": "failure"}
                )
                raise ValueError(critical_msg)

            self._logger.info(
                (
                    "Merging with instances and series formats dataframe completed. "
                    f"Final shape: {merged_df.shape}"
                ),
                extra={"step": 5, "final_shape": merged_df.shape}
            )

            self._logger.debug(
                f"Function {class_name}.{func_name} completed successfully",
                extra={"status": "success"}
            )

            return merged_df

        except Exception as e:
            critical_msg = f"Fatal error in _merge_with_instances_and_series_format: {e}"
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise

    def _remove_null_data(self, df: pd.DataFrame):
        """
        Removes records containing any null values and logs the dropped data.
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(
            f"Starting function {class_name}.{func_name}"
        )

        try:
            data_df = df.copy()

            self._logger.info("Removing null metadata in the dataframe...", extra={"step": 6})

            # Identify records with at least one null value
            # axis=1 checks row-wise; .any() returns True if any column is NaN
            mask = data_df.isna().any(axis=1)
            null_records_df = data_df.loc[mask]

            # DEBUG: display all the records with null values (if there are any)
            if not null_records_df.empty:
                warning_msg = (
                    f"Found {len(null_records_df)} records with null values. They will be dropped"
                    f"{null_records_df.to_string()}"
                )
                self._logger.warning(warning_msg)

                debug_msg = (
                    f"Null records detailed view:\n {null_records_df.to_string()}"
                )
                self._logger.debug(debug_msg)

            # Remove all the records with null elements
            data_df = data_df.dropna()

            info_msg = (
                f"Null values removed from the dataframe. "
                f"Final shape: {data_df.shape}"
            )
            self._logger.info(
                info_msg,
                extra={"step": 6, "final_shape": data_df.shape}
            )

            self._logger.debug(
                f"Function {class_name}.{func_name} completed successfully",
                extra={"status": "success"}
            )

            return data_df

        except Exception as e:
            critical_msg = f"Fatal error in {class_name}.{func_name}: {e}"
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure"}
            )
            raise

    def _revise_xy_coords_in_merged_dataframe(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adjusts (x, y) coordinates based on centered padding offsets.
        Ensures coordinates remain valid even if some format data is missing.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(
            f"Starting function {class_name}.{func_name}"
        )

        data_df = df.copy()

        try:
            # Adapt coordinates based on format differences
            self._logger.info("Adapting coordinates using vectorized operations", extra={"step": 7})

            # Extract dimensions into separate series for calculation
            # Actual format  (width, height)
            def safe_parse(val):
                if pd.isna(val) or not isinstance(val, str):
                    return val
                try:
                    return ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    return None

            data_df['instance_format'] = data_df['instance_format'].apply(safe_parse)
            data_df['target_series_format'] = data_df['target_series_format'].apply(safe_parse)

            # Identify valid rows to prevent propagating NaNs to coordinates
            mask = (
                data_df['instance_format'].notna()
                & data_df['target_series_format'].notna()
                & (data_df['severity'] != 'normal/mild')
            )

            # Actual image format (width, height)
            actual_w = data_df.loc[mask, 'instance_format'].str[0].astype(np.int64)
            actual_h = data_df.loc[mask, 'instance_format'].str[1].astype(np.int64)

            # Target series format: (width, height)
            target_w = data_df.loc[mask, 'target_series_format'].str[0].astype(np.int64)
            target_h = data_df.loc[mask, 'target_series_format'].str[1].astype(np.int64)

            # Calculate padding offset (centered padding logic)
            offset_x = (target_w - actual_w) / 2.0
            offset_y = (target_h - actual_h) / 2.0

            # Apply the shift to all rows at once
            data_df.loc[mask, 'x'] += offset_x
            data_df.loc[mask, 'y'] += offset_y

            # Make the x and y coordinates linked with the 'normal/mild'
            # case remain out of the image frame
            mask_2 = data_df['severity'] == 'normal/mild'
            data_df.loc[mask_2, 'x'] = -1.0
            data_df.loc[mask_2, 'y'] = -1.0

            self._logger.info("All coordinates revised successfully", extra={"step": 7})
            self._logger.debug(
                f"Function {class_name}.{func_name} completed successfully",
                extra={"status": "success"}
            )

            return data_df

        except Exception as e:
            critical_msg = f"Fatal error in {class_name}.{func_name}: {e}"
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise

    def _normalize_identifier_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes identifier types and column naming for database consistency.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(
            f"Starting function {class_name}.{func_name}"
        )

        data_df = df.copy()

        try:
            self._logger.info("Normalizing identifier types...", extra={"step": 5})
            data_df['study_id'] = data_df['study_id'].astype(np.int64)
            data_df['series_id'] = data_df['series_id'].astype(np.int64)
            data_df['instance_number'] = data_df['instance_number'].astype(int)
            data_df.columns = [c.lower() for c in data_df.columns]

            self._logger.debug(
                f"Function {class_name}.{func_name} completed successfully",
                extra={"status": "success"}
            )

            return data_df

        except Exception as e:
            critical_msg = f"Fatal error in {class_name}.{func_name}: {e}"
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise

    @log_method()
    def _encode_dataframe(
        self,
        data_df: pd.DataFrame,
        *,
        logger: Optional[logging.Logger] = None
    ) -> pd.DataFrame:

        """
            Converts categorical textual metadata fields in a DataFrame to numerical values.
            This process is essential for preparing data for serialization
            or machine learning models,replacing descriptive strings with compact integers.

            Args:
                data_df: The DataFrame containing metadata to be converted.
                    Example of column values: {
                        "condition_level": "spinal_canal_stenosis_l1-2",
                        "severity": "Normal/Mild", ...
                    }

            Returns:
                pd.DataFrame: The DataFrame with the specified columns converted to integer values.
        """
        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger
        logger.debug(f"Starting function {class_name}.{func_name}")

        try:
            if data_df.empty:
                critical_msg = "Empty DataFrame"

                self._logger.critical(
                    critical_msg,
                    exc_info=True,
                    extra={"status": "failure"}
                )
                raise ValueError(critical_msg)

            # Define columns to encode and their corresponding mapping variables
            columns_to_encode = ["condition_level", "series_description", "severity"]

            # Create mappings for each column
            mappings_dict = self._create_mappings(data_df, columns_to_encode)

            # Apply encoding to each column
            metadata_df = self._apply_encodings(data_df, columns_to_encode, mappings_dict)

            msg_debug = f"Function {class_name}.{func_name} completed successfully"
            logger.debug(msg_debug, extra={"status": "success"})

            return metadata_df

        except Exception as e:
            critical_msg = f"Fatal error in function {class_name}.{func_name}: {str(e)}"
            logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise

    @log_method()
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

        logger.debug(f"Starting function {class_name}.{func_name}")

        mappings = {}
        try:
            for column in columns_to_encode:
                if column not in metadata_df.columns:
                    self._logger.warning(f"Column '{column}' not found in DataFrame. Skipping.")
                    continue

                # Sort values to ensure mapping [0, 1, 2...] is always the same
                values = sorted(metadata_df[column].dropna().unique().tolist())

                # We assume self._create_string_to_int_mapper returns an object
                # with a .mapping attribute
                mapper = self._create_string_to_int_mapper(values)
                mappings[column] = mapper.mapping

                self._logger.info(
                    f"Created mapping for '{column}': {len(values)} categories found."
                )

            msg_debug = f"Function {class_name}.{func_name} completed successfully"
            logger.debug(msg_debug, extra={"status": "success"})

            return mappings

        except Exception as e:
            critical_msg = (
                f"Error in {self.__class__.__name__}.{func_name} "
                f"while processing column '{column}': {e}"
            )
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure"}
            )
            raise

    @log_method()
    def _apply_encodings(self, metadata_df: pd.DataFrame,
                         columns_to_encode: List[str],
                         mappings_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
            Applies the encoding mappings to each specified column in the DataFrame.

            Args:
                metadata_df: The DataFrame containing metadata to be converted.
                columns_to_encode: Dictionary mapping column names to their mapping variable names.
                mappings_dict: Dictionary of mapping dictionaries for each column.

            Returns:
                pd.DataFrame: The DataFrame with the specified columns converted to integer values.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        self._logger.debug(f"Starting function {class_name}.{func_name}")

        try:
            update_metadata_df = metadata_df.copy()
            for column in columns_to_encode:
                update_metadata_df[column] = (
                    update_metadata_df[column].map(mappings_dict[column]).fillna(-1).astype(int)
                )

            msg_debug = f"Function {class_name}.{func_name} completed successfully"
            self._logger.debug(msg_debug, extra={"status": "success"})

            return update_metadata_df

        except Exception as e:
            critical_msg = f"Fatal error in {class_name}.{func_name}: {e}"
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure"}
            )
            raise

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
                observed_pathologies_lst (list): A list of strings to be mapped
                                (e.g., ["Normal", "Stenosis"])
                                The order of the strings defines their integer value.

            Returns:
                callable: A function that maps an input string to an integer.
                          The resulting function includes 'mapping' and 'reverse_mapping
                          dictionaries as attributes.
        """

        func_name = inspect.currentframe().f_code.co_name
        class_name = self.__class__.__name__

        logger = logger or self._logger
        logger.debug(f"Starting function {class_name}.{func_name}")

        try:
            # Create the primary dictionary {string: integer} using enumeration.
            # The integer (i) corresponds to the string's index + 1 in the list.
            mapping_dict = {key: idx for idx, key in enumerate(observed_pathologies_lst)}

            # Create the optional reverse dictionary {integer: string}
            # for inspection/deserialization.
            reverse_mapping_dict = {idx: key for idx, key in enumerate(observed_pathologies_lst)}

            def mapper(key: str) -> int:
                """
                Maps an input string to its corresponding integer index.
                """
                # Use .get() to return the mapped integer.
                # Returns -1 if the input string is not found (unknown category).
                return mapping_dict.get(key, -1)

            # Attach the mapping dictionaries as attributes to the mapper function.
            # This allows users to inspect or reverse the mapping later.
            mapper.mapping = mapping_dict
            mapper.reverse_mapping = reverse_mapping_dict

            # Return the callable function
            logger.debug(
                f"Function {class_name}.{func_name} completed successfully",
                extra={"status": "success"}
            )
            return mapper

        except Exception as e:
            critical_msg = f"Error in function {class_name}.{func_name} : {str(e)}"
            logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise
