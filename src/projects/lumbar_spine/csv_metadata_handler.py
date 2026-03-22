# coding: utf-8

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
import SimpleITK as sitk
from typing import Optional
import logging
import inspect
import ast
from pathlib import Path
from src.core.utils.logger import log_method


def to_tuple(val: str) -> Tuple[int, int]:
    """
    Helper function to manage str vs. tuple
    """
    return ast.literal_eval(val) if isinstance(val, str) else val


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
        label_enriched: str,
        train: str,
        config: dict,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initializes the handler by setting up paths and triggering data loading.

        Args:
            dicom_studies_dir (str): Path to the DICOM files directory.
            description (str): Path to the series descriptions CSV.
            label_coordinates (str): Path to the label coordinates CSV.
            label_enriched (str): Path to the enriched labels CSV.
            train (str): Path to the training labels CSV.
            config (dict): Configuration dictionary for processing.
            logger (Optional[logging.Logger]): Optional logger instance.
        """
        self._logger = logger or logging.getLogger(self.__class__.__name__)

        # Load and ensure all data is treated as string initially to prevent merging errors.
        self._dicom_studies_dir = Path(dicom_studies_dir)
        self._config = config
        self._format_cache = {}
        self._paths_dict: dict[str, Path] = {}
        self._series_desc_df = None
        self._label_coords_df = None
        self._train_df = None

        # Setup file paths
        self._setup_paths(series_description, label_coordinates, label_enriched, train)

        # Load dataframe and sanitize its data
        self._load_and_cleanse_data()

        self._logger.info(
            "Initializing CSVMetadataHandler object",
            extra={
                "action": "init",
                "paths": {key: str(val) for key, val in self._paths_dict.items()}
            }
        )

    def _setup_paths(
        self,
        series_description: str,
        label_coordinates: str,
        label_enriched: str,
        train: str
    ) -> None:
        """
        Defines paths for raw data and the enriched version (cache)
        """

        raw_paths = {
            'series_description': series_description,
            'train': train,
            'label_raw': label_coordinates,
            'label_enriched': label_enriched
        }

        root_dir_cfg = self._config.get("root_dir", None)
        if root_dir_cfg is None:
            error_msg = (
                "Fatal error: the parameter 'root_dir' "
                "is required but was not found. "
                "Please check your YAML file structure."
            )
            raise ValueError(error_msg)

        for key, original_path in raw_paths.items():
            # If the path is relative (e.g., "train.csv"), it is joined with root_dir.
            path = Path(original_path)

            # If it is already an absolute path (e.g., "/kaggle/input/..."),
            # the / operator leaves it unchanged.
            self._paths_dict[key] = Path(root_dir_cfg) / path

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
        # Load core metadata files
        self._series_desc_df = pd.read_csv(self._paths_dict['series_description'])
        self._train_df = pd.read_csv(self._paths_dict['train'])

        # Decide which coordinates file to use
        if self._paths_dict['label_enriched'].is_file():
            target_path = self._paths_dict['label_enriched']
            self._logger.info(f"Loading enriched coordinates from: {target_path.name}")
        else:
            target_path = self._paths_dict['label_raw']
            self._logger.info(f"Loading raw coordinates from: {target_path.name}")

        self._label_coords_df = pd.read_csv(target_path)

        # 1. Cleanse input data (converts everything to lowercase strings)
        self._train_df = self._filter_null_dataframe_values(self._train_df)
        self._series_desc_df = self._filter_null_dataframe_values(self._series_desc_df)

        # 2. Convert identifiers with error handling
        try:
            self._series_desc_df[['study_id', 'series_id']] = (
                self._series_desc_df[['study_id', 'series_id']].astype(int)
            )

            # Processing coordinates and formats
            self._label_coords_df = self._filter_null_dataframe_values(self._label_coords_df)

            int_cols = ['study_id', 'series_id', 'instance_number']
            self._label_coords_df[int_cols] = (
                self._label_coords_df[int_cols].astype(int)
            )

            self._label_coords_df[['x', 'y']] = (
                self._label_coords_df[['x', 'y']].astype(float)
            )

            if not self._paths_dict['label_enriched'].is_file():
                self._label_coords_df = self._scale_series_format_locations(self._label_coords_df)

                output_path = self._paths_dict['label_enriched']
                output_path.parent.mkdir(parents=True, exist_ok=True)

                self._label_coords_df.to_csv(output_path, index=False)

        except ValueError as e:
            self._logger.error(
                f"Type conversion failed: {str(e)}. "
                "Ensure CSV files do not contain non-numeric characters in ID columns."
            )
            raise  # Re-raise to stop the pipeline if data integrity is compromised

        # 3. Handle 'actual_file_format' tuple conversion
        if 'actual_file_format' in self._label_coords_df.columns:
            unique_formats = self._label_coords_df['actual_file_format'].unique()
            format_map = {
                fmt: ast.literal_eval(fmt.strip()) if isinstance(fmt, str) else fmt
                for fmt in unique_formats if pd.notna(fmt)
            }

            self._label_coords_df['actual_file_format'] = (
                self._label_coords_df['actual_file_format'].map(format_map)
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

        try:
            # Perform the initial merge and preprocessing upon instantiation.
            merged_df = self._merge_metadata()

            if merged_df.empty:
                error_msg = "Fatal error: empty DataFrame"
                logger.error(
                    error_msg,
                    exc_info=True,
                    extra={"status": "failed"}
                )
                raise ValueError(error_msg)

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
                )
                logger.warning(warning_msg)

            return encoded_metadata_df

        except Exception as e:
            self._logger.error(
                f"Error initializing CSVMetadataHandler: {str(e)}",
                exc_info=True,
                extra={"status": "failed", "error": str(e)}
            )
            raise

    def _filter_null_dataframe_values(
        self,
        data_df: pd.DataFrame
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
        data_df = (
            data_df.dropna()                                # 1. Remove actual NaN/None values
            .astype(str)                                    # 2. Convert remaining data to strings
            .apply(lambda s: s.str.lower().str.strip())     # 3. Standardize text format
            .replace('nan', pd.NA)                          # 4. Handle cases with "nan" strings
            .dropna()                                       # 5. Final sweep of newly created nulls
        )

        return data_df

    def _scale_series_format_locations(
        self,
        data_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Unifies coordinates based on series-level DICOM resolution.

        This method performs a multi-step cleaning and standardization process:

        2. Inspects DICOM file headers in parallel to retrieve actual pixel dimensions.
        3. Identifies the target format for each series (based on the largest dimensions
           found within that series).
        4. Calculates and applies a centered padding offset to (x, y) coordinates for
           any file not matching the series' target format.

        Args:
            data_df (pd.DataFrame): The raw metadata DataFrame containing
                at least 'study_id', 'series_id', 'instance_number', 'x', and 'y'.
                This DataFrame is supposedly cleansed

        Returns:
            pd.DataFrame: A sanitized DataFrame with 'actual_file_format' and
                'expected_file_format' columns, and updated unified coordinates.

        Note:
            A preliminary screening session showed several DICOM file formats used
            in the training data. While series are usually consistent, 35 series
            contain mixed formats (e.g., 640x640 and 608x608 simultaneously).

            To ensure downstream consistency, all files within a series are aligned
            to the largest detected format. Coordinates are updated in the DataFrame
            to reflect the future centered padding (e.g., a +16 pixel shift for
            608 to 640 conversion).

            The physical DICOM files remain unchanged at this stage and will be
            reformatted during the final image processing pipeline.
        """

        # 1. Initial cleansing
        data_df = data_df.copy()

        # 2. Retrieve file formats in parallel
        self._logger.info("Starting parallel DICOM format inspection")

        if 'actual_file_format' not in data_df.columns:
            must_inspect = True
        else:
            must_inspect = data_df['actual_file_format'].dropna().empty

        if must_inspect:
            # The dataframe is transformed in a list of dictionaries for the multithreading.
            records = data_df.to_dict('records')

            # By default, max_workers=None use the number of processors x5
            with ThreadPoolExecutor(max_workers=4) as executor:
                formats = list(tqdm(
                    executor.map(self._get_file_formats, records),
                    total=len(records),
                    desc="Reading file formats"
                ))
            data_df['actual_file_format'] = formats

        try:
            # 3. Unify format per series
            # Determine the "expected" format (the largest) for each series
            series_groups = data_df.groupby('series_id')['actual_file_format']

            # Identify mixed series for logging purposes
            mixed_series = series_groups.apply(lambda x: len(set(x)) > 1)
            mixed_series_ids = mixed_series[mixed_series].index.to_list()

            if mixed_series_ids:
                warning_msg = f"Detected {len(mixed_series_ids)} series with mixed DICOM formats"
                self._logger.warning(
                    warning_msg,
                    extra={"mixed_series_ids": mixed_series_ids}
                )

            # Map the largest format found in each series as the target
            # the string format must be converted into a tuple, using ast.
            series_target = series_groups.apply(
                lambda x: max(set(x), key=lambda f: f[0]*f[1])
            ).to_dict()

            data_df['expected_series_target'] = data_df['series_id'].map(series_target)

            # 4. Adapt coordinates based on format differences
            self._logger.info("Adapting coordinates using vectorized operations")

            # Extract dimensions into separate series for calculation
            # Actual format  (width, height)
            actual_w = data_df['actual_file_format'].str[0]
            actual_h = data_df['actual_file_format'].str[1]

            # Expected_series_target: (width, height)
            target_w = data_df['expected_series_target'].str[0]
            target_h = data_df['expected_series_target'].str[1]

            # Calculate padding offset (centered padding logic)
            # offset = (Target_Dim - Actual_Dim) / 2
            offset_x = (target_w - actual_w) / 2.0
            offset_y = (target_h - actual_h) / 2.0

            # Apply the shift to all rows at once
            data_df['x'] += offset_x
            data_df['y'] += offset_y

        except Exception as e:
            self.logger.error(
                f"Coordinate scaling failed: {str(e)}",
                exc_info=True,
                extra={"action": "scale_series_format", "error_type": type(e).__name__}
            )
            raise ValueError(f"Data inconsistency during coordinate scaling: {e}")

        modified_mask = (offset_x != 0) | (offset_y != 0)
        self._logger.info(f"Coordinate unification complete. {modified_mask.sum()} labels updated.")

        data_df = (
            data_df.drop(columns=['actual_file_format'])
            .rename(columns={'expected_series_target': 'actual_file_format'})
        )

        return data_df

    def _get_file_formats(
        self,
        record: dict
    ) -> Tuple[int, int]:
        """
        Retrieve DICOM dimensions from dictionary record
        """
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

            return (result)  # Returns (Width, Height)

        except Exception as e:
            self._logger.error(
                f"Error reading file format: {str(e)}",
                exc_info=True,
                extra={"status": "failed", "error": str(e)}
            )
            raise

    def _merge_metadata(self) -> pd.DataFrame:
        """
            Merges the training DataFrame with label coordinates and series descriptions.
            Includes detailed logging for each step of the merge process.
        """

        self._logger.info("Starting metadata merge process", extra={"action": "merge_metadata"})
        try:

            tmp_train_df = self._melt_and_clean_train_df()
            tmp_train_df = self._merge_with_label_coordinates(tmp_train_df)
            merged_df = self._merge_with_series_descriptions(tmp_train_df)
            merged_df = self._normalize_identifier_types(merged_df)

            self._logger.info(
                "Metadata merge completed successfully",
                extra={"status": "success", "final_shape": merged_df.shape}
            )
            return merged_df

        except Exception as e:
            self._logger.error(
                f"Error merging metadata: {str(e)}",
                exc_info=True,
                extra={"status": "failed", "error": str(e)}
            )
            raise

    def _melt_and_clean_train_df(self) -> pd.DataFrame:
        """
        Melt and clean the training DataFrame.
        """

        try:
            tmp_train_df = self._train_df.melt(
                id_vars="study_id",
                var_name="condition_level",
                value_name="severity"
            )

            if tmp_train_df.empty:
                error_msg = "Fatal error: method _melt_and_clean_train_df. Empty DataFrame"
                self._logger.error(error_msg, exc_info=True, extra={"status": "failed"})
                raise ValueError(error_msg)

            initial_count = len(tmp_train_df)

            tmp_train_df = tmp_train_df.dropna()
            final_count = len(tmp_train_df)

            if final_count == 0:
                error_msg = "Fatal error: method _melt_and_clean_train_df. Empty DataFrame"
                self._logger.error(error_msg, exc_info=True, extra={"status": "failed"})
                raise ValueError(error_msg)

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
            tmp_train_df['study_id'] = tmp_train_df['study_id'].astype(int)

            return tmp_train_df

        except Exception as e:
            self._logger.error(
                f"Error melting and cleaning training DataFrame: {str(e)}",
                exc_info=True,
                extra={"status": "failed", "error": str(e)}
            )
            raise

    def _merge_with_label_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the DataFrame with label coordinates.
        """

        try:
            self._logger.info("Merging with label coordinates...", extra={"step": 3})
            self._label_coords_df['condition_level'] = (
                self._label_coords_df['condition'] + '_' + self._label_coords_df['level']
            )

            self._label_coords_df['condition_level'] = (
                self._label_coords_df['condition_level']
                .str.replace(" ", "_", regex=False)
                .str.replace("/", "_", regex=False)
            )

            self._label_coords_df = self._label_coords_df.drop(columns=['condition', 'level'])

            merged_df = df.merge(
                self._label_coords_df,
                on=["study_id", 'condition_level'],
                how='inner'
            )

            if merged_df.empty:
                error_msg = "function pd.DataFrame.merge() failed. Empty DataFrame"
                raise ValueError(error_msg)

            self._logger.info(
                f"Merged with label coordinates. Shape: {merged_df.shape}",
                extra={"step": 3, "shape": merged_df.shape}
            )

        except Exception as e:
            error_msg = f"Fatal error in _merge_with_label_coordinates: {e}"
            self._logger.error(
                error_msg,
                exc_info=True,
                extra={"status": "failed", "error": str({e})}
            )
            raise e

        return merged_df

    def _merge_with_series_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the DataFrame with series descriptions.
        """

        try:
            self._logger.info("Merging with series descriptions...", extra={"step": 4})

            merged_df = df.merge(self._series_desc_df, on=['study_id', 'series_id'], how='inner')

            if merged_df.empty:
                error_msg = "function pd.DataFrame.merge() failed. Empty DataFrame"
                raise ValueError(error_msg)

            self._logger.info(
                f"Merged with series descriptions. Final shape: {merged_df.shape}",
                extra={"step": 4, "final_shape": merged_df.shape}
            )

        except Exception as e:
            error_msg = f"Fatal error in _merge_with_series_descriptions: {e}"
            self._logger.error(
                error_msg,
                exc_info=True,
                extra={"status": "failed", "error": str({e})}
            )
            raise e

        return merged_df

    def _normalize_identifier_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize identifier types.
        """

        self._logger.info("Normalizing identifier types...", extra={"step": 5})
        df['study_id'] = df['study_id'].astype(np.int64)
        df['series_id'] = df['series_id'].astype(np.int64)
        df['instance_number'] = df['instance_number'].astype(int)
        df.columns = [c.lower() for c in df.columns]

        return df

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
        logger.info(f"Starting function {class_name}.{func_name}")

        try:
            if metadata_df.empty:
                error_msg = "Empty DataFrame"
                raise ValueError(error_msg)

            # Define columns to encode and their corresponding mapping variables
            columns_to_encode = ["condition_level", "series_description", "severity"]

            # Create mappings for each column
            mappings = self._create_mappings(metadata_df, columns_to_encode)

            # Apply encoding to each column
            metadata_df = self._apply_encodings(metadata_df, columns_to_encode, mappings)

            msg_info = "Function _encode_dataframe completed successfully"
            logger.info(msg_info, extra={"status": "success"})
            return metadata_df

        except Exception as e:
            error_msg = (
                f"Fatal error in function {class_name}.{func_name}: {str(e)}"
            )

            logger.error(
                error_msg,
                exc_info=True,
                extra={"status": "failed", "error": str(e)}
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
        logger = logger or self._logger

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

            return mappings

        except Exception as e:
            error_msg = (
                f"Error in {self.__class__.__name__}.{func_name} "
                f"while processing column '{column}': {e}"
            )
            self._logger.error(
                error_msg,
                exc_info=True
            )
            raise

    @log_method()
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
        update_metadata_df = metadata_df.copy()
        for column in columns_to_encode:
            update_metadata_df[column] = (
                update_metadata_df[column].map(mappings[column]).fillna(-1).astype(int)
            )

        return update_metadata_df

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
