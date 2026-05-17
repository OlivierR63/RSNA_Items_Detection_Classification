# coding: utf-8

from pathlib import Path
from typing import Any, Dict, List
import yaml
import logging
import json
import numpy as np
from src.core.utils.singleton_meta import SingletonMeta
from src.config.schema import OneOf, Sequence, REQUIRED_SCHEMA


class ConfigLoader(metaclass=SingletonMeta):
    """
    A singleton class that loads configuration settings from a YAML file
    and resolves relative paths.

    This class handles the parsing of the configuration file and ensures that
    all relevant directory and file paths (e.g., DICOM and CSV files)
    are made absolute relative to the location of the configuration file itself.
    Since it implements the singleton pattern, only one instance of the
    configuration is maintained throughout the application's lifecycle.
    """

    def __init__(self, config_path: str = None) -> None:
        """
        Initializes the singleton instance by reading the YAML file and resolving paths.
        Subsequent calls to the constructor return the existing instance without
        re-executing this logic.

        Args:
            config_path (str, optional): File system path to the YAML configuration file.
            Required only for the first instantiation

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the YAML file is malformed.
        """
        # Check if the singleton has already been initialized to avoid redundant setup
        if hasattr(self, '_config'):
            return  # Already initialized by a previous call

        # Ensure the configuration path is provided during the initial call
        if config_path is None:
            raise ValueError(
                "ConfigLoader must be initialized with a config_path on its first call."
            )

        # Check file existence
        config_file_path = Path(config_path).resolve()
        if not config_file_path.exists():
            raise FileNotFoundError(f"Configuration file {config_path} not found.")

        self._load_and_initialize_dict(config_file_path)
        self._resolve_all_paths(config_file_path.parent)

        # Final check: Verify the compliance of the config dictionary
        self._check_config_compliance()

    def _load_and_initialize_dict(self, config_file_path: Path) -> None:
        """
        Loads the YAML configuration file and initializes the internal dictionary.

        This method reads the configuration file, handles potential YAML parsing
        errors, and sets up essential base attributes such as the root directory
        and default placeholders for runtime variables.

        Args:
            config_file_path (Path): The resolved path to the YAML configuration file.

        Raises:
            ValueError: If the YAML file is malformed or cannot be parsed.
        """
        # Determine the directory containing the configuration file.
        # All relative paths in the config will be resolved based on this directory.
        config_dir = config_file_path.parent

        # Load the configuration data first (Initialization of self._config)
        try:
            with config_file_path.open('r') as f:
                self._config: Dict[str, Any] = yaml.safe_load(f) or {}  # Handles empty YAML files

        except yaml.YAMLError as e:
            raise ValueError(f"Error loading YAML configuration file: {e}")

        # Store the determined root directory (the YAML file's location)
        self._config["root_dir"] = str(config_dir)

        # At this time, just initialize the attribute "series_depth"
        self._config["series_depth"] = None

    def _resolve_all_paths(self, config_dir: Path) -> None:
        """
        Helper to resolve paths, reducing __init__ complexity.
        """
        # Reference to the paths dictionary to avoid repetitive lookups
        paths_cfg = self._config.get('paths', {})

        # --- Resolve Core Relative Paths ---
        # Iterate through known keys to convert relative paths (starting with '.') to absolute
        core_keys = [
            "dicom_studies",
            "tfrecord",
            "tfrecord_metadata_cache",
            "output",
            "inspection",
            "checkpoint",
            "log_mirror"
        ]

        for key in core_keys:
            self._resolve_single_paths(paths_cfg, key, config_dir)

        # --- Resolve CSV File Paths ---
        # Handle nested dictionary for CSV file locations
        if "csv" in paths_cfg:
            csv_dict = paths_cfg["csv"]
            for csv_key in csv_dict:
                self._resolve_single_paths(csv_dict, csv_key, config_dir)

    def _resolve_single_paths(self, target_dict: dict, key: str, root: Path) -> None:
        """
        Logic for one single path resolution
        """
        if key in target_dict:
            val = target_dict[key]

            # Check if the value is a string and starts with a dot (relative path)
            if isinstance(val, str) and val.startswith('.'):
                # Resolve path relative to the config file location
                resolved_path = (root / val).resolve()
                target_dict[key] = str(resolved_path)

    def set_value(self, key: str, value: Any) -> None:
        """
        Sets a configuration value for a specific key.

        Args:
            key (str): The configuration key to update.
            value (Any): The value to assign to the key.
        """
        self._config[key] = value

    def get_value(self, key: str, default: str = None) -> Any:
        """
        Retrieves a single configuration value by its key.

        Args:
            key (str): The name of the configuration parameter to retrieve.
            default (Any, optional): Default value if key is not found.

        Returns:
            Any: The configuration value associated with the key.
        """
        return self._config.get(key, default)

    def get(self) -> dict:
        """
        Returns the entire processed configuration dictionary.

        Returns:
            dict: The complete configuration dictionary, with all relative
                  paths resolved to absolute paths.
        """
        return self._config

    def get_series_depth(
        self,
        tfrecord_cache_dir: str,
        dicom_studies_dir: str,
        percentile: float,
        logger: logging.Logger
    ) -> int:
        """
        Determines the target depth (number of slices) for image series normalization.

        The depth is calculated as a specific percentile of the distribution of
        slice counts across all available studies. This ensures a consistent input
        shape for the model while limiting the impact of extreme outliers.

        The function implements a smart cache mechanism that persists the
        calculated depth in a JSON file. The cache is automatically invalidated
        if the number of studies changes or if any study directory has been
        modified (based on filesystem timestamps).

        Args:
            config (Dict[str, str]): Configuration dictionary containing paths
                and threshold parameters.
            logger (logging.Logger, optional): Logger for tracking progress and
                cache status. Defaults to None.

        Returns:
            int: The calculated reference depth (number of slices) based on
                the configured percentile.
        """
        if logger:
            logger.info("Starting  function calculate_series_depth")

        TFRecord_cache_dir = Path(tfrecord_cache_dir).resolve()
        dicom_studies_dir = Path(dicom_studies_dir).resolve()

        if logger:
            logger.info(f"dicom_studies_dir = {dicom_studies_dir}")

        cache_file = TFRecord_cache_dir / "cache.json"

        studies_dirs_list = [study for study in dicom_studies_dir.iterdir() if study.is_dir()]
        studies_count = len(studies_dirs_list)

        if logger:
            logger.info(f"Function calculate_series_depth: found {studies_count} studies")

        if studies_count == 0:
            return 0

        series_depth = None

        # Smart Cache Management
        if cache_file.exists():
            series_depth = self._get_depth_from_cache(cache_file, studies_dirs_list, logger)

        # Depth calculation (Only if series_depth has not been recovered from the cache)
        if series_depth is None:
            series_depth = self._calculate_series_depth(studies_dirs_list, percentile)
            self._save_depth_to_cache(cache_file, studies_count, series_depth, logger)

        if logger:
            logger.info(
                "Function calculate_series_depth : "
                f"calculated series depth = {series_depth}"
            )

        return series_depth

    def _get_depth_from_cache(
        self,
        file_path: Path,
        studies_list: List[Path],
        logger: logging.Logger
    ) -> int:
        """
        Retrieves the series depth from the JSON cache if it is still valid.

        The cache is considered valid only if:
        1. The number of studies in the current directory matches the
           'studies_count' stored in the cache file.
        2. No study directory has a modification timestamp (mtime) newer than
           the cache file itself, ensuring data consistency.

        Args:
            file_path (Path): Path to the 'cache.json' file.
            studies_list (List[Path]): Current list of study directory paths
                used to verify cache freshness.
            logger (logging.Logger): Logger instance for reporting cache
                status or read errors.

        Returns:
            int: The cached series depth if the cache is valid.
                 Returns None if the cache is missing, stale, or corrupted,
                 triggering a recalculation.
        """

        studies_count = len(studies_list)
        try:
            cache_mtime = file_path.stat().st_mtime
            with file_path.open('r') as f:
                cache_data = json.load(f)

                # Check 1: Invalidate cache if the number of files has changed
                if cache_data.get('studies_count') == studies_count:

                    # Check 2: Has any study directory been modified since cache creation?
                    # Note: Modifying / adding files inside a folder update sits time
                    is_cache_stale = any(
                        s.stat().st_mtime > cache_mtime for s in studies_list
                    )

                    if not is_cache_stale:
                        series_depth = int(cache_data['series_depth'])
                        if logger:
                            logger.info("Series depth loaded from cache")
                        return series_depth

        except Exception as e:
            if logger:
                warning_msg = f"Cache read failed, recalculating: {e}"
                logger.warning(warning_msg)
            pass

    def _calculate_series_depth(
        self,
        studies_list: List[Path],
        percentile: float
    ) -> int:

        """
        Calculates the reference slice depth based on a distribution of DICOM counts.

        This method iterates through all provided study directories, counts the
        number of DICOM files (*.dcm) within each series sub-directory, and
        aggregates these counts. It then determines a single target depth by
        applying a percentile calculation (e.g., 95th percentile) to the
        collected distribution.

        Args:
            studies_list (List[Path]): A list of paths pointing to individual
                study directories.
            percentile (float): The percentile (0-100) used to determine the
                target depth from the distribution of slice counts.

        Returns:
            int: The calculated target depth, representing the number of slices.
        """
        series_depth = 0

        depth_list = []

        for study in studies_list:
            # Find the maximum depth across series in this study, defaults to 0
            depth_list.extend(
                [
                    len(list(series.glob('*.dcm')))
                    for series in study.iterdir()
                    if series.is_dir()
                ]
            )

        # Handle potential empty depth_list to avoid errors in np.percentile
        if not depth_list:
            return 0

        series_depth = int(np.percentile(depth_list, percentile))
        return series_depth

    def _save_depth_to_cache(
        self,
        cache_file: Path,
        studies_count: int,
        series_depth: int,
        logger: logging.Logger
    ) -> None:

        """
        Persists the calculated series depth and study count to a JSON cache file.

        This helper method ensures the destination directory exists, then writes
        the provided metadata to a JSON file. If the operation fails (e.g., due
        to permission issues or disk full), a warning is logged but no
        exception is raised to avoid interrupting the main execution flow.

        Args:
            cache_file (Path): The file system path where the cache should be saved.
            studies_count (int): The total number of study directories found during calculation.
            series_depth (int): The calculated reference depth (percentile of slices).
            logger (logging.Logger): Logger instance for reporting errors or
                status.

        Returns:
            None
        """

        cache_dir = cache_file.parent
        # Save to Cache
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with cache_file.open('w') as f:
                json.dump(
                    {'studies_count': studies_count, 'series_depth': series_depth},
                    f
                )

        except Exception as e:
            warning_msg = f"Unable to save the cache file in {cache_dir} : {e}"
            if logger:
                logger.warning(warning_msg)

    def _check_config_compliance(self) -> None:
        """
        Validates the integrity, structure, and data types of the YAML configuration.

        This method performs an exhaustive check of the configuration dictionary against
        a predefined schema (REQUIRED_SCHEMA). It enforces a 'fail-fast' policy by
        verifying:
        1. Key existence and nesting depth.
        2. Data type compliance (including multi-type support via OneOf).
        3. Structural integrity of sequences (via Sequence).
        4. Categorical value validity (enumerations).
        5. Logical business rules (e.g., scaling ranges, positive constraints).

        The process is terminated immediately upon the first violation to prevent
        downstream failures in the data pipeline or training loop.

        Raises:
            ValueError: If a mandatory key is missing, a value is logically
                inconsistent, or an enumeration constraint is violated.
            TypeError: If a configuration value or sequence does not match
                the expected Python type(s).

        Note:
            This is an internal validation step typically called during
            the initialization of the ConfigLoader.
        """

        self._recursive_validate(self._config, REQUIRED_SCHEMA, "root")
        self._validate_business_rules()

    def _recursive_validate(
        self,
        data: Any,
        schema: Any,
        path: str
    ) -> None:
        """
        Deep validation engine for configuration integrity.
        """
        for key, expected in schema.items():
            current_path = f"{path} -> {key}"

            # 1. Existence check
            if key not in data:
                raise ValueError(f"Missing mandatory key: '{current_path}'")

            value = data[key]

            match expected:
                case dict():
                    # 2. Section check (Recursion)
                    if not isinstance(value, dict):
                        raise TypeError(f"Section '{current_path}' must be a dictionary.")
                    self._recursive_validate(value, expected, current_path)

                case set():
                    # 3. Enumeration check (Set)
                    if value not in expected:
                        raise ValueError(
                            f"Invalid value for '{current_path}': "
                            f"expected one of {expected}, got '{value}'"
                        )

                case OneOf() as marker:
                    # 4. Scalar Type choice (OneOf)
                    if not isinstance(value, marker.types):
                        raise TypeError(
                            f"Type mismatch at '{current_path}': "
                            f"expected one of {marker.types}, got {type(value).__name__}"
                        )

                case Sequence() as marker:
                    # 5. Fixed-sequence check (Sequence)
                    if not isinstance(value, (list, tuple)) or len(value) != len(marker.structure):
                        raise TypeError(
                            f"'{current_path}' must be a sequence of length {len(marker.structure)}"
                        )
                    for i, (item, target_type) in enumerate(zip(value, marker.structure)):
                        if not isinstance(item, target_type):
                            raise TypeError(
                                f"Element {i} in '{current_path}' must be "
                                f"{target_type.__name__}, got {type(item).__name__}"
                            )

                case _:
                    # 6. Strict Single Type check
                    if not isinstance(value, expected):
                        raise TypeError(
                            f"Type mismatch at '{current_path}': "
                            f"expected {expected.__name__}, got {type(value).__name__}"
                        )

    def _validate_business_rules(self) -> None:
        """
        Specific logical constraints that go beyond type checking.
        """
        cfg = self._config

        # Scaling logic
        scaling = cfg['models']['backbone_2d']['scaling']
        if scaling['min'] >= scaling['max']:
            raise ValueError(
                f"Configuration error: scaling 'min' ({scaling['min']}) "
                f"must be strictly less than 'max' ({scaling['max']})."
            )

        # Batch size logic
        if cfg['training']['batch_size'] <= 0:
            raise ValueError("Configuration error: 'batch_size' must be strictly positive.")
