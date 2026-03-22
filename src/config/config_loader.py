# coding: utf-8

from pathlib import Path
from typing import Any, Dict
import yaml
import logging
import json
import numpy as np


class ConfigLoader:
    """
    Loads configuration settings from a YAML file and resolves relative paths.

    This class handles the parsing of the configuration file and ensures that
    all relevant directory and file paths (e.g., DICOM and CSV files)
    are made absolute relative to the location of the configuration file itself.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the loader by reading the YAML file and resolving paths.

        Args:
            config_path (str): File system path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the YAML file is malformed.
        """

        # Check file existence
        config_file_path = Path(config_path).resolve()
        if not config_file_path.exists():
            raise FileNotFoundError(f"Configuration file {config_path} not found.")

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

        # Reference to the paths dictionary to avoid repetitive lookups
        paths_cfg = self._config.get('paths', {})

        # --- Resolve Core Relative Paths ---
        # Iterate through known keys to convert relative paths (starting with '.') to absolute
        core_keys = [
            "dicom_studies",
            "tfrecord",
            "output",
            "inspection",
            "checkpoint",
            "log_mirror"
        ]

        for key in core_keys:
            if key in paths_cfg:
                val = paths_cfg[key]
                # Check if the value is a string and starts with a dot (relative path)
                if isinstance(val, str) and val.startswith('.'):
                    # Resolve path relative to the config file location
                    resolved_path = (config_dir / val).resolve()
                    paths_cfg[key] = str(resolved_path)

        # --- Resolve CSV File Paths ---
        # Handle nested dictionary for CSV file locations
        if "csv" in paths_cfg:
            csv_dict = paths_cfg["csv"]
            for csv_key in csv_dict:
                val = csv_dict[csv_key]
                if isinstance(val, str) and val.startswith('.'):
                    resolved_csv = (config_dir / val).resolve()
                    csv_dict[csv_key] = str(resolved_csv)

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

    def calculate_series_depth(
        self,
        tfrecord_dir: str,
        dicom_studies_dir: str,
        percentile: int,
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

        TFRecord_dir = Path(tfrecord_dir).resolve()
        dicom_studies_dir = Path(dicom_studies_dir).resolve()

        if logger:
            logger.info(f"dicom_studies_dir = {dicom_studies_dir}")

        depth_cache_file = TFRecord_dir / "depth_metadata_cache.json"

        studies_dirs_list = [study for study in dicom_studies_dir.iterdir() if study.is_dir()]
        studies_count = len(studies_dirs_list)

        if logger:
            logger.info(f"Function calculate_series_depth: found {studies_count} studies")

        if studies_count == 0:
            return 0

        series_depth = None

        # Smart Cache Management
        if depth_cache_file.exists():
            try:
                cache_mtime = depth_cache_file.stat().st_mtime
                with depth_cache_file.open('r') as f:
                    depth_cache_data = json.load(f)

                    # Check 1: Invalidate cache if the number of files has changed
                    if depth_cache_data.get('studies_count') == studies_count:

                        # Check 2: Has any study directory been modified since cache creation?
                        # Note: Modifying / adding files inside a folder update sits time
                        is_cache_stale = any(
                            s.stat().st_mtime > cache_mtime for s in studies_dirs_list
                        )

                        if not is_cache_stale:
                            series_depth = depth_cache_data['series_depth']
                            if logger:
                                logger.info("Series depth loaded from cache")

            except Exception as e:
                if logger:
                    warning_msg = f"Cache read failed, recalculating: {e}"
                    logger.warning(warning_msg)
                pass

        # Depth calculation (Only if series_depth has not been recovered from the cache)
        if series_depth is None:
            series_depth = 0

            depth_list = []

            for study in studies_dirs_list:
                # Find the maximum depth across series in this study, defaults to 0
                depth_list.extend(
                    [
                        len(list(series.glob('*.dcm')))
                        for series in study.iterdir()
                        if series.is_dir()
                    ]
                )

            series_depth = int(np.percentile(depth_list, percentile))

            # Save to Cache
            try:
                TFRecord_dir.mkdir(parents=True, exist_ok=True)
                with depth_cache_file.open('w') as cache_file:
                    json.dump(
                        {'studies_count': studies_count, 'series_depth': series_depth},
                        cache_file
                    )

            except Exception as e:
                warning_msg = f"Unable to save the cache file in {TFRecord_dir} : {e}"
                if logger:
                    logger.warning(warning_msg)

        if logger:
            logger.info(
                "Function calculate_series_depth : "
                f"calculated series depth = {series_depth}"
            )

        return series_depth
