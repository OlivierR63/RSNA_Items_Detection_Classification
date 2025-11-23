# coding: utf-8

from pathlib import Path
from typing import Any, Dict
import yaml


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
            with open(config_path, 'r') as f:
                self._config: Dict[str, Any] = yaml.safe_load(f) or {}  # Handles empty YAML files

        except yaml.YAMLError as e:
            raise ValueError(f"Error loading YAML configuration file: {e}")

        # Store the determined root directory (the YAML file's location)
        self._config["root_dir"] = str(config_dir)

        # --- Resolve Core Relative Paths ---
        # The paths 'dicom_root_dir' and 'output_dir' are resolved relative
        # to the configuration file's location.
        for key in ["dicom_study_dir", "output_dir"]:
            if key in self._config:
                # Resolve the path and convert the resulting Path object back to a string.
                self._config[key] = str(config_dir / self._config[key])

        # --- Resolve CSV File Paths ---
        # All paths within the 'csv_files' dictionary are also resolved relative
        # to the configuration file's location.
        if "csv_files" in self._config:
            for csv_key in self._config["csv_files"]:
                self._config["csv_files"][csv_key] = str(
                    config_dir / self._config["csv_files"][csv_key]
                )

    def get_value(self, key: str, default: str = None) -> Any:
        """
        Retrieves a single configuration value by its key.

        Args:
            key (str): The name of the configuration parameter to retrieve.

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
