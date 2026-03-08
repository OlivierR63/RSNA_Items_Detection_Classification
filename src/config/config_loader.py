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

        # Reference to the paths dictionary to avoid repetitive lookups
        paths_config = self._config.get('paths', {})

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
            if key in paths_config:
                val = paths_config[key]
                # Check if the value is a string and starts with a dot (relative path)
                if isinstance(val, str) and val.startswith('.'):
                    # Resolve path relative to the config file location
                    resolved_path = (config_dir / val).resolve()
                    paths_config[key] = str(resolved_path)
                    print(f"DEBUG: Resolved {key} to {paths_config[key]}")

        # --- Resolve CSV File Paths ---
        # Handle nested dictionary for CSV file locations
        if "csv" in paths_config:
            csv_dict = paths_config["csv"]
            for csv_key in csv_dict:
                val = csv_dict[csv_key]
                if isinstance(val, str) and val.startswith('.'):
                    resolved_csv = (config_dir / val).resolve()
                    csv_dict[csv_key] = str(resolved_csv)
                    print(f"DEBUG: Resolved CSV {csv_key} to {csv_dict[csv_key]}")

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
