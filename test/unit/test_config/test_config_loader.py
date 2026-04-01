# coding: utf-8

import pytest
import yaml
import logging
import time
from pathlib import Path
from src.config.config_loader import ConfigLoader
from unittest.mock import patch


# Local helper to ensure no Path objects are passed to yaml.dump
def stringify_paths(data: dict) -> dict:
    """
    Recursively converts all pathlib.Path objects in a dictionary to strings.
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, dict):
            result[key] = stringify_paths(value)
        else:
            result[key] = value
    return result


class TestConfigLoader:
    """
    Comprehensive unit tests for ConfigLoader.
    Covers initialization, path resolution, and robust cache management
    including edge cases and exception handling.
    """

    @pytest.fixture
    def yaml_config_path(self, tmp_path, mock_config):
        """
        Helper to create a real YAML file from the mock_config fixture.
        """
        def path_to_str(obj):
            if isinstance(obj, dict):
                return {k: path_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(path_to_str(mock_config), f)
        return config_file

    # --- SECTION 1: Initialization & YAML Exceptions ---

    def test_init_raises_file_not_found_error(self):
        """
        Checks if FileNotFoundError is raised for missing config file.
        """
        with pytest.raises(FileNotFoundError):
            ConfigLoader("non_existent_path.yaml")

    def test_init_raises_value_error_on_malformed_yaml(self, tmp_path):
        """
        Tests the catch of yaml.YAMLError and re-raising as ValueError.
        """
        bad_yaml = tmp_path / "malformed.yaml"

        # Invalid YAML syntax (duplicated mapping key with illegal structure)
        bad_yaml.write_text("paths:\n  dicom_studies: : invalid")

        with pytest.raises(ValueError, match="Error loading YAML configuration file"):
            ConfigLoader(str(bad_yaml))

    # --- SECTION 2: Path Resolution (Blocks 2 & 3) ---

    def test_init_resolves_relative_paths(self, mock_config, tmp_path):
        """
        Ensure dots (./) in core and CSV paths are resolved to absolute paths.
        """
        # 1. Setup the project structure within the temporary directory
        # project_dir will host the config.yaml file
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Define expected absolute paths for validation
        expected_dicom_path = project_dir / "data" / "dicom"
        expected_csv_path = project_dir / "data" / "train.csv"

        # 3. Modify mock_config to use RELATIVE paths (starting with ./)
        # These paths are relative to where the config.yaml will be saved
        custom_config = mock_config.copy()
        custom_config["paths"]["dicom_studies"] = "./data/dicom"
        custom_config["paths"]["csv"]["train"] = "./data/train.csv"

        # 4. Write the YAML configuration file to disk
        config_file = project_dir / "config.yaml"
        config_file.write_text(yaml.dump(stringify_paths(custom_config)))

        # 5. Initialize the ConfigLoader
        # The loader should detect its own location and resolve the relative strings
        loader = ConfigLoader(str(config_file))
        paths_cfg = loader.get()["paths"]

        # 6. Assertions: Verify that relative strings were transformed into absolute paths
        # We compare the output against the resolved Path objects created in step 1
        assert paths_cfg["dicom_studies"] == str(expected_dicom_path.resolve())
        assert paths_cfg["csv"]["train"] == str(expected_csv_path.resolve())

    # --- SECTION 3: Cache Management & Logical Blocks ---

    def test_calculate_depth_returns_zero_on_empty_dir(self, tmp_path, mock_config, mock_logger):
        """
        Returns 0 if no studies are found in the DICOM directory.
        """
        # 1. Setup empty directories
        empty_dir = tmp_path / "empty_dicom"
        empty_dir.mkdir()
        tfr_dir = tmp_path / "tfrecord"
        tfr_dir.mkdir()

        # 2. Create a valid dummy config file
        # We use mock_config to ensure all required fields (logging, etc.) are present
        dummy_config_file = tmp_path / "dummy.yaml"

        # Ensure the config is compliant by using the full mock_config fixture
        clean_config = stringify_paths(mock_config)
        dummy_config_file.write_text(yaml.dump(clean_config))

        # 3. Initialize the loader
        loader = ConfigLoader(str(dummy_config_file))

        # 4. Execute and Assert
        depth = loader.calculate_series_depth(
            tfrecord_dir=str(tfr_dir),
            dicom_studies_dir=str(empty_dir),
            percentile=95,
            logger=mock_logger
        )

        assert depth == 0

    def test_calculate_series_depth_handles_corrupt_cache_file(
        self,
        yaml_config_path,
        setup_dicom_tree_structure,
        mock_logger,
        caplog
    ):
        """
        Tests handling of a corrupt JSON cache file during read.
        """
        # 1. Initialize the loader using the provided YAML fixture
        loader = ConfigLoader(str(yaml_config_path))
        paths = loader.get()["paths"]

        # 2. Ensure the TFRecord directory exists for cache placement
        tfr_dir = Path(paths["tfrecord"])
        tfr_dir.mkdir(parents=True, exist_ok=True)

        # 3. Create a corrupt cache file with invalid JSON content
        # This simulates a file that exists but cannot be parsed by the json module
        cache_file = tfr_dir / "depth_metadata_cache.json"
        cache_file.write_text("INVALID_JSON_CONTENT")

        # 4. Execute the depth calculation while capturing logs
        # The function should catch the JSON decode error and log a warning instead of crashing
        with caplog.at_level(logging.WARNING):
            depth = loader.calculate_series_depth(
                tfrecord_dir=str(tfr_dir),
                dicom_studies_dir=paths["dicom_studies"],
                percentile=95,
                logger=mock_logger
            )

        # 5. Assertions
        # The calculation should proceed as if no cache existed (returning 5
        # from setup_dicom_tree_structure)
        assert depth == 5

        # Verify that the specific recovery message was logged during the execution
        assert "Cache read failed, recalculating" in caplog.text

    def test_calculate_depth_cache_invalidation_mtime(
        self,
        yaml_config_path,
        setup_dicom_tree_structure,
        mock_logger
    ):
        """
        Tests cache loading and invalidation via directory modification (mtime).
        """
        # 1. Initialize loader and retrieve directory paths
        loader = ConfigLoader(str(yaml_config_path))
        paths = loader.get()["paths"]
        tfr_dir = paths["tfrecord"]
        dicom_dir = Path(paths["dicom_studies"])

        # 2. First execution: Generate and save the initial cache file
        loader.calculate_series_depth(tfr_dir, str(dicom_dir), 95, mock_logger)
        cache_file = Path(tfr_dir) / "depth_metadata_cache.json"
        mtime_initial = cache_file.stat().st_mtime

        # 3. Introduce a small delay to ensure a measurable difference in mtime
        time.sleep(0.1)

        # 4. Modify a subdirectory within the DICOM structure
        # This action changes the mtime of the parent directory,
        # which should trigger the cache invalidation logic.
        (dicom_dir / "1010" / "new_file.dcm").write_text("update")

        # 5. Second execution: Recalculate depth
        # The loader should detect that the source directory is newer than the cache
        loader.calculate_series_depth(tfr_dir, str(dicom_dir), 95, mock_logger)

        # 6. Assertion: Verify that the cache file was overwritten (new mtime)
        assert cache_file.stat().st_mtime > mtime_initial

    def test_calculate_series_depth_handles_cache_write_permission_error(
        self,
        yaml_config_path,
        setup_dicom_tree_structure,
        mock_logger,
        caplog
    ):
        """
        Tests the exception handling when the cache file cannot be saved
        (e.g., due to permission issues).
        """
        # 1. Initialize loader with the provided YAML fixture
        loader = ConfigLoader(str(yaml_config_path))
        paths = loader.get()["paths"]
        tfr_dir = Path(paths["tfrecord"])

        # 2. Ensure the directory exists before attempting to restrict it
        tfr_dir.mkdir(parents=True, exist_ok=True)

        # 3. Use a mock to simulate a PermissionError during file writing.
        # This is necessary because os.chmod does not reliably restrict
        # directory write access across different operating systems (especially Windows).
        with patch("pathlib.Path.open") as mock_open:

            # Define a side effect that raises an error ONLY when opening for writing ('w')
            def side_effect(file_instance, mode='r', *args, **kwargs):
                if 'w' in mode:
                    raise PermissionError("Mocked Access Denied Error")
                # Fallback to the real open method for other operations (like reading)
                return original_open(file_instance, mode, *args, **kwargs)

            # Store the original method to use it inside the side effect to avoid infinite recursion
            original_open = Path.open
            mock_open.side_effect = side_effect

            # 4. Capture logs at WARNING level to verify the error handling
            with caplog.at_level(logging.WARNING):
                depth = loader.calculate_series_depth(
                    tfrecord_dir=str(tfr_dir),
                    dicom_studies_dir=paths["dicom_studies"],
                    percentile=95,
                    logger=mock_logger
                )

        # 5. Assertions
        # Check if the function still returns the correct depth despite the save failure
        assert depth == 5

        # Verify that the specific warning message regarding cache save failure is present in logs
        assert "Unable to save the cache file" in caplog.text

    # --- SECTION 4: General Utilities ---

    def test_get_and_set_value(self, yaml_config_path):
        """
        Verifies standard getter and setter methods of the ConfigLoader.
        """
        # 1. Initialize the loader with a valid configuration file
        loader = ConfigLoader(str(yaml_config_path))

        # 2. Test the setter: Add a new dynamic key-value pair
        loader.set_value("dynamic_key", "value")

        # 3. Test the getter: Retrieve the previously set value
        assert loader.get_value("dynamic_key") == "value"

        # 4. Test the getter with a missing key:
        # It should return the provided default value ("default")
        assert loader.get_value("non_existent", "default") == "default"
