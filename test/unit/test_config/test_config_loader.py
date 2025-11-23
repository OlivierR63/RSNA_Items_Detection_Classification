import pytest
from pathlib import Path
import yaml
from src.config.config_loader import ConfigLoader


@pytest.fixture
def valid_config_path(tmp_path):
    """Create a temporary valid YAML config file."""
    config = {
        "dicom_root_dir": "data/dicom",
        "output_dir": "output",
        "csv_files": {
            "train": "data/train.csv",
            "test": "data/test.csv"
        }
    }
    config_file = tmp_path / "valid_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    return config_file


@pytest.fixture
def invalid_config_path(tmp_path):
    """Create a temporary invalid YAML config file (missing key)."""
    config = {
        "dicom_root_dir": "data/dicom",
        "output_dir": "output",
        "csv_files": {
            "train": "data/train.csv"
            # Missing 'test' key
        }
    }
    config_file = tmp_path / "invalid_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    return config_file


def test_load_valid_config(valid_config_path):
    """Test loading a valid YAML config file."""
    loader = ConfigLoader(str(valid_config_path))
    config = loader.get()

    # Check if paths are resolved correctly
    assert Path(config["dicom_root_dir"]).name == "dicom"
    assert Path(config["output_dir"]).name == "output"
    assert Path(config["csv_files"]["train"]).name == "train.csv"
    assert Path(config["csv_files"]["test"]).name == "test.csv"


def test_get_value(valid_config_path):
    """Test retrieving a single value from the config."""

    loader = ConfigLoader(str(valid_config_path))
    path = loader.get_value("dicom_root_dir")

    # Convert the path to a Path object to normalize separators
    path_obj = Path(path)

    # Check that the last two components are "data" and "dicom"
    assert path_obj.parts[-2:] == ("data", "dicom")

    # Check that a non-existent key returns the default value
    assert loader.get_value("nonexistent_key", "default") == "default"


def test_get_all_config(valid_config_path):
    """
        Test retrieving the entire config.
    """
    loader = ConfigLoader(str(valid_config_path))
    config = loader.get()

    assert "dicom_root_dir" in config
    assert "csv_files" in config


def test_missing_config_file():
    """Test loading a non-existent config file."""
    with pytest.raises(FileNotFoundError):
        ConfigLoader("nonexistent_path.yaml")


def test_invalid_yaml(tmp_path):
    """Test loading a malformed YAML file."""
    config_file = tmp_path / "invalid.yaml"
    with open(config_file, 'w') as f:
        f.write("invalid: yaml: file")  # Not valid YAML
    with pytest.raises(ValueError):
        ConfigLoader(str(config_file))


def test_missing_key_behavior(valid_config_path):
    """Test behavior when a key is missing in the config."""
    loader = ConfigLoader(str(valid_config_path))
    assert loader.get_value(key="nonexistent_key") is None
    assert loader.get_value("nonexistent_key", "default") == "default"
