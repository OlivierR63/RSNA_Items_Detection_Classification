# coding: utf-8

import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd


@pytest.fixture
def mock_config(tmp_path):
    """Fixture for a mocked configuration dictionary."""
    return {
        "root_dir": str(Path(__file__).resolve().parent),
        "dicom_study_dir": "fixtures/dicom_unique_sample",
        "tfrecord_sample_dir": "fixtures/tfrecords",
        "tfrecord_dir": str(tmp_path / "tfrecords"),
        "csv_files": {
            "series_description": "fixtures/csv_samples/mock_train_series_descriptions.csv",
            "label_coordinates": "fixtures/csv_samples/mock_train_label_coordinates.csv",
            "train": "fixtures/csv_samples/mock_train.csv"
        },
        "output_dir": str(tmp_path),
        "model_2d": {"type": "mock_model"},
        "model_3d": {"type": "mock_model"},
        "batch_size": 8,
        "epochs": 2
    }


@pytest.fixture
def mock_logger():
    """Fixture for a mocked logger instance (MagicMock)."""
    logger = MagicMock(spec=logging.Logger)
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.warning = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest.fixture
def mock_setup(mock_config, mock_logger):
    """
        Fixture to initialize attributes common to all tests.
    """

    # Mock the get_current_logger function to return the mock_logger
    with patch("src.core.utils.logger.get_current_logger", return_value=mock_logger):
        yield mock_config, mock_logger


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path):
    """
    Configures the test environment automatically for each test.
    Creates necessary temporary directories using pytest's tmp_path fixture.
    """
    # Create temporary directories
    (tmp_path / "logs").mkdir()
    (tmp_path / "tfrecords").mkdir()
    yield
    # Cleanup is implicitly handled by tmp_path


@pytest.fixture
def mock_csv_metadata():
    """Fixture to provide a mock CSVMetadata instance."""
    mock_csv_metadata = MagicMock()
    mock_csv_metadata._merged_df = pd.DataFrame({
        "study_id": [12345678],
        "series_id": [87654321],
        "instance_number": [1],
        "condition": [2],
        "severity": [1],
        "series_description": [0],
        "level": [3],
        "x": [12.34],
        "y": [56.78]
    })
    return mock_csv_metadata


@pytest.fixture
def mock_convert_dicom():
    """Fixture for a mocked _convert_dicom_to_tfrecords method."""
    return MagicMock()
