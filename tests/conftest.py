# coding: utf-8

import pytest
import logging
from pathlib import Path
from unittest.mock import MagicMock
import tensorflow as tf
import pandas as pd


@pytest.fixture
def mock_config():
    """Fixture for a mocked configuration dictionary."""
    return {
        "output_dir": "tests/tmp",
        "tfrecord_dir": "tests/fixtures/tfrecords_samples",
        "batch_size": 8,
        "epochs": 2,
        "model_3d": {"type": "mock_model"},
        "csv_files": {
            "series_description": "tests/fixtures/csv_samples/mock_train_series_description.csv",
            "label_coordinates": "tests/fixtures/csv_samples/mock_train_label_coordinates.csv",
            "train": "tests/fixtures/csv_samples/mock_train.csv"
        },
        "dicom_root_dir": "tests/fixtures/dicom_samples"
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


@pytest.fixture(scope="session")
def dicom_samples_root(request):
    """
    Returns the absolute, cross-platform path to the test DICOM samples directory.
    Uses __file__ to locate the fixture directory relative to conftest.py, 
    supporting the 'session' scope.
    """
    # 1. Get the path of the conftest.py file itself
    # __file__ is always the path of the current module
    conftest_path = Path(__file__).resolve()
    
    # 2. Get the root directory of the tests (the parent of conftest.py)
    tests_root = conftest_path.parent
    
    # 3. Construct the path to the fixture directory: tests/fixtures/dicom_samples
    dicom_path = tests_root / "fixtures" / "dicom_samples"
    
    if not dicom_path.is_dir():
        # Skip the test if the required fixture data is missing
        pytest.skip(f"DICOM fixture directory not found at: {dicom_path}")
        
    # Return the absolute path as a string (using str() is common for TF compatibility)
    return str(dicom_path.resolve())


@pytest.fixture
def mock_csv_metadata():
    """Fixture for a mocked CSVMetadata class."""
    mock_csv_metadata = MagicMock()
    return mock_csv_metadata

@pytest.fixture
def mock_convert_dicom():
    """Fixture for a mocked _convert_dicom_to_tfrecords method."""
    mock_convert_dicom = MagicMock()
    return mock_convert_dicom
