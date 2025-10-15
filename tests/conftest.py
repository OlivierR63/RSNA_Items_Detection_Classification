# tests/conftest.py
import pytest
import logging
from pathlib import Path
from unittest.mock import MagicMock
import tensorflow as tf
import pandas as pd

@pytest.fixture
def mock_config():
    """Fixture pour un config mocké."""
    return {
        "output_dir": "tests/tmp",
        "batch_size": 8,
        "epochs": 2,
        "model_3d": {"type": "mock_model"},
        "csv_files": {
            "series_description": "tests/data/mock_series.csv",
            "label_coordinates": "tests/data/mock_coordinates.csv",
            "train": "tests/data/mock_train.csv"
        },
        "dicom_root_dir": "tests/data/mock_dicom"
    }

@pytest.fixture
def mock_logger():
    """Fixture pour un logger mocké."""
    logger = MagicMock(spec=logging.Logger)
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.warning = MagicMock()
    logger.debug = MagicMock()
    return logger

@pytest.fixture(autouse=True)
def setup_test_env(tmp_path):
    """Configure l'environnement de test."""
    # Crée des répertoires temporaires
    (tmp_path / "logs").mkdir()
    (tmp_path / "tfrecords").mkdir()
    yield
    # Nettoyage (optionnel, pytest le fait déjà)
