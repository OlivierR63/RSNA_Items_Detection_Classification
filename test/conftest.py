# coding: utf-8

import pytest
import logging
import pandas as pd
import tensorflow as tf
from unittest.mock import MagicMock
from src.projects.lumbar_spine.tfrecord_files_manager import TFRecordFilesManager


@pytest.fixture
def mock_config(tmp_path):
    return{
        "tfrecord_dir":tmp_path/"tfrecord_dir",
        "max_records": 25,
        "root_dir": tmp_path/"root",
        "dicom_studies_dir": tmp_path/"root/dicom",
        "csv_files":{
            "description": tmp_path/"root/csv/description.csv",
            "label_coordinates": tmp_path/"root/csv/coordinates.csv",
            "train": tmp_path/"train.csv"
        },
        "checkpoint_path": tmp_path/"root/checkpoints/model.keras",
        "output_dir": tmp_path / "root/output",
        "model_3d": {"type": "cnn3d", "input_shape": [7, 7, 1280], "num_classes": 3},
        "batch_size": 2,
        "epochs": 1,
        "patience": 3,
        "nb_cores": 2,
        "model_2d": { 'type': 'MobileNetV2', 'img_shape': (224, 224, 3)},
        "aggregator": {'filters': 64}
    }


@pytest.fixture
def setup_csv_files(tmp_path):
    # Create test CSV files
    series_description_path = tmp_path / "description.csv"
    label_coordinates_path = tmp_path / "label_coordinates.csv"
    train_path = tmp_path / "train.csv"

    # Write series_description.csv
    series_description_path.write_text(
        "study_id,series_id,description\n"
        "'1','1','description_1'\n"
        "'2','2','description_2'"
    )

    # Write label_coordinates.csv
    label_coordinates_path.write_text(
        "study_id,series_id,instance_number,condition,level,x,y\n"
        "'1','1','1','condition_1','level_1','10','20'\n"
        "'2','2','2','condition_2','level_2','30','40'"
    )

    # Write train.csv
    train_path.write_text(
        "study_id,condition1_level1,condition2_level2\n"
        "'1','severity_1','severity_2',\n"
        "'2','severity_3','severity_2'"
    )

    return {
        "description": str(series_description_path),
        "label_coordinates": str(label_coordinates_path),
        "train": str(train_path)
    }


# Unified logger fixture renamed to mock_logger to maintain 
# compatibility with existing tests while enabling caplog features.
@pytest.fixture
def mock_logger(caplog):
    """
    Unified logger fixture.
    Works as a real logger for the manager, but allows message capture via caplog.
    """
    # Get the specific logger used in your project
    test_logger = logging.getLogger("lumbar_spine_test")
    
    # Set to DEBUG to ensure caplog catches every detail (INFO, WARNING, etc.).
    # Without this, low-level messages used for debugging logic might be filtered out before capture.
    caplog.set_level(logging.DEBUG, logger="lumbar_spine_test")
    
    return test_logger


@pytest.fixture(autouse=True)
def setup_dicom_tree_structure(tmp_path):
    """
    Generates a mock DICOM directory structure for testing.
    Root: tmp_path/root/dicom
    Studies: 1010 (3 files/series), 1020 (4 files/series), 1030 (5 files/series)
    Series: (StudyID * 10) + [10, 20, 30]
    """
    # Define the root directory for TFRecords
    base_dir = tmp_path / "root/dicom"

    # If the base directory already exists, we skip the generation
    if base_dir.exists():
        return

    base_dir.mkdir(parents=True, exist_ok=True)

    study_ids = [1010, 1020, 1030]
    series_offsets = [10, 20, 30]
    cardinal = [3, 4, 5]

    for s_id, nb_files in zip(study_ids, cardinal):
        # Create study directory
        study_path = base_dir / str(s_id)
        study_path.mkdir(exist_ok=True)

        for offset in series_offsets:
            # Calculate series ID: (1010 * 10) + 10 = 10110
            series_id = (s_id * 10) + offset
            series_path = study_path / str(series_id)
            series_path.mkdir(exist_ok=True)

            # Generate a random number of DICOM files (3 to 10)
            for idx in range(1, nb_files + 1):
                instance_file = series_path / f"{idx}.dcm"
                
                # Write non-empty binary content
                # We use a dummy DICOM-like header prefix for realism
                if not instance_file.exists():
                    instance_file.write_bytes(b"\x00" * 128 + b"DICM" + b"\xff\xfe")
    
    return base_dir


@pytest.fixture
def mock_metadata():
    """
    Creates a DataFrame matching the DICOM structure:
    1010 (3 series * 3 files), 1020 (3 series * 4 files), 1030 (3 series * 5 files)
    """
    data = {
        "study_id": [],
        "series_id": [],
        "instance_number": [],
        "condition": [],
        "description": [],
        "level": [],
        "severity": [],
        "x": [],
        "y": []
    }

    # Align metadata with the setup_dicom_tree_structure logic
    configs = [
        (1010, [10110, 10120, 10130], 3, "Sagittal T1", "Spinal Canal Stenosis", "L5/S1", "Moderate", 100, 200),
        (1020, [10210, 10220, 10230], 4, "Sagittal T2", "Right Subarticular Stenosis", "L1/L2", "Normal/Mild", 125, 275),
        (1030, [10310, 10320, 10330], 5, "Axial T2", "Left Neural Foraminal Narrowing", "L3/L4", "Severe", 300, 500)
    ]

    for study_id, series_list, nb_files, description, condition, level, severity, x_loc, y_loc in configs:
        for series_id in series_list:
            for inst_num in range(1, nb_files + 1):
                data["study_id"].append(study_id)
                data["series_id"].append(series_id)
                data["instance_number"].append(inst_num)
                data["condition"].append(condition)
                data["description"].append(description)
                data["level"].append(level)
                data["severity"].append(severity)
                data["x"].append(x_loc)
                data["y"].append(y_loc)

    # Return the DataFrame as the fixture value
    return pd.DataFrame(data)


@pytest.fixture
def mock_encoded_metadata():
    """
    Creates a DataFrame matching the DICOM structure:
    1010 (3 series * 3 files), 1020 (3 series * 4 files), 1030 (3 series * 5 files)
    """
    data = {
        "study_id": [],
        "series_id": [],
        "instance_number": [],
        "condition_level": [],
        "description": [],
        "severity": [],
        "x": [],
        "y": []
    }

    # Align metadata with the setup_dicom_tree_structure logic
    configs = [
        (1010, [10110, 10120, 10130], 3, 0, 4, 1, 100, 200),
        (1020, [10210, 10220, 10230], 4, 1, 15, 0, 125, 275),
        (1030, [10310, 10320, 10330], 5, 2, 12, 2, 300, 500)
    ]

    for study_id, series_list, nb_files, description, condition_level, severity, x_loc, y_loc in configs:
        for series_id in series_list:
            for inst_num in range(1, nb_files + 1):
                data["study_id"].append(study_id)
                data["series_id"].append(series_id)
                data["instance_number"].append(inst_num)
                data["condition_level"].append(condition_level)
                data["description"].append(description)
                data["severity"].append(severity)
                data["x"].append(x_loc)
                data["y"].append(y_loc)

    # Return the DataFrame as the fixture value
    return pd.DataFrame(data)


@pytest.fixture
def mock_writer():
    """Fixture for a mocked TFRecordWriter instance."""
    # Using MagicMock as a context manager to support 'with' statements
    writer = MagicMock(spec=tf.io.TFRecordWriter)
    writer.__enter__.return_value = writer
    return writer


# Global manager fixture accessible by all test classes
@pytest.fixture
def mock_tfrecord_files_manager(mock_config, mock_logger):
    # Initialize manager with fixtures from contest.py
    return TFRecordFilesManager(config=mock_config, logger=mock_logger)

