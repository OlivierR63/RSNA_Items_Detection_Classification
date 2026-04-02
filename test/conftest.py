# coding: utf-8

import pytest
import logging
import pandas as pd
import tensorflow as tf
from unittest.mock import MagicMock
from src.projects.lumbar_spine.tfrecord_files_manager import TFRecordFilesManager


@pytest.fixture
def mock_config(tmp_path, setup_csv_files):
    """
    Configuration fixture updated to use paths from setup_csv_files.
    """
    return {
        "root_dir": tmp_path,
        "paths": {
            "dicom_studies": tmp_path / "dicom",
            "tfrecord": tmp_path / "tfrecord_dir",
            "output": tmp_path / "output",
            "tf_cache": tmp_path / "tensorflow_cache/lumbar_train",
            "checkpoint": tmp_path / "checkpoints/model.keras",
            "inspection": tmp_path / "input_data_inspection",
            "log_mirror": tmp_path / "logs/full_session_output.log",
            "csv": {
                "series_description": tmp_path / "csv/train_series_description.csv",
                "label_coordinates": tmp_path / "csv/train_label_coordinates.csv",
                "label_enriched": tmp_path / "csv/label_enriched.csv",
                "train": tmp_path / "csv/train.csv"
            }
        },

        "data_specs": {
            "series_depth_percentile": 95,
            "max_records_per_frame": 25,
            "dataset_buffer_size_mb": 100,
        },

        "models": {
            "backbone_2d": {
                "type": "MobileNetV2",
                "img_shape": [224, 224, 3],
                "freeze": True,
                "scaling": {"min": -1.0, "max": 1.0}
            },

            "head_3d": {
                "type": "cnn3d",
                "filters": 64
            }
        },

        "training": {
            "batch_size": 2,
            "epochs": 10,
            "train_split_ratio": 0.8
        },

        "optimizer": {
            "type": "adam",
            "learning_rate": 0.0001,
            "clipnorm": 1.0
        },

        "callbacks": {
            "patience": 5,
            "resume_mode": "last"
        },

        "compilation": {
            "loss_weights": {
                "severity_output": 1.0,
                "location_output": 30.0
            },
            "run_eagerly": True
        },

        "system": {
            "nb_cores": 7,
            "log_retention_days": 30,
            "memory_threshold_percent": 95,
            "threshold_temperature": 85,
            "seed": 42
        },

        "series_depth": 5,

        "dataset_steering": {
            "interleave": {
                "parallel_files": 1,
                "block_per_file": 1,
                "deterministic": False
            },
            "group_studies": 1,
            "prefetch_batches": 1,
            "num_parallel_calls": 1,
            "use_cache": False
        },

        "logging": {
            "level": "DEBUG",
            "console_display": True,
            "use_json": False
        }
    }


@pytest.fixture
def setup_csv_files(tmp_path):
    """
    Creates real CSV files in a temporary directory for testing.
    Internal formatting follows standard CSV requirements (no extra quotes).
    """
    # Define paths within the tmp_path
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "series_description": csv_dir / "train_series_description.csv",
        "label_coordinates": csv_dir / "label_coordinates.csv",
        "label_enriched": csv_dir / "label_coordinates_format.csv",
        "train": csv_dir / "train.csv"
    }

    # Write series_description.csv - Raw values without extra single quotes
    paths["series_description"].write_text(
        "study_id,series_id,series_description\n"
        "1,1,description_1\n"
        "2,2,description_2"
    )

    # Write label_coordinates.csv
    paths["label_coordinates"].write_text(
        "study_id,series_id,instance_number,condition,level,x,y,actual_file_format\n"
        "1,1,1,condition_1,level_1,10.0,20.0,\"(224,224)\"\n"
        "2,2,2,condition_2,level_2,30.0,40.0,\"(224,224)\""
    )

    # Write label_coordinates_format.csv
    # Note: actual_file_format is stored as a string representing a tuple
    paths["label_enriched"].write_text(
        "study_id,series_id,instance_number,condition,level,x,y,actual_file_format\n"
        "1,1,1,condition_1,level_1,10.0,20.0,\"(640, 640)\"\n"
        "2,2,2,condition_2,level_2,30.0,40.0,\"(320, 320)\""
    )

    # Write train.csv
    paths["train"].write_text(
        "study_id,condition_1_level_1,condition_2_level_2\n"
        "1,severity_1,severity_2\n"
        "2,severity_3,severity_2"
    )

    # Return as strings for compatibility with the handler's __init__
    return {key: str(val) for key, val in paths.items()}


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
    # Without this, low-level messages used for debugging logic
    # might be filtered out before capture.
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
    base_dir = tmp_path / "dicom"

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

            # Generate a fixed number of DICOM files per study (3, 4, or 5)
            for idx in range(1, nb_files + 1):
                instance_file = series_path / f"{idx}.dcm"

                # Write non-empty binary content
                # We use a dummy DICOM-like header prefix for realism
                if not instance_file.exists():
                    instance_file.write_bytes(b"\x00" * 128 + b"DICOM" + b"\xff\xfe")

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
        "condition_level": [],
        "series_description": [],
        "severity": [],
        "x": [],
        "y": [],
        "actual_file_format": []
    }

    # Align metadata with the setup_dicom_tree_structure logic
    configs = [
        (
            1010, [10110, 10120, 10130], 3,
            "Sagittal T1", "Spinal Canal Stenosis_L5/S1", "Moderate",
            100, 200, (224, 224, 3)
        ),

        (
            1020,
            [10210, 10220, 10230], 4,
            "Sagittal T2", "Right Subarticular Stenosis_L1/L2", "Normal/Mild",
            125, 275, (224, 224, 3)
        ),

        (
            1030, [10310, 10320, 10330], 5,
            "Axial T2", "Left Neural Foraminal Narrowing_L3/L4", "Severe",
            300, 500, (224, 224, 3)
        )
    ]

    for config in configs:
        (
            study_id,
            series_list,
            nb_files,
            description,
            condition_level,
            severity,
            x_loc,
            y_loc,
            file_format
        ) = config

        for series_id in series_list:
            for inst_num in range(1, nb_files + 1):
                data["study_id"].append(study_id)
                data["series_id"].append(series_id)
                data["instance_number"].append(inst_num)
                data["condition_level"].append(condition_level)
                data["series_description"].append(description)
                data["severity"].append(severity)
                data["x"].append(x_loc)
                data["y"].append(y_loc)
                data["actual_file_format"].append(file_format)

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
        "series_description": [],
        "severity": [],
        "x": [],
        "y": [],
        "actual_file_format": []
    }

    # Align metadata with the setup_dicom_tree_structure logic
    configs = [
        (1010, [10110, 10120, 10130], 3, 0, 4, 1, 100, 200, (224, 224, 3)),
        (1020, [10210, 10220, 10230], 4, 1, 15, 0, 125, 275, (224, 224, 3)),
        (1030, [10310, 10320, 10330], 5, 2, 12, 2, 300, 500, (224, 224, 3))
    ]

    for config in configs:
        (
            study_id,
            series_list,
            nb_files,
            description,
            condition_level,
            severity,
            x_loc,
            y_loc,
            file_format
        ) = config

        for series_id in series_list:
            for inst_num in range(1, nb_files + 1):
                data["study_id"].append(study_id)
                data["series_id"].append(series_id)
                data["instance_number"].append(inst_num)
                data["condition_level"].append(condition_level)
                data["series_description"].append(description)
                data["severity"].append(severity)
                data["x"].append(x_loc)
                data["y"].append(y_loc)
                data["actual_file_format"].append(file_format)

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
