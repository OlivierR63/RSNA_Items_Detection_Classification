# coding: utf-8

import pytest
import tensorflow as tf
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.projects.lumbar_spine.tfrecord_files_manager import TFRecordFilesManager
from pathlib import Path
import SimpleITK as sitk


@pytest.fixture
def mock_setup(mock_config, mock_logger):
    """
        Fixture to initialize common attributes for all tests.
    """

    dicom_studies_dir = Path(mock_config["paths"]["dicom_studies"])
    tfrecord_dir = Path(mock_config["paths"]["tfrecord"])

    metadata_df = pd.DataFrame(
        {
            "study_id": [1, 1, 2, 4003253, 123456789],
            "series_id": [1, 2, 1, 1234567, 1],
            "series_description": [0, 1, 2, 1, 2],
            "instance_number": [1, 1, 1, 1, 1],
            "condition_level": [0, 1, 2, 3, 4],
            "severity": [0, "0", "1", 2, 0],
            "x": [200.0, 250.0, 350.0, 350.0, 400.0],
            "y": [25.0, 125.0, 225.0, 325.0, 425.0],
            "actual_file_format": [
                (224, 224, 3),
                (640, 640, 3),
                (325, 325, 3),
                (224, 224, 3),
                (224, 224, 3)
            ]
        }
    )

    # Mock the get_current_logger function to return the mock_logger
    with patch("src.core.utils.logger.get_current_logger", return_value=mock_logger):
        yield mock_config, mock_logger, dicom_studies_dir, tfrecord_dir, metadata_df


def test_integration_dicom_serialization(mock_setup, tmp_path):
    """
    Integration test to ensure that the DICOM processing pipeline
    correctly serializes data into a TFRecord file and that the data
    can be read back accurately.
    """
    mock_config, mock_logger, _, _, metadata_df = mock_setup
    file_manager = TFRecordFilesManager(mock_config, logger=mock_logger)

    # 1. Setup filesystem and IDs
    tmp_dir = tmp_path / "4003253/1234567"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    series_id = int(tmp_dir.name)
    study_id = int(tmp_dir.parent.name)

    # 2. Prepare metadata for this specific instance
    mask = (metadata_df['study_id'] == study_id) & (metadata_df['series_id'] == series_id)
    relevant_metadata = metadata_df.loc[mask].drop_duplicates()

    instance_nb = int(relevant_metadata['instance_number'].values[0])

    dicom_path = tmp_dir / f"{instance_nb}.dcm"
    dicom_path.write_text("pseudo-binary-data")

    # 3. Mock SimpleITK to return expected shapes and components
    # (Matches what _load_normalized_dicom expects)
    mock_image = MagicMock(spec=sitk.Image)
    mock_image.GetNumberOfComponentsPerPixel.return_value = 1
    mock_image.GetHeight.return_value = 512
    mock_image.GetWidth.return_value = 512

    # SimpleITK GetArrayFromImage returns (Depth, Height, Width)
    fake_array = np.zeros((1, 512, 512), dtype=np.int16)

    # 4. EXECUTION
    tfrecord_output_path = tmp_path / "output"
    tfrecord_output_path.mkdir(parents=True, exist_ok=True)
    tfrecord_file = (tfrecord_output_path / f"{study_id}.tfrecord").resolve()

    with tf.io.TFRecordWriter(str(tfrecord_file)) as writer:
        with patch("SimpleITK.ReadImage", return_value=mock_image), \
             patch("SimpleITK.GetArrayFromImage", return_value=fake_array):

            # Call the main orchestration method
            # Note: _process_single_dicom_instance internally calls our new refactored methods
            test_ok = file_manager._process_single_dicom_instance(
                series_path=dicom_path.parent,
                series_min=0,
                series_max=255,
                input_features_df=metadata_df,  # Pass the full DF or filtered one
                labels_df=metadata_df[['condition_level', 'severity', 'x', 'y']],
                instance_num=str(instance_nb),
                writer=writer,
                is_padding=False
            )

    # 5. VERIFICATIONS
    assert test_ok is True
    assert tfrecord_file.exists()
    assert tfrecord_file.stat().st_size > 0

    # 6. Content Validation (De-serialization)
    raw_dataset = tf.data.TFRecordDataset(str(tfrecord_file))

    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        # Verify Metadata Traceability
        assert features["study_id"].int64_list.value[0] == study_id
        assert features["series_id"].int64_list.value[0] == series_id
        assert features["instance_number"].int64_list.value[0] == instance_nb

        # Verify Image Presence
        assert "image" in features

        # Verify Labels Vector (25 levels * 4 features = 100 floats)
        records = features["records"].float_list.value
        assert len(records) == 100

        # Verify a specific known label (Level 3 for instance)
        # Based on your logic: [Level_ID, Severity, X, Y]
        # If Level 3 is at index 3:
        start = 3 * 4
        if records[start] == 3.0:  # Check if ID is correctly mapped
            assert records[start+1] == 2.0  # Severity
            assert records[start+2] == 350.0  # X
