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
    Integration test to ensure that DICOM data is correctly
    serialized into bytes and sent to the writer.
    """
    mock_config, mock_logger, _, _, metadata_df = mock_setup
    file_manager = TFRecordFilesManager(mock_config, logger=mock_logger)

    # 1. Setup a dummy DICOM path and stats

    tmp_dir = tmp_path/"4003253/1234567"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    series_id = int(tmp_dir.name)
    study_id = int(tmp_dir.parent.name)

    # Drop all useless rows in the dataframe
    mask = (metadata_df['study_id'] == study_id) & (metadata_df['series_id'] == series_id)
    metadata_df = metadata_df.loc[mask].drop_duplicates()

    series_description = int(metadata_df['series_description'].values[0])
    instance_nb = int(metadata_df['instance_number'].values[0])

    dicom_path = tmp_dir / f"{instance_nb}.dcm"
    dicom_path.write_text("pseudo-binary-data")  # Simplified for the example

    series_min, series_max = 0, 255

    input_features_df = metadata_df[
        [
            'study_id',
            'series_id',
            'series_description',
            'instance_number',
            'actual_file_format'
        ]
    ].drop_duplicates()

    labels_df = metadata_df[
        [
            'condition_level',
            'severity',
            'x',
            'y'
        ]
    ].drop_duplicates()

    mock_image = MagicMock(spec=sitk.Image)
    mock_image.GetHeight.return_value = 512
    mock_image.GetWidth.return_value = 512

    # 4. EXECUTION with real Writer
    # We use the context manager to ensure the file is flushed and closed
    tfrecord_output_path = tmp_path/"output"
    tfrecord_output_path.mkdir(parents=True, exist_ok=True)

    tfrecord_file = (tfrecord_output_path/f"{study_id}.tfrecord").resolve()

    with tf.io.TFRecordWriter(str(tfrecord_file)) as writer:

        # Patch SimpleITK ReadImage to avoid "Unable to determine ImageIO reader"
        with patch("SimpleITK.ReadImage", return_value=mock_image):

            # Patch GetArrayFromImage if your function uses it to get pixel data
            with patch(
                "SimpleITK.GetArrayFromImage",
                return_value=np.zeros((1, 512, 512), dtype=np.int16)
            ):

                # 4. EXECUTION
                # We call the internal method that actually does the heavy lifting
                test_ok = file_manager._process_single_dicom_instance(
                    series_path=dicom_path.parent,
                    series_min=series_min,
                    series_max=series_max,
                    input_features_df=input_features_df,
                    labels_df=labels_df,
                    instance_num=dicom_path.stem,
                    writer=writer,
                    is_padding=False
                )

    # 5. VERIFICATIONS
    # 5.1 Check function return
    assert test_ok is True

    # 5.2. Check physical file existence and size
    assert tfrecord_file.exists()
    assert tfrecord_file.stat().st_size > 0

    # 5.3. Verify content by reading the file back
    raw_dataset = tf.data.TFRecordDataset(str(tfrecord_file))

    # Extract the first (and only) record
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        # Validate core features are present
        features = example.features.feature
        assert "image" in features
        assert "records" in features
        assert "study_id" in features
        assert "series_id" in features
        assert "series_description" in features
        assert "instance_number" in features
        assert features["study_id"].int64_list.value[0] == study_id
        assert features["series_id"].int64_list.value[0] == series_id
        assert features["series_description"].int64_list.value[0] == series_description
        assert features["instance_number"].int64_list.value[0] == instance_nb

        # Final check on our reindexed vector (25*4 = 100)
        records = features["records"].float_list.value

        assert len(records) == 100

        # Comprehensive check for the whole vector structure
        for idx in range(25):
            start = idx * 4

            if idx == 3:  # Our Level 3
                assert records[start:start+4] == [3.0, 2.0, 350.0, 325.0]

            else:
                # Check if empty levels still carry their ID at the first index
                # Adjust to 0.0 if empty levels are completely zeroed out
                expected_level_id = float(idx)
                assert records[start] == expected_level_id
                assert all(v == 0 for v in records[start+1:start+4])
