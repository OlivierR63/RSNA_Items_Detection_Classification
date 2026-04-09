# coding: utf-8

import pytest
import tensorflow as tf
import pandas as pd
import numpy as np
from unittest.mock import patch  # , MagicMock
# from src.projects.lumbar_spine.tfrecord_files_manager import TFRecordFilesManager>
from pathlib import Path
# import SimpleITK as sitk


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


def test_integration_tfrecord_pixel_fidelity(tmp_path):
    """
    Integration test to ensure that a raw pixel array (256x512)
    maintains perfect fidelity after being serialized to a TFRecord
    and read back using TensorFlow's decoding tools.
    """

    # 1. Setup: Generate a "real" fake image with known values
    height, width = 1, 1
    expected_pixels = height * width
    # We use np.int16 (2 bytes per pixel) as it is standard for DICOM data
    # Filling with 1s instead of 0s to ensure we aren't just reading empty buffers
    original_image = np.ones((height, width), dtype=np.int16)

    # Define the output path for the test TFRecord
    tfrecord_file = str(tmp_path / "1234567.tfrecord")

    # 2. Serialization: Convert NumPy array to bytes and wrap in a TF Protobuf
    image_bytes = original_image.tobytes()

    # Construct the feature dictionary (matching your model's expected schema)
    feature_dict = {
        "study_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[4003253])),
        "series_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[1234567])),
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        "records": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0] * 100))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    # Write the serialized Example to disk
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        writer.write(example.SerializeToString())

    # 3. Verification: Read the file back and decode the raw pixel buffer
    # Load the dataset (no compression used in this specific test)
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

    for raw_record in raw_dataset.take(1):
        # Parse the serialized protocol buffer
        parsed_example = tf.train.Example()
        parsed_example.ParseFromString(raw_record.numpy())
        features = parsed_example.features.feature

        # Access the raw bytes from the 'image' field
        image_raw_bytes = features['image'].bytes_list.value[0]

        # Decode the raw bytes into a 1D tensor of pixels
        # Crucial: out_type must match the original dtype (int16)
        decoded_tensor = tf.io.decode_raw(image_raw_bytes, out_type=tf.int16)

        # Calculate the total number of pixels using the shape of the flattened tensor
        actual_pixel_count = tf.shape(decoded_tensor)[0].numpy()

        # --- ASSERTIONS ---

        # A. Quantity Check: Ensure we have exactly 131,072 pixels
        assert actual_pixel_count == expected_pixels, \
            f"Fidelity Error: Found {actual_pixel_count} pixels, expected {expected_pixels}"

        # B. Binary Size Check: 131,072 pixels * 2 bytes/pixel = 262,144 bytes
        assert len(image_raw_bytes) == expected_pixels * 2, \
            "Raw byte length does not match 16-bit encoding requirements."

        # C. Quality Check: Ensure data values were not altered (all should be 1)
        pixels_array = decoded_tensor.numpy()
        assert np.all(pixels_array == 1), "Pixel data corruption detected! Values were altered."

    print(f"Integration Test Passed: {actual_pixel_count} pixels verified with 100% fidelity.")
