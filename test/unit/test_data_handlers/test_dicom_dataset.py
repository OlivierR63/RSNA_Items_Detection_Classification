# coding: utf-8

import tensorflow as tf
import pytest
from unittest.mock import patch
from src.core.data_handlers.dicom_dataset import DicomTFDataset
from pathlib import Path

# NOTE: The base_dir should be replaced with a path accessible
#       in the testing environment (e.g., relative path)
# Using 'pathlib' or 'os.path.join' is generally safer for cross-platform compatibility.
# For this example, we keep the original structure for reference but acknowledge the risk.


def test_dicom_tf_dataset(mock_config):
    """
        Test the DicomTFDataset class to ensure it correctly loads DICOM files
        and handles dynamic shapes via padded_batch.

        This test verifies that:
        - The expected directory exists and contains DICOM files.
        - The dataset is not empty and can be iterated over.
        - The output is a tuple (image_batch, shape_batch).
        - The batch shapes are consistent with batch size and padded dimensions.
        - No significant loading errors (indicated by all -1.0 values) occur.
    """

    # --- Setup ---
    # The path is injected by the pytest fixture
    dicom_dir = mock_config["root_dir"] + '/' + mock_config["dicom_study_dir"]
    BATCH_SIZE = 2

    # Check that the directory exists and contains DICOM files
    assert tf.io.gfile.exists(dicom_dir), f"Directory {dicom_dir} does not exist"
    dicom_files = tf.io.gfile.glob(f"{dicom_dir}/*/*/*.dcm")
    assert len(dicom_files) >= 1, "No DICOM files found in the test fixture directory"

    # Create the dataset
    dataset = DicomTFDataset(root_dir=dicom_dir)
    tf_dataset = dataset.create_tf_dataset(batch_size=BATCH_SIZE)

    # Check that the dataset is not None
    assert tf_dataset is not None, "Dataset is None"

    # --- Iteration and Assertion ---
    count = 0
    # The dataset now yields a tuple: (image_batch, shape_batch)
    for image_batch, shape_batch in tf_dataset:
        count += 1

        # Check the first batch structure and shape
        if count >= 1:
            # 1. Check Image Batch Shape
            # Expected shape: (BATCH_SIZE, PADDED_HEIGHT, PADDED_WIDTH, PADDED_DEPTH)
            assert_msg = f"Image batch expected 4D tensor, got shape {image_batch.shape}"
            assert len(image_batch.shape) == 4, assert_msg

            assert_msg = f"Expected batch size {BATCH_SIZE}, got {image_batch.shape[0]}"
            assert image_batch.shape[0] == BATCH_SIZE, assert_msg

            # Padded dimensions (1, 2, 3) are dynamic (None in padded_shapes)
            assert_msg = "Padded image dimensions should be known after batching."
            assert image_batch.shape[1] is not None and image_batch.shape[2] is not None, assert_msg

            # 2. Check Shape Batch Shape
            # Expected shape: (BATCH_SIZE, 3) -> [H, W, D/C] for each image
            assert_msg = f"Shape batch expected 2D tensor, got shape {shape_batch.shape}"
            assert len(shape_batch.shape) == 2, assert_msg

            assert_msg = (
                                "Shape batch expected first dimension to be ",
                                f"batch size {BATCH_SIZE}, got {shape_batch.shape[0]}"
                            )
            assert shape_batch.shape[0] == BATCH_SIZE, assert_msg

            assert_msg = f"Shape vector expected size 3, got {shape_batch.shape[1]}"
            assert shape_batch.shape[1] == 3, assert_msg

            # 3. Check for loading errors
            # Verify that not all values in the image batch are the error code (-1.0).
            # We check if the sum of all elements is not equal to -1.0 * total elements.
            total_elements = tf.cast(tf.reduce_prod(tf.shape(image_batch)), tf.float32)
            error_sum = total_elements * -1.0

            # Check if the sum of the image batch is NOT equal to the error sum
            # We use a small epsilon for floating point comparison robustness
            assert_msg = (
                            "All values in image batch are -1.0, "
                            "indicating an error in loading DICOM files."
                         )
            assert tf.abs(tf.reduce_sum(image_batch) - error_sum) > 1e-6, assert_msg

            break  # Stop after checking the first batch


def test_dicom_tf_dataset_invalid_root_dir():
    """
    Test that DicomTFDataset raises a ValueError if the root directory does not exist.
    """
    with pytest.raises(ValueError):
        DicomTFDataset(root_dir="/nonexistent/directory")


def test_dicom_tf_dataset_empty_directory(tmp_path):
    """
        Test that DicomTFDataset raises a FileNotFoundError if no DICOM files are found.
    """
    # Create an empty directory structure
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        DicomTFDataset(root_dir=str(empty_dir))


def test_dicom_tf_dataset_load_error(mock_config):
    """
        Test that DicomTFDataset handles DICOM loading errors gracefully.
    """

    dicom_samples_root = str(Path(mock_config["root_dir"]) / mock_config["dicom_study_dir"])

    dataset = DicomTFDataset(root_dir=dicom_samples_root)

    with patch("SimpleITK.ReadImage", side_effect=Exception("Mocked error")):
        dummy_image, dummy_shape = dataset._py_load_dicom_tf(tf.constant(b"dummy_path"))
        assert tf.reduce_all(tf.equal(dummy_image, -1.0)), "Dummy image should be filled with -1.0"
        assert tf.reduce_all(tf.equal(dummy_shape, [1, 1, 1])), "Dummy shape should be [1, 1, 1]"


def test_dicom_tf_dataset_load_dicom(mock_config):
    """
        Test the _load_dicom method directly.
    """
    dicom_samples_root = str(Path(mock_config["root_dir"]) / mock_config["dicom_study_dir"])
    dataset = DicomTFDataset(root_dir=dicom_samples_root)
    first_file = dataset._file_paths_list[0]
    image_tensor, shape_tensor = dataset._load_dicom(tf.constant(first_file))
    assert isinstance(image_tensor, tf.Tensor), "Image should be a TensorFlow tensor"
    assert isinstance(shape_tensor, tf.Tensor), "Shape should be a TensorFlow tensor"


def test_dicom_tf_dataset_without_padding(mock_config):
    """
    Test that DicomTFDataset creates a batched dataset without padding when use_padding=False.
    This ensures the line `dataset = dataset.batch(batch_size=batch_size)` is covered.
    """

    dicom_samples_root = str(Path(mock_config["root_dir"]) / mock_config["dicom_study_dir"])

    # Create the dataset
    dataset = DicomTFDataset(root_dir=dicom_samples_root)

    # Create a dataset with padding disabled
    tf_dataset = dataset.create_tf_dataset(batch_size=2, use_padding=False)

    # Verify the dataset is not None
    assert tf_dataset is not None, "Dataset is None"

    # Iterate over the dataset to ensure it works without padding
    for image_batch, shape_batch in tf_dataset:

        # Check that the batch size is correct
        assert image_batch.shape[0] == 2, f"Expected batch size 2, got {image_batch.shape[0]}"

        # Check that the shapes are consistent
        assert_msg = f"Image batch expected 4D tensor, got shape {image_batch.shape}"
        assert len(image_batch.shape) == 4, assert_msg

        assert_msg = f"Shape batch expected 2D tensor, got shape {shape_batch.shape}"
        assert len(shape_batch.shape) == 2, assert_msg
        break  # Only check the first batch
