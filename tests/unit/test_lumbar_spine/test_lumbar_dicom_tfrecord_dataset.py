# tests/test_lumbar_dicom_tfrecord_dataset.py
from unittest.mock import patch, MagicMock
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
import pytest
import pandas as pd
import tensorflow as tf
from tests import conftest
from pathlib import Path


# Fix: Mock _convert_dicom_to_tfrecords globally to prevent I/O during initialization.
# Fix: Mock get_current_logger to avoid the RuntimeError caused by @log_method.
@patch("src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset.LumbarDicomTFRecordDataset._convert_dicom_to_tfrecords")
@patch("src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset.CSVMetadata")
@patch("src.core.utils.logger.get_current_logger")
def test_generate_tfrecord_files(mock_get_logger, mock_csv_metadata, mock_convert_dicom, mock_config, mock_logger):
    """
    Tests the TFRecord files generation process, which is triggered upon object initialization.
    The underlying I/O operations are mocked.
    """
    
    # Assign the mock_logger
    mock_get_logger.return_value = mock_logger 
    
    # Mock data for the CSVMetadata class
    mock_csv_instance = MagicMock()
    mock_csv_metadata.return_value = mock_csv_instance
    mock_csv_instance._merged_df = pd.DataFrame({"study_id": [1], "series_id": [1], "instance_number": [1]})

    # Initialization calls _generate_tfrecord_files() which calls the mocked _convert_dicom_to_tfrecords.
    dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)
    
    # Define the _tfrecord_dir path to simulate the expected state after initialization
    dataset._tfrecord_dir = Path("tests/tmp/tfrecords")
    dataset._tfrecord_dir.mkdir(parents=True, exist_ok=True)

    # Verification checks
    # 1. Verify the 'start' log happened at some point (using assert_any_call)
    mock_logger.info.assert_any_call("Starting generate_tfrecord_file", extra={"action": "generate_tf_records"})

    # 2. Verify the 'completion' log was the last call (using assert_called_with)
    mock_logger.info.assert_called_with("DICOM to TFRecord conversion completed.", extra={"status": "success"})

    # 3. The I/O method _convert_dicom_to_tfrecords must have been called exactly once
    mock_convert_dicom.assert_called_once()


@patch("tensorflow.data.Dataset.list_files")
@patch("src.core.utils.logger.get_current_logger")
def test_create_tf_dataset(mock_get_logger, mock_list_files, mock_config, mock_logger):
    """Tests the creation of the TensorFlow Dataset pipeline."""
    
    # Assign the mock_logger
    mock_get_logger.return_value = mock_logger
    
    # --- Fix for chained mocking issue (AssertionError: Full diff: -<MagicMock name='interleave()...') ---
    
    # 1. Create the final expected mock dataset
    final_mock_dataset = MagicMock(name='final_dataset')

    # 2. Configure the object returned by list_files (The first step of the pipeline)
    mock_list_files.return_value.configure_mock(**{
        'interleave.return_value.shuffle.return_value.batch.return_value.prefetch.return_value': final_mock_dataset
    })
    
    
    # Mock _generate_tfrecord_files during initialization to ensure no side effects
    with patch.object(LumbarDicomTFRecordDataset, '_generate_tfrecord_files', return_value=None):
        dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)
        
    dataset._tfrecord_pattern = "tests/tmp/tfrecords/*.tfrecord"

    result = dataset.create_tf_dataset(batch_size=8)

    # Verification that the output is the result of the entire chain (final_mock_dataset)
    assert result == final_mock_dataset
    
    # Log verifications
    
    # Verification 1: Start/configuration log was called (assert_any_call)
    mock_logger.info.assert_any_call(
        "Creating TF Dataset with batch_size=8", 
        extra={"action": "create_dataset", "batch_size": 8}
    )
    
    # Verification 2: Success/end log is the LAST call (assert_called_with)
    mock_logger.info.assert_called_with(
        "Dataset pipeline created successfully", 
        extra={"status": "success"}
    )

    # We can now also check the calls to the chained methods on the mock returned by list_files.
    mock_list_files.assert_called_once()
    mock_list_files.return_value.interleave.assert_called_once()
    mock_list_files.return_value.interleave.return_value.shuffle.assert_called_once()
    mock_list_files.return_value.interleave.return_value.shuffle.return_value.batch.assert_called_once_with(8)


# Fix: Mock get_current_logger
@patch("src.core.utils.logger.get_current_logger")
def test_parse_tfrecord(mock_get_logger, mock_logger, mock_config):
    """
    Tests the parsing of a single TFRecord entry by mocking the Python 
    deserialization logic (tf.py_function).
    """
    # Assign the mock_logger
    mock_get_logger.return_value = mock_logger
    
    # Mock _generate_tfrecord_files during initialization to avoid side effects
    with patch.object(LumbarDicomTFRecordDataset, '_generate_tfrecord_files', return_value=None):
        dataset = LumbarDicomTFRecordDataset(config = mock_config, logger=mock_logger)

    # Create a mock TFRecord example (the input to the function)
    mock_proto = tf.constant(b"fake_proto")

    # Define the mock return for tf.py_function
    # 6 Scalars (tf.int32) for the header + 1 flattened Tensor (tf.float32) for the records
    mock_header_tensors = [
        tf.constant(1, dtype=tf.int32),   # study_id
        tf.constant(1, dtype=tf.int32),   # series_id
        tf.constant(1, dtype=tf.int32),   # instance_number
        tf.constant(2, dtype=tf.int32),   # description (encoded)
        tf.constant(3, dtype=tf.int32),   # condition (encoded)
        tf.constant(0, dtype=tf.int32),   # nb_records
    ]
    # Flattened records Tensor (MAX_RECORDS * 4 = 100 elements, 25 is the max size)
    mock_records_flat = tf.zeros([100], dtype=tf.float32)
    mock_py_function_result = mock_header_tensors + [mock_records_flat]

    with patch("tensorflow.io.parse_single_example") as mock_parse:
        mock_parse.return_value = {
            "image": tf.constant(b"fake_image"),
            "metadata": tf.constant(b"fake_metadata")
        }
        with patch("tensorflow.io.parse_tensor") as mock_parse_tensor:
            # Mock the return of image deserialization (1D)
            mock_parse_tensor.return_value = tf.constant([1.0])
            with patch("tensorflow.reshape") as mock_reshape:
                # Mock the reshaping of the image to its final shape (64, 64, 64, 1)
                mock_reshape.return_value = tf.ones([64, 64, 64, 1], dtype=tf.float32)
                
                with patch("tensorflow.py_function") as mock_py_function:
                    # Mock for the return of py_deserialize_and_flatten
                    mock_py_function.return_value = mock_py_function_result
                    
                    # Mocks to simulate normalization and casting (final steps)
                    with patch("tensorflow.reduce_max", return_value=tf.constant(1.0)):
                        # Prevent tf.cast from crashing on mocked Tensors
                        with patch("tensorflow.cast", side_effect=lambda x, dtype: x): 

                            image, metadata = dataset._parse_tfrecord(mock_proto)

                            # Verifications
                            assert image.shape == (64, 64, 64, 1)
                            assert metadata["study_id"].numpy() == 1
                            print(f"metadata['records'].shape = {metadata['records'].shape}")
                            assert tuple(metadata["records"].shape) == (25, 4)                           
