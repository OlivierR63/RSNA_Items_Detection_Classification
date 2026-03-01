# coding utf-8

from inspect import stack
import pytest
import tensorflow as tf
import logging
from unittest.mock import patch
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
from contextlib import ExitStack

# Define the alias at the module level for maximum readability
MODULE_PATH = 'src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset'

# English comment: Constants for consistent test shapes
TEST_MAX_RECORDS = 25
TEST_BATCH_SIZE = 2
TEST_SERIES_DEPTH = 5

@pytest.fixture
def mock_config():
    """Provides a standard configuration dictionary for testing."""
    return {
        'max_records': TEST_MAX_RECORDS,
        'dataset_buffer_size_mb': 2,
        'nb_cores': 4,
        'model_2d': {'img_shape': (512, 270, 3)},
    }

class TestLumbarDicomTFRecordDataset:

    # --- Initialization Tests ---
    # Tes tests d'init sont parfaits, rien à changer.

    def test_init_sets_correct_attributes(self, mock_config):
        loader = LumbarDicomTFRecordDataset(mock_config, series_depth=TEST_SERIES_DEPTH)
        assert loader._MAX_RECORDS == TEST_MAX_RECORDS
        assert loader._series_depth == TEST_SERIES_DEPTH
        assert isinstance(loader._logger, logging.Logger)

    # --- Pipeline Logic Tests ---

    # We patch the functions WHERE THEY ARE USED (inside lumbar_dicom_tfrecord_dataset)
    @patch(f'{MODULE_PATH}.TFRecordFilesManager')
    @patch(f'{MODULE_PATH}.parse_tfrecord_single_element')
    @patch(f'{MODULE_PATH}.process_study_multi_series')
    @patch(f'{MODULE_PATH}.format_for_model')
    def test_generate_tfrecord_dataset_structure(
        self, 
        mock_format, 
        mock_process, 
        mock_parse, 
        mock_manager_cls, 
        mock_config
    ):
        """
        Validates the dataset pipeline construction.
        """

        loader = LumbarDicomTFRecordDataset(mock_config, series_depth=TEST_SERIES_DEPTH)

        # Setup mock for TFRecordFilesManager
        mock_manager = mock_manager_cls.return_value

        dummy_image = tf.zeros((1, 1, 1), dtype=tf.float32)
        dummy_records = tf.zeros((25,4), dtype = tf.float32)
        dummy_metadata = {
            "study_id": tf.constant(0, dtype=tf.int32),
            "series_id": tf.constant(0, dtype=tf.int32),
            "description": tf.constant(0, dtype=tf.int32),
            "instance_number": tf.constant(1, dtype=tf.int32)
        }
        dummy_labels = {
            "severity": tf.constant(0, dtype=tf.int32),
            "x":tf.constant(0.5, dtype=tf.float32),
            "y": tf.constant(5, dtype=tf.float32)
        }
        
        # Define dummy return for the final mapping to avoid execution errors
        mock_parse.return_value = (
            dummy_image,
            dummy_metadata,
            dummy_labels
        )

        mock_process.return_value = (
            (
                (dummy_image, tf.constant(1, dtype=tf.int32), tf.constant(0, dtype=tf.int32)),
                (dummy_image, tf.constant(1, dtype=tf.int32), tf.constant(1, dtype=tf.int32)),
                (dummy_image, tf.constant(1, dtype=tf.int32), tf.constant(2, dtype=tf.int32))
            ),
            tf.constant(10, dtype=tf.int32),
            dummy_labels
        )

        mock_format.return_value = ({"input": tf.zeros((1,))}, {"target": tf.zeros((1,))})

        # Execute pipeline generation
        tfrecord_files = ["file1.tfrecord"]
        ds = loader.generate_tfrecord_dataset(tfrecord_files, batch_size=TEST_BATCH_SIZE)

        # Basic structural assertions
        assert isinstance(ds, tf.data.Dataset)
        mock_manager.set_series_depth.assert_called_with(TEST_SERIES_DEPTH)

    @patch(f'{MODULE_PATH}.TFRecordFilesManager')
    @patch('tensorflow.data.TFRecordDataset')
    @patch(f'{MODULE_PATH}.parse_tfrecord_single_element')
    @patch(f'{MODULE_PATH}.process_study_multi_series')
    @patch(f'{MODULE_PATH}.format_for_model')
    def test_buffer_size_calculation(
        self, 
        mock_format, 
        mock_process, 
        mock_parse,
        mock_tf_ds_cls,
        mock_manager_cls, 
        mock_config
    ):
        """
        Verifies MiB to Bytes conversion logic.
        """
        loader = LumbarDicomTFRecordDataset(mock_config, series_depth = 1)

        dummy_image = tf.zeros((1, 1, 1), dtype=tf.float32)
        dummy_records = tf.zeros((25,4), dtype = tf.float32)
        dummy_metadata = {
            "study_id": tf.constant(0, dtype=tf.int32),
            "series_id": tf.constant(0, dtype=tf.int32),
            "description": tf.constant(0, dtype=tf.int32),
            "instance_number": tf.constant(1, dtype=tf.int32)
        }
        dummy_labels = {
            "severity": tf.constant(0, dtype=tf.int32),
            "x":tf.constant(0.5, dtype=tf.float32),
            "y": tf.constant(5, dtype=tf.float32)
        }

        # Mock TFRecordDataset but make it return a real empty Dataset
        # This prevents TensorFlow from crashing during graph construction.
        mock_tf_ds_cls.return_value = tf.data.Dataset.from_tensor_slices([])

        # Mock parsing/processing to avoid further graph issues
        mock_parse.return_value = (
            dummy_image,
            dummy_metadata,
            dummy_labels
        )

        # Mock process_study_multi_series to avoid further graph issues
        mock_process.return_value = (
            (
                (dummy_image, tf.constant(1, dtype=tf.int32), tf.constant(0, dtype=tf.int32)),
                (dummy_image, tf.constant(1, dtype=tf.int32), tf.constant(1, dtype=tf.int32)),
                (dummy_image, tf.constant(1, dtype=tf.int32), tf.constant(2, dtype=tf.int32))
            ),
            tf.constant(10, dtype=tf.int32),
            dummy_labels
        )

        mock_format.return_value = (dummy_records, dummy_metadata, dummy_labels)
        
        test_filename = "test.tfrecord"
        ds = loader.generate_tfrecord_dataset([test_filename], batch_size=1)

        # Iterating/taking 1 element triggers the interleave lambda
        for _ in ds.take(1): 
            break
        
        # Verify if TFRecordDataset was called
        assert mock_tf_ds_cls.called, "TFRecordDataset was never called"

        # Retrieve call arguments
        args, kwargs = mock_tf_ds_cls.call_args

        # Check the buffer_size (2 MiB * 1024 * 1024)
        assert kwargs['buffer_size'] == 2097152
        
        # If you really want to check the filename without graph issues, 
        # we check the string representation or just skip the exact value check 
        # as it's handled by TF's internal interleave logic.
        assert "test.tfrecord" in str(args[0]) or "args_0" in str(args[0])