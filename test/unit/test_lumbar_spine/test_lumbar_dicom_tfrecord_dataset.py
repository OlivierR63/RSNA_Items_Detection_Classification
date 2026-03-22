# coding utf-8

import tensorflow as tf
import logging
from unittest.mock import patch
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset

# Define the alias at the module level for maximum readability
MODULE_PATH = 'src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset'

# English comment: Constants for consistent test shapes
TEST_MAX_RECORDS = 25
TEST_BATCH_SIZE = 2
TEST_SERIES_DEPTH = 5


class TestLumbarDicomTFRecordDataset:

    # --- Initialization Tests ---

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
        dummy_metadata = {
            "study_id": tf.constant(0, dtype=tf.int32),
            "series_id": tf.constant(0, dtype=tf.int32),
            "description": tf.constant(0, dtype=tf.int32),
            "instance_number": tf.constant(1, dtype=tf.int32)
        }
        dummy_labels = {
            "severity": tf.constant(0, dtype=tf.int32),
            "x": tf.constant(0.5, dtype=tf.float32),
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
