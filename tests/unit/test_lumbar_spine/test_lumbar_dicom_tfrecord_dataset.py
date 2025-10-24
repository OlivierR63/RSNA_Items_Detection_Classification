# tests/test_lumbar_dicom_tfrecord_dataset.py
from unittest.mock import patch, MagicMock
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
import pytest
import pandas as pd
import tensorflow as tf
from pathlib import Path


class TestLumbarDicomTFRecordDataset:
    @pytest.fixture(autouse=True)
    def setup(self, mock_config, mock_logger):
        """Fixture to initialize attributes common to all tests."""
        self.mock_config = mock_config
        self.mock_logger = mock_logger
        self.dataset = None  # Sera initialisé dans chaque test si nécessaire

    def test_generate_tfrecord_files(self, mock_csv_metadata, mock_convert_dicom, tmp_path):
        """
            Tests the TFRecord files generation process,
            which is triggered upon object initialization.
            The underlying I/O operations are mocked.
        """
        # Mock get_current_logger
        logger_path = "src.core.utils.logger.get_current_logger"
        with patch(logger_path, return_value=self.mock_logger):

            # Mock the CSVMetadata class to avoid file reading
            csv_metadata_chain = (
                            "src.projects.lumbar_spine."
                            "lumbar_dicom_tfrecord_dataset.CSVMetadata"
                            )
            with patch(csv_metadata_chain, return_value=mock_csv_metadata):

                # Configure the mock_csv_metadata to return a mock dataframe
                mock_csv_metadata._merged_df = pd.DataFrame({
                    "study_id": [12345678],
                    "series_id": [87654321],
                    "instance_number": [1],
                    "condition": [2],
                    "severity": [1],
                    "series_description": [0],
                    "level": [3],
                    "x": [12.34],
                    "y": [56.78]
                })

                # Mock _convert_dicom_to_tfrecords to avoid I/O operations
                with patch.object(LumbarDicomTFRecordDataset,
                                  '_convert_dicom_to_tfrecords',
                                  mock_convert_dicom):

                    # Initialize the dataset,
                    # WHICH SHOULD TRIGGER _generate_tfrecord_files
                    dataset = LumbarDicomTFRecordDataset(self.mock_config, logger=self.mock_logger)

                    # Define the _tfrecord_dir path to simulate
                    # the expected state after initialization
                    dataset._tfrecord_dir = Path(tmp_path/"tfrecords")
                    dataset._tfrecord_dir.mkdir(parents=True, exist_ok=True)

                    # Verification checks
                    self.mock_logger.info.assert_any_call(
                        "Starting generate_tfrecord_file",
                        extra={"action": "generate_tf_records"}
                    )
                    self.mock_logger.info.assert_called_with(
                        "DICOM to TFRecord conversion completed.",
                        extra={"status": "success"}
                    )
                    mock_convert_dicom.assert_called_once()

    def test_create_tf_dataset(self):
        """
            Tests the creation of the TensorFlow Dataset pipeline.
        """
        logger_path = "src.core.utils.logger.get_current_logger"
        with patch(logger_path, return_value=self.mock_logger):

            # 1. Create a final mock dataset to be returned 
            final_mock_dataset = MagicMock(name='final_dataset')

            # 2. Use patch to mock tensorflow.data.Dataset.list_files
            tf_list_files_str = "tensorflow.data.Dataset.list_files"
            with patch(tf_list_files_str) as mock_list_files:

                mock_chain = (
                    "interleave.return_value."
                    "shuffle.return_value."
                    "batch.return_value."
                    "prefetch.return_value"
                )

                mock_list_files.return_value.configure_mock(
                    **{mock_chain: final_mock_dataset}
                )

                # Mock _generate_tfrecord_files during initialization
                # to ensure no side effects
                with patch.object(
                                    LumbarDicomTFRecordDataset,
                                    '_generate_tfrecord_files',
                                    return_value=None
                                    ):
                    self.dataset = LumbarDicomTFRecordDataset(
                                                      self.mock_config,
                                                      logger=self.mock_logger)

                self.dataset._tfrecord_pattern = (
                            "tmp_path/tfrecords/*.tfrecord"
                        )
                result = self.dataset.create_tf_dataset(batch_size=8)

                # Verification that the output is the result
                # of the entire chain (final_mock_dataset)
                assert result == final_mock_dataset

                # Log verifications
                self.mock_logger.info.assert_any_call(
                    "Creating TF Dataset with batch_size=8",
                    extra={"action": "create_dataset", "batch_size": 8}
                )

                self.mock_logger.info.assert_called_with(
                    "Dataset pipeline created successfully",
                    extra={"status": "success"}
                )

                # Check the calls to the chained methods
                # on the mock returned by list_files
                mock_list_files.assert_called_once()

                # Get the mock returned by list_files
                mock_interleave = mock_list_files.return_value.interleave
                mock_interleave.assert_called_once()

                # Get the mock returned by interleave
                mock_shuffle = mock_interleave.return_value.shuffle
                mock_shuffle.assert_called_once()

                # Get the mock returned by shuffle
                mock_batch = mock_shuffle.return_value.batch
                mock_batch.assert_called_once_with(8)

    def test_py_deserialize_and_flatten_success(self):
        """
            Tests the successful deserialization and flattening of metadata.
        """
        mock_metadata_bytes = tf.constant(b"fake_metadata_bytes")

        with patch.object(LumbarDicomTFRecordDataset,
                          '_deserialize_metadata',
                          return_value={
                                        'study_id': 1,
                                        'series_id': 1,
                                        'instance_number': 1,
                                        'description': 2,
                                        'condition': 3,
                                        'nb_records': 2,
                                        'records': [
                                                        (0, 1, 12.34, 56.78),
                                                        (1, 0, 90.12, 34.56)
                                                    ]
                                        }):
            with patch.object(LumbarDicomTFRecordDataset,
                              '_generate_tfrecord_files',
                              return_value=None):

                self.dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                            logger=self.mock_logger)

                result = self.dataset._py_deserialize_and_flatten(
                                                        mock_metadata_bytes)

                assert len(result) == 7
                for i in range(6):
                    assert result[i].dtype == tf.int32
                
                assert result[-1].dtype == tf.float32
                assert result[0].numpy() == 1
                assert result[5].numpy() == 2
                assert result[-1].shape == (100,)

    def test_parse_tfrecord(self):
        """Tests the parsing of a single TFRecord entry."""
        with patch("tensorflow.io.parse_single_example") as mock_parse_single_example, \
             patch("tensorflow.io.parse_tensor") as mock_parse_tensor, \
             patch("tensorflow.reshape") as mock_reshape, \
             patch("tensorflow.py_function") as mock_py_function, \
             patch("tensorflow.reduce_max") as mock_reduce_max, \
             patch("tensorflow.cast") as mock_cast, \
             patch("src.core.utils.logger.get_current_logger", return_value=self.mock_logger):

            # Mock _generate_tfrecord_files during initialization to avoid side effects
            with patch.object(LumbarDicomTFRecordDataset,
                                '_generate_tfrecord_files',
                                return_value=None):
                self.dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                            logger=self.mock_logger)

                # Create a mock TFRecord example (the input to the function)
                mock_proto = tf.constant(b"fake_proto")

                # Mock parse_single_example to return a valid structure
                mock_parse_single_example.return_value = {
                    "image": tf.constant([1] * 64 * 64 * 64 * 1, dtype=tf.uint16),
                    "metadata": tf.constant(b"fake_metadata")
                }

                # Mock parse_tensor to return a valid 1D tensor
                mock_parse_tensor.return_value = tf.constant(
                                                        [1] * 64 * 64 * 64 * 1,
                                                        dtype=tf.uint16
                                                        )

                # Mock reshape to return different values on successive calls:
                # 1st call for image, 2nd call for records
                mock_reshape.side_effect = [
                    tf.ones([64, 64, 64, 1], dtype=tf.float32),  # image
                    tf.ones([25, 4], dtype=tf.float32)           # records
                ]

                # Define the mock return for tf.py_function
                mock_header_tensors = [
                    tf.constant(1, dtype=tf.int32),   # study_id
                    tf.constant(1, dtype=tf.int32),   # series_id
                    tf.constant(1, dtype=tf.int32),   # instance_number
                    tf.constant(2, dtype=tf.int32),   # description (encoded)
                    tf.constant(3, dtype=tf.int32),   # condition (encoded)
                    tf.constant(2, dtype=tf.int32),   # nb_records
                ]
                mock_records_flat = tf.zeros([100], dtype=tf.float32)
                mock_py_function.return_value = (
                        mock_header_tensors + [mock_records_flat]
                )

                # Mock reduce_max to return a constant
                mock_reduce_max.return_value = tf.constant(1.0)

                # Mock cast to return the input as-is
                mock_cast.side_effect = lambda x, dtype: x

                # Call the method
                image, metadata = self.dataset._parse_tfrecord(mock_proto)

                # Verifications
                assert image.shape == (64, 64, 64, 1)
                assert metadata["study_id"].numpy() == 1
                assert metadata["nb_records"].numpy() == 2
                assert tuple(metadata["records"].shape) == (25, 4)

    def test_get_metadata_for_file(self):
        """
            Tests the retrieval of metadata for a specific DICOM file.
        """

        # Create a mock file path
        mock_file_path = "/fake/root_dir/1/2/1.dcm"

        # Create a mock metadata DataFrame
        mock_metadata_df = pd.DataFrame({
            "study_id": [1, 1, 2],
            "series_id": [1, 2, 1],
            "instance_number": [1, 2, 1],
            "other_column": ["value1", "value2", "value3"]
        })

        with patch("src.core.utils.logger.get_current_logger", return_value=self.mock_logger):

            # Initialize the dataset
            dataset = LumbarDicomTFRecordDataset(self.mock_config, logger=self.mock_logger)

            with (
                  patch.object(dataset, '_generate_tfrecord_files', return_value=None),
                  patch.object(dataset, '_serialize_metadata') as mock_serialize_metadata
                  ):

                # Configure the mock for _serialize_metadata
                mock_serialize_metadata.return_value = (
                    b"serialized_metadata_bytes"
                )

                # Call the method
                result = dataset._get_metadata_for_file(mock_file_path,
                                                        mock_metadata_df)

                # Verifications
                # Check that the logger was called correctly
                self.mock_logger.info.assert_any_call(
                    "Starting retrieving metadata from CSV files"
                )

                # Check that _serialize_metadata was called with the correct arguments
                mock_serialize_metadata.assert_called_once_with(
                                               "1", "2", "1", mock_metadata_df)

                # Check that the result is the expected serialized metadata
                assert result == b"serialized_metadata_bytes"

                # Test with None metadata_df
                result = dataset._get_metadata_for_file(mock_file_path, None)
                assert result == b''

                # Check that the logger was called again for the second call
                msg_str = "Starting retrieving metadata from CSV files"
                self.mock_logger.info.assert_any_call(msg_str)
