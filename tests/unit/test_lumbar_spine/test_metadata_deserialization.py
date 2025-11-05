# coding: utf-8

import io
import struct
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
from typing import Tuple, Any
import tensorflow as tf


class TestMetadataDeserialization:
    """
        Unit tests for metadata deserialization in LumbarDicomTFRecordDataset.
    """

    @pytest.fixture(autouse=True)
    def setup(self, mock_config, mock_logger):
        """Fixture to initialize common attributes for all tests."""

        self.mock_config = mock_config
        self.mock_logger = mock_logger
        self.root_dir = Path(self.mock_config["dicom_study_dir"])
        self.output_dir = Path(self.mock_config["tfrecord_dir"])

    def test_deserialize_metadata(self):
        """
            Test the complete metadata deserialization process.
        """
        # Setup
        metadata_bytes = (
            b'\x00\x00\x00\x00\x01'  # study_id: 1
            b'\x00\x00\x00\x00\x02'  # series_id: 2
            b'\x00\x01'              # instance_number: 1 (big-endian)
            b'\x03'                 # description: 3
            b'\x04'                 # condition: 4
            b'\x02'                 # nb_records: 2
            b'\x01'                 # record 1: level: 1
            b'\x02'                 # record 1: severity: 2
            b'\x00\x07\xD0'          # record 1: x: 2000 (20.00)
            b'\x00\x13\x88'          # record 1: y: 5000 (50.00)
            b'\x03'                 # record 2: level: 3
            b'\x01'                 # record 2: severity: 1
            b'\x00\x0F\xA0'          # record 2: x: 4000 (40.00)
            b'\x00\x1E\xE0'          # record 2: y: 7904 (79.04)
        )

        # Mock logger
        logger = MagicMock()

        logger_path = "src.core.utils.logger.get_current_logger"
        with patch(logger_path, return_value=self.mock_logger):

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config, logger=self.mock_logger)

            # Execute
            result = dataset._deserialize_metadata(metadata_bytes, logger=logger)

            # Assert
            assert result == {
                'study_id': 1,
                'series_id': 2,
                'instance_number': 1,
                'description': 3,
                'condition': 4,
                'nb_records': 2,
                'records': [
                    (1, 2, 20.00, 50.00),
                    (3, 1, 40.00, 79.04)
                ]
            }

    def test_deserialize_metadata_empty(self):
        """Test deserialization with empty byte sequence."""

        logger_path = "src.core.utils.logger.get_current_logger"
        with patch(logger_path, return_value=self.mock_logger):

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config, logger=self.mock_logger)

            # Execute & Assert
            with pytest.raises(ValueError, match="Input byte sequence is empty"):
                dataset._deserialize_metadata(b'')

    def test_deserialize_metadata_invalid_input(self):
        """Test deserialization with invalid input."""

        logger_path = "src.core.utils.logger.get_current_logger"
        with patch(logger_path, return_value=self.mock_logger):
            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                 logger=self.mock_logger)

            # Test with empty byte sequence
            with pytest.raises(ValueError, match="Input byte sequence is empty"):
                dataset._deserialize_metadata(b'')

            # Test with non-byte sequence input
            with pytest.raises(ValueError, match="Input must be a byte sequence"):
                dataset._deserialize_metadata("not a byte sequence")

            # Test with insufficient buffer length
            with pytest.raises(struct.error, match="Invalid buffer length"):
                dataset._deserialize_metadata(b'\x00\x00')  # Too short buffer

    def test_deserialize_metadata_too_short(self):
        """Test deserialization with too short byte sequence."""

        logger_path = "src.core.utils.logger.get_current_logger"
        with patch(logger_path, return_value=self.mock_logger):

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                 logger=self.mock_logger)

            # Execute & Assert
            with pytest.raises(struct.error):
                dataset._deserialize_metadata(b'\x00\x00\x00\x00\x01',
                                              logger=self.mock_logger)

    def test_deserialize_header(self):
        """Test header deserialization."""
        # Setup
        buffer = io.BytesIO(
            b'\x00\x00\x00\x00\x01'  # study_id: 1
            b'\x00\x00\x00\x00\x02'  # series_id: 2
            b'\x00\x01'              # instance_number: 1 (big-endian)
            b'\x03'                 # description: 3
            b'\x04'                 # condition: 4
            b'\x02'                 # nb_records: 2
        )

        logger_path = "src.core.utils.logger.get_current_logger"
        with patch(logger_path, return_value=self.mock_logger):

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                 logger=self.mock_logger)

            # Execute
            result = dataset._deserialize_header(buffer)

            # Assert
            assert result == {
                'study_id': 1,
                'series_id': 2,
                'instance_number': 1,
                'description': 3,
                'condition': 4,
                'nb_records': 2
            }

    def test_deserialize_records(self):
        """Test records deserialization."""
        # Setup
        buffer = io.BytesIO(
            b'\x01'                 # record 1: level: 1
            b'\x02'                 # record 1: severity: 2
            b'\x00\x07\xD0'          # record 1: x: 2000 (20.00)
            b'\x00\x13\x88'          # record 1: y: 5000 (50.00)
            b'\x03'                 # record 2: level: 3
            b'\x01'                 # record 2: severity: 1
            b'\x00\x0F\xA0'          # record 2: x: 4000 (40.00)
            b'\x00\x1E\xE0'          # record 2: y: 7904 (79.04)
        )

        logger_path = "src.core.utils.logger.get_current_logger"
        with patch(logger_path, return_value=self.mock_logger):

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                 logger=self.mock_logger)

            # Execute
            result = dataset._deserialize_records(buffer, 2)

            # Assert
            assert result == [
                (1, 2, 20.00, 50.00),
                (3, 1, 40.00, 79.04)
            ]

    def test_deserialize_records_empty(self):
        """Test records deserialization with zero records."""
        # Setup
        buffer = io.BytesIO(b'')
        nb_records = 0

        logger_path = "src.core.utils.logger.get_current_logger"
        with patch(logger_path, return_value=self.mock_logger):

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                 logger=self.mock_logger)

            # Execute
            result = dataset._deserialize_records(buffer, nb_records)

            # Assert
            assert result == []

    def test_deserialize_metadata_handles_exception_and_raises(
                                                                self,
                                                                mock_setup: Tuple[dict, MagicMock],
                                                                tmp_path: Path
                                                               ) -> None:
        """
        Covers the 'except' block in _deserialize_metadata.
        Verifies that an exception during the deserialization process is caught, logged
        with status 'failed', and then a new Exception is re-raised.
        """
        mock_config, mock_logger = mock_setup

        # Initialize the Dataset (skipping TFRecord generation)
        with patch.object(
                            LumbarDicomTFRecordDataset,
                            '_generate_tfrecord_files',
                            return_value=None
                          ):
            dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        exception_message = "Simulated deserialization component failure"

        # Mock: Inject an exception during the header deserialization (inside the try block)
        with patch.object(dataset, '_deserialize_header',
                          side_effect=RuntimeError(exception_message)):

            # Use 31 bytes to pass the minimal length check (MINIMUM_BUFFER_LENGTH = 31)
            mock_metadata_bytes = b'\x00' * 31

            # Assert that the outer generic Exception is correctly re-raised
            with pytest.raises(Exception) as excinfo:
                dataset._deserialize_metadata(mock_metadata_bytes, logger=mock_logger)

            # Verification 1: The error was propagated with the correct message format
            # The final raise is: raise Exception(f"Error deserializing metadata: {str(e)}")
            assert f"Error deserializing metadata: {exception_message}" in str(excinfo.value)

            # Verification 2: The error was logged
            mock_logger.error.assert_called_once()
            args, kwargs = mock_logger.error.call_args

            # Check logged message and error details
            # The logged message is: "Error in function _deserialize_metadata: {str(e)}"
            assert "Error in function _deserialize_metadata" in args[0]
            assert exception_message in args[0]

            # Check the 'extra' arguments for the required status and error info
            assert kwargs["extra"]["status"] == "failed"
            assert kwargs["extra"]["error"] == exception_message
            assert kwargs["exc_info"] is True

    def test_py_deserialize_and_flatten_success(
                                                    self,
                                                    mock_setup: Tuple[dict[str, Any], MagicMock],
                                                    tmp_path: Path
                                                 ) -> None:
        """
            Tests the successful deserialization and flattening of metadata.
        """

        mock_config, mock_logger = mock_setup
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

                dataset = LumbarDicomTFRecordDataset(mock_config,
                                                     logger=mock_logger)

                result = dataset._py_deserialize_and_flatten(
                                                        mock_metadata_bytes)

                assert len(result) == 7
                for i in range(6):
                    assert result[i].dtype == tf.int32

                assert result[-1].dtype == tf.float32
                assert result[0].numpy() == 1
                assert result[5].numpy() == 2
                assert result[-1].shape == (100,)

    def test_py_deserialize_and_flatten_missing_key_raises_value_error(
        self,
        mock_setup: Tuple[dict, MagicMock],
        tmp_path: Path
    ) -> None:
        """
            Verifies that a ValueError is raised if a required key is missing
            in the deserialized metadata dictionary ad that this error is caught
            by the outer generic handler, which logs the issue and returns
            the default (zeroed) tensors.
        """
        mock_config, mock_logger = mock_setup

        # 1. Initialize the Dataset (skipping TFRecord generation)
        with patch.object(
                            LumbarDicomTFRecordDataset,
                            '_generate_tfrecord_files',
                            return_value=None
                          ):
            dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # 2. Simulate incomplete metadata dictionary (missing 'study_id')
            incomplete_metadata = {
                'series_id': 1, 'instance_number': 1,
                'description': 0, 'condition': 0, 'nb_records': 0,
                'records': []
            }

        # 3. Mock _deserialize_metadata to return the incomplete dictionary
            with patch.object(dataset, '_deserialize_metadata',
                              return_value=incomplete_metadata):
                # The input tensor is required by the signature, but not used by the mock
                mock_tensor = tf.constant(b"mock_bytes")

                # Assert that ValueError is raised
                result_tensors = dataset._py_deserialize_and_flatten(mock_tensor)

                # Check the returned tensors : should be the default zeroed tensors
                assert len(result_tensors) == 7

                # Check the header tensors (6 x tf.int32, value 0)
                for idx in range(6):
                    assert result_tensors[idx].dtype == tf.int32
                    assert result_tensors[idx].numpy() == 0

                # Check the records tensor (tf.float32, shape 100, all 0.0)
                assert result_tensors[6].dtype == tf.float32
                assert result_tensors[6].shape == (100,)
                assert (result_tensors[6].numpy() == 0.0).all()

                # Verification 2: Chzek the error log
                mock_logger.error.assert_called_once()
                args, kwargs = mock_logger.error.call_args

                # The logged message should contain the original ValueError details
                assert "Error in _py_deserialize_and_flatten" in args[0]
                assert "Missing required key in deserialized metadata: study_id" in args[0]
                assert kwargs["exc_info"] is True

    def test_py_deserialize_and_flatten_handles_exception_and_returns_defaults(
        self,
        mock_setup: Tuple[dict, MagicMock],
        tmp_path: Path
    ) -> None:
        """
            Verifies that the 'except' block catches a generic exception, logs the error,
            and returns the default (zeroed) tensors.
        """
        mock_config, mock_logger = mock_setup

        # 1. Initialize the Dataset (skipping TFRecord generation)
        with patch.object(
                            LumbarDicomTFRecordDataset,
                            '_generate_tfrecord_files',
                            return_value=None
                          ):
            dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        exception_message = "Simulated internal failure for generic exception"

        # 2. Mock _deserialize_metadata to raise a generic exception
        # (This will trigger the 'except' block after the numpy() call)
        with patch.object(dataset, '_deserialize_metadata',
                          side_effect=Exception(exception_message)):
            # The input tensor is required by the signature, but not used by the mock
            mock_tensor = tf.constant(b"mock_bytes")

            # 3. Call the method (the exception will be caught and handled)
            result_tensors = dataset._py_deserialize_and_flatten(mock_tensor)

            # 4. Verification 1: Structure of the returned tensors
            assert len(result_tensors) == 7

            # Check header tensors (6 x tf.int32, value 0)
            for idx in range(6):
                assert result_tensors[idx].dtype == tf.int32
                assert result_tensors[idx].numpy() == 0

            # Check records tensor (tf.float32, shape 100, all 0.0)
            assert result_tensors[6].dtype == tf.float32
            assert result_tensors[6].shape == (100,)
            assert (result_tensors[6].numpy() == 0.0).all()

            # 5. Verification 2: The error was logged (Line 379)
            mock_logger.error.assert_called_once()
            args, kwargs = mock_logger.error.call_args

            assert f"Error in _py_deserialize_and_flatten: {exception_message}" in args[0]
            assert kwargs["exc_info"] is True
