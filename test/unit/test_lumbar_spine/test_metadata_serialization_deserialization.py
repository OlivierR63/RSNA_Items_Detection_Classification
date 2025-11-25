# coding: utf-8

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
from pathlib import Path
from typing import Tuple, Any
import tensorflow as tf
import io
import re
import struct


@pytest.fixture
def setup_dataframe():
    """Fixture to create a sample DataFrame for testing."""
    data = {
                "study_id": [1, 1, 2, 2, 1],
                "series_id": [10, 10, 20, 20, 10],
                "instance_number": [100, 200, 100, 200, 100],
                "condition": [1, 1, 2, 2, 1],
                "series_description": [3, 3, 4, 4, 3],
                "level": [1, 2, 1, 2, 2],
                "severity": [2, 1, 2, 1, 1],
                "x": [1.23, 4.56, 7.89, 0.12, 4.56],
                "y": [2.34, 5.67, 8.90, 1.23, 5.67]
            }
    return pd.DataFrame(data)


@pytest.fixture
def dataset_instance(
                        mock_setup: Tuple[dict[str, Any], MagicMock],
                        tmp_path: Path
                     ) -> None:
    """
        Fixture to create an instance of LumbarDicomTFRecordDataset
        with a mock logger.
    """
    mock_config, mock_logger = mock_setup

    with patch.object(LumbarDicomTFRecordDataset, '__init__', return_value=None):
        instance = LumbarDicomTFRecordDataset.__new__(LumbarDicomTFRecordDataset)
        instance._MAX_RECORDS = 25
        instance._MAX_RECORDS_FLAT = instance._MAX_RECORDS * 4
        instance.logger = mock_logger
        instance._config = mock_config
    return instance


class TestMetadataIntegrity:
    """
        Tests for serialization and deserialization of metadata
        in LumbarDicomTFRecordDataset, ensuring data integrity across the full cycle.
        This class merges all tests from the original serialization and deserialization files.
    """

    def test_serialize_metadata_successful(
                                    self,
                                    setup_dataframe: pd.DataFrame,
                                    dataset_instance: LumbarDicomTFRecordDataset,
                                    mock_setup: Tuple[dict[str, Any], MagicMock],
                                    tmp_path: Path
                                ) -> None:
        """
            Test the main serialization function.
        """

        _, mock_logger = mock_setup

        # Test with valid data. We will test two cases:
        # 1. A single record
        mask_1 = (
                   (setup_dataframe["study_id"] == 1) &
                   (setup_dataframe["series_id"] == 10) &
                   (setup_dataframe["instance_number"] == 200)
                )
        mask_dataframe_1 = setup_dataframe[mask_1]
        result_1 = dataset_instance._serialize_metadata(
                                                        mask_dataframe_1,
                                                        logger=mock_logger
                                                      )

        print(f"result_1 = {result_1}")

        # Verify the result is a non-empty bytes object
        assert isinstance(result_1, bytes)
        assert len(result_1) > 0

        # Verify the header part. Explanation:
        #      - Header is first 15 bytes
        #      - Payload follows : 8 bytes (Only one record)
        #      Total = 15 + 8 = 23 bytes
        assert len(result_1) == 23

        # Verify logger calls
        msg_str = "Starting function _serialize_metadata"
        mock_logger.info.assert_any_call(msg_str)

        msg_str = "Function _serialize_metadata completed successfully"
        mock_logger.info.assert_called_with(
                                                msg_str,
                                                extra={"status": "success"}
                                            )

        # 2. Multiple records (2 records)
        mask_2 = (
                   (setup_dataframe["study_id"] == 1) &
                   (setup_dataframe["series_id"] == 10) &
                   (setup_dataframe["instance_number"] == 100)
                )
        mask_dataframe_2 = setup_dataframe[mask_2]
        result_2 = dataset_instance._serialize_metadata(
                                                        mask_dataframe_2,
                                                        logger=mock_logger
                                                      )

        print(f"result_2 = {result_2}")

        # Verify the result is a non-empty bytes object
        assert isinstance(result_2, bytes)
        assert len(result_2) > 0

        # Verify the header part. Explanation:
        #      - Header is first 15 bytes
        #      - Payload follows : 16 bytes (2 records of 8 bytes each)
        #      Total = 15 + 16 = 31 bytes
        assert len(result_2) == 31

        # Verify logger calls
        msg_str = "Starting function _serialize_metadata"
        mock_logger.info.assert_any_call(msg_str)

        msg_str = "Function _serialize_metadata completed successfully"
        mock_logger.info.assert_called_with(
                                                msg_str,
                                                extra={"status": "success"}
                                            )

    def test_serialize_metadata_null_raises_exception(
        self,
        dataset_instance: LumbarDicomTFRecordDataset,
        mock_setup: Tuple[dict[str, Any], MagicMock]
    ) -> None:
        """
        Tests the null value detection block.
        Ensures the function logs an ERROR and raises an Exception immediately
        when nulls are detected.

        This test assumes that the error message is constructed as a single string (msg_error)
        which is then logged and raised. If the logging is changed to multiple calls,
        this test logic needs to be adjusted accordingly to check for all logging calls.
        """
        _, mock_logger = mock_setup

        # 1. Create a DataFrame with nulls in specific columns
        null_data_df = pd.DataFrame({
            'ID': [1, 2],
            'Value1': ['A', pd.NA],  # Null value 1
            'Value2': [None, 'B'],    # Null value 2
            'Value3': ['C', 'D']      # No nulls
        })

        mock_logger.reset_mock()

        # 2. Call function with nulls and assert that an Exception is raised
        with pytest.raises(ValueError) as excinfo:
            dataset_instance._serialize_metadata(null_data_df, logger=mock_logger)

        # 3. Define the expected error messages (assuming the log output matches the exception text)
        null_columns = ['Value1', 'Value2']

        expected_msg_part1 = "Null values detected in data_df before serialization."
        expected_msg_part2 = f"Columns affected: {null_columns}."
        expected_msg_part3 = "Serialization might fail or produce corrupted records."

        expected_full_error_message = (
            f"{expected_msg_part1} "
            f"{expected_msg_part2} "
            f"{expected_msg_part3}"
        )

        # 4. Assert that the error logger was called exactly three times
        mock_logger.error.acall_count == 3

        # 5. Assert the content of the error messages
        mock_logger.error.assert_any_call(expected_msg_part1)
        mock_logger.error.assert_any_call(expected_msg_part2)
        mock_logger.error.assert_any_call(expected_msg_part3)

        # 6. Check raised exception message
        assert str(excinfo.value) == expected_full_error_message

        # 7. Assert success info log was NOT called
        info_calls = [
            call for call in mock_logger.info.call_args_list
            if "completed successfully" in call[0][0]
        ]
        assert len(info_calls) == 0

    def test_serialize_metadata_handles_exception_and_raises(
                                                                self,
                                                                mock_setup: Tuple[dict, MagicMock],
                                                                tmp_path: Path
                                                              ) -> None:
        """
            Covers the 'except' block in _serialize_metadata.
            Verifies that an exception during serialization is caught, logged
            with status 'failed', and then correctly re-raised.
        """
        mock_config, mock_logger = mock_setup
        exception_message = "Simulated serialization component failure"

        # Setup Data: We need a non-empty DataFrame so the function proceeds
        # past the 'if records_df.empty' check.
        mock_data_df = pd.DataFrame(
                                        {
                                            'study_id': ['123'],
                                            'series_id': ['456'],
                                            'instance_number': [1],
                                            'series_description': [0],
                                            'condition': [0],
                                            'nb_records': [1],
                                            'records': [[]]
                                        }
                                     )

        # Mock & Initialization
        with (
                patch.object(
                                LumbarDicomTFRecordDataset,
                                '_generate_tfrecord_files',
                                return_value=None
                              ),

                # Inject the exception during the payload serialization (inside the try block)
                patch.object(LumbarDicomTFRecordDataset, '_serialize_payload',
                             side_effect=RuntimeError(exception_message))
              ):
            dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

            # Assert that the exception is correctly re-raised (L705)
            with pytest.raises(RuntimeError) as excinfo:

                # Call the method
                _ = dataset._serialize_metadata(
                                                    data_df=mock_data_df,
                                                    logger=mock_logger
                )

            # 4. Verification 1: The error was propagated with the correct message
            assert exception_message in str(excinfo.value)

            # 5. Verification 2: The error was logged (L702-704)
            mock_logger.error.assert_called_once()
            args, kwargs = mock_logger.error.call_args

            # Check logged message and error details
            assert "Error in function _serialize_metadata()" in args[0]
            assert exception_message in args[0]

            # Check the 'extra' arguments for the required status and error info
            assert kwargs["extra"]["status"] == "failed"
            assert kwargs["extra"]["error"] == exception_message
            assert kwargs["exc_info"] is True

    def test_serialize_metadata_empty_records(
                                                self,
                                                setup_dataframe: pd.DataFrame,
                                                dataset_instance: LumbarDicomTFRecordDataset,
                                                mock_setup: Tuple[dict[str, Any]],
                                                tmp_path: Path
                                               ) -> None:
        """Test the main serialization function with no matching records."""

        _, mock_logger = mock_setup

        # Test with non-matching data
        mask = (
                     (setup_dataframe["study_id"] == 999) &
                     (setup_dataframe["series_id"] == 999) &
                     (setup_dataframe["instance_number"] == 999)
               )

        mask_dataframe = setup_dataframe[mask]

        # ASSERTION 1 (CRITICAL): Assert that the call to _serialize_metadata
        # with an empty DataFrame raises a ValueError, originating from the check
        # within _serialize_header.
        err_msg = "Cannot serialize header: Input DataFrame is None or empty."
        with pytest.raises(ValueError, match=err_msg):
            dataset_instance._serialize_metadata(mask_dataframe, logger=mock_logger)

        # ASSERTION 2: Assert that the error was logged by the exception handler
        # in _serialize_metadata.
        mock_logger.error.assert_called_once()

        # ASSERTION 3: Ensure that no successful data logging (like 'info' or 'warning')
        # happened, as the process resulted in a failure.
        mock_logger.warning.assert_not_called()
        mock_logger.info.assert_called_once()

    def test_serialize_header_successful(
                                self,
                                setup_dataframe: pd.DataFrame,
                                dataset_instance: LumbarDicomTFRecordDataset,
                                tmp_path: Path
                              ) -> None:
        """
            Test the header serialization function.
        """

        # Filter the DataFrame to get records for:
        # study_id=1, series_id=10, instance_number=100
        records_df = setup_dataframe[
            (setup_dataframe["study_id"] == 1) &
            (setup_dataframe["series_id"] == 10) &
            (setup_dataframe["instance_number"] == 100)
        ]

        # Call the function
        result = dataset_instance._serialize_header(records_df)

        # Verify the result is a bytes object of the correct length
        assert isinstance(result, bytes)
        assert len(result) == 15  # Header should be 15 bytes

        # Verify the header content
        study_id = int.from_bytes(result[0:5], byteorder='big')
        series_id = int.from_bytes(result[5:10], byteorder='big')
        instance_number = int.from_bytes(result[10:12], byteorder='big')
        description = int.from_bytes(result[12:13], byteorder='big')
        condition = int.from_bytes(result[13:14], byteorder='big')
        nb_records = int.from_bytes(result[14:15], byteorder='big')

        assert study_id == 1
        assert series_id == 10
        assert instance_number == 100
        assert description == 3
        assert condition == 1
        assert nb_records == 2

    def test_serialize_header_exceeds_limit(
                                                self,
                                                mock_setup: Tuple[dict[str, Any], MagicMock]
                                            ) -> None:
        """
            Tests the limit check (Line 652: raise ValueError) in _serialize_header
            by providing a DataFrame with more than 25 records.
        """
        mock_config, mock_logger = mock_setup

        # Initialize the dataset object
        dataset_obj = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # Mock _generate_tfrecord_files to ensure no side effects during init
        with patch.object(dataset_obj, '_generate_tfrecord_files', return_value=None):

            # 1. Create a mock DataFrame with 26 records (1 over the limit of 25)
            # Only the length matters, but we need to include required columns for header logic
            data = {
                'study_id': "123",
                'series_id': "456",
                'instance_number': "7",
                'condition': [1] * 26,
                'series_description': [2] * 26,
                'level': [10] * 26,
                'severity': [20] * 26,
                'x': [0.1] * 26,
                'y': [0.2] * 26,
            }
            records_dataframe = pd.DataFrame(data)

            # 2. Assert that the ValueError is raised
            expected_error_message = "The number of records exceeds the limit of 25."

            with pytest.raises(ValueError, match=expected_error_message):
                dataset_obj._serialize_header(records_dataframe)

    def test_serialize_header_null_raises_exception(
                                    self,
                                    setup_dataframe: pd.DataFrame,
                                    dataset_instance: LumbarDicomTFRecordDataset,
                                    tmp_path: Path
                                ) -> None:
        """
        Tests the header's internal check for null values in critical columns.
        Ensures a ValueError is raised when nulls are found in HEADER_COLS.
        """
        # 1. Create a records_df with null values in a critical header column ('condition')
        # We clone a valid record set and introduce a null
        records_df_null = setup_dataframe[
            (setup_dataframe["study_id"] == 1) &
            (setup_dataframe["series_id"] == 10) &
            (setup_dataframe["instance_number"] == 100)
        ].copy()  # Must use .copy() to avoid SettingWithCopyWarning

        # Introduce a null value in 'condition'
        records_df_null.loc[records_df_null.index[0], 'condition'] = pd.NA

        # The expected error message pattern
        expected_cols = "['condition']"
        expected_message = (
            f"Cannot serialize header: "
            f"Null values detected in critical header columns: {expected_cols}. "
            "Serialization requires all header values to be non-null integers."
        )

        # 2. Assert that the ValueError is raised with the correct message
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            dataset_instance._serialize_header(records_df_null)

    def test_serialize_header_non_unique_raises_exception(
                                            self,
                                            dataset_instance: LumbarDicomTFRecordDataset
                                        ) -> None:
        """
        Tests the header's uniqueness check for critical header columns.
        Ensures a ValueError is raised when a column (like 'series_description')
        contains more than one unique value for a single header block.

        This test specifically covers the validation block:
        for col in [...]:
            if len(records_df[col].unique()) != 1:
                raise ValueError(...)
        """
        # 1. Create a records_df where two records share the same file key,
        # but one critical header property is inconsistent ('series_description')
        data_corrupted = {
            "study_id": [1, 1],
            "series_id": [10, 10],
            "instance_number": [100, 100],
            "condition": [1, 1],
            "series_description": [3, 4],  # This is the inconsistent column (2 unique values)
            "level": [1, 2],
            "severity": [2, 1],
            "x": [1.23, 4.56],
            "y": [2.34, 5.67]
        }
        records_df_corrupted = pd.DataFrame(data_corrupted)

        # 2. Define the expected error message for 'series_description'
        col = 'series_description'
        expected_message = (
            f"Cannot serialize header: Column '{col}' must contain exactly one unique value, "
            f"but found 2."
        )

        # 3. Assert that the ValueError is raised with the correct message
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            dataset_instance._serialize_header(records_df_corrupted)

    def test_serialize_payload_successful(
                            self,
                            setup_dataframe: pd.DataFrame,
                            dataset_instance: LumbarDicomTFRecordDataset,
                            tmp_path: Path
                           ) -> None:
        """
            Test the payload serialization function.
        """

        # Filter the DataFrame to get records for:
        # study_id=1, series_id=10, instance_number=100
        records_df = setup_dataframe[
            (setup_dataframe["study_id"] == 1) &
            (setup_dataframe["series_id"] == 10) &
            (setup_dataframe["instance_number"] == 100)
        ]

        # Call the function
        result = dataset_instance._serialize_payload(records_df)

        # Verify the result is a bytes object of the correct length
        assert isinstance(result, bytes)
        assert len(result) == 16  # 2 records of 8 bytes each

        # Verify the payload content for the first record
        level1 = int.from_bytes(result[0:1], byteorder='big')
        severity1 = int.from_bytes(result[1:2], byteorder='big')
        x1 = int.from_bytes(result[2:5], byteorder='big')
        y1 = int.from_bytes(result[5:8], byteorder='big')

        assert level1 == 1
        assert severity1 == 2
        assert x1 == 123  # 1.23 * 100
        assert y1 == 234  # 2.34 * 100

        # Verify the payload content for the second record
        level2 = int.from_bytes(result[8:9], byteorder='big')
        severity2 = int.from_bytes(result[9:10], byteorder='big')
        x2 = int.from_bytes(result[10:13], byteorder='big')
        y2 = int.from_bytes(result[13:16], byteorder='big')

        assert level2 == 2
        assert severity2 == 1
        assert x2 == 456  # 4.56 * 100
        assert y2 == 567  # 5.67 * 100

    def test_serialize_payload_null_raises_exception(
                                            self,
                                            setup_dataframe: pd.DataFrame,
                                            dataset_instance: LumbarDicomTFRecordDataset
                                        ) -> None:
        """
        Tests the payload's internal check for null values in critical columns.
        Ensures a ValueError is raised when nulls are found in PAYLOAD_COLS.
        """
        # 1. Create a records_df with null values in a critical payload column ('level')
        records_df_null = setup_dataframe[
            (setup_dataframe["study_id"] == 1) &
            (setup_dataframe["series_id"] == 10) &
            (setup_dataframe["instance_number"] == 100)
        ].copy()

        # Introduce a null value in 'level'
        records_df_null.loc[records_df_null.index[0], 'level'] = pd.NA

        # The expected error message pattern
        expected_cols = "['level']"
        expected_message = (
            f"Cannot serialize payload: "
            f"Null values detected in critical payload columns: {expected_cols}. "
            f"Serialization requires all payload values to be non-null."
        )

        # 2. Assert that the ValueError is raised with the correct message
        # Use re.escape() to match the literal string exactly
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            dataset_instance._serialize_payload(records_df_null)

    @pytest.mark.parametrize(
                                "invalid_df",
                                [None, pd.DataFrame({})], ids=["None", "Empty DataFrame"]
                             )
    def test_serialize_payload_none_or_empty_raises_exception(
                                            self,
                                            dataset_instance: LumbarDicomTFRecordDataset,
                                            invalid_df: Any
                                        ) -> None:
        """
        Tests the initial check for None or empty DataFrame in _serialize_payload.
        Ensures a ValueError is raised when the input is None or an empty DataFrame.

        This covers the validation block:
        if records_df is None or records_df.empty:
            raise ValueError("Cannot serialize payload: Input DataFrame is None or empty.")
        """
        expected_message = "Cannot serialize payload: Input DataFrame is None or empty."

        # Assert that the ValueError is raised with the correct message
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            dataset_instance._serialize_payload(invalid_df)

    def test_deserialize_metadata_successfull(self, mock_setup, tmp_path):
        """
            Test the complete metadata deserialization process.
        """
        mock_config, mock_logger = mock_setup

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

        logger_path = "src.core.utils.logger.get_current_logger"
        with patch(logger_path, return_value=mock_logger):

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

            # Execute
            result = dataset._deserialize_metadata(metadata_bytes, logger=mock_logger)

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

    def test_deserialize_metadata_empty(self, mock_setup, tmp_path):
        """Test deserialization with empty byte sequence."""
        mock_config, mock_logger = mock_setup

        logger_path = "src.core.utils.logger.get_current_logger"
        with patch(logger_path, return_value=mock_logger):

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

            # Execute & Assert
            with pytest.raises(ValueError, match="Input byte sequence is empty"):
                dataset._deserialize_metadata(b'')

    def test_deserialize_metadata_invalid_input(self, mock_setup, tmp_path):
        """Test deserialization with invalid input."""
        mock_config, mock_logger = mock_setup

        logger_path = "src.core.utils.logger.get_current_logger"
        with patch(logger_path, return_value=mock_logger):
            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(mock_config,
                                                 logger=mock_logger)

            # Test with empty byte sequence
            with pytest.raises(ValueError, match="Input byte sequence is empty"):
                dataset._deserialize_metadata(b'')

            # Test with non-byte sequence input
            with pytest.raises(ValueError, match="Input must be a byte sequence"):
                dataset._deserialize_metadata("not a byte sequence")

            # Test with insufficient buffer length
            with pytest.raises(struct.error, match="Invalid buffer length"):
                dataset._deserialize_metadata(b'\x00\x00')  # Too short buffer

    def test_deserialize_metadata_too_short(self, mock_setup, tmp_path):
        """Test deserialization with too short byte sequence."""

        mock_config, mock_logger = mock_setup

        logger_path = "src.core.utils.logger.get_current_logger"
        with patch(logger_path, return_value=mock_logger):

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(mock_config,
                                                 logger=mock_logger)

            # Execute & Assert
            with pytest.raises(struct.error):
                dataset._deserialize_metadata(b'\x00\x00\x00\x00\x01',
                                              logger=mock_logger)

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

    def test_deserialize_header_successful(self, mock_setup, tmp_path):
        """Test header deserialization."""

        mock_config, mock_logger = mock_setup

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
        with patch(logger_path, return_value=mock_logger):

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(mock_config,
                                                 logger=mock_logger)

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

    def test_deserialize_records_successful(self, mock_setup, tmp_path):
        """Test records deserialization."""

        mock_config, mock_logger = mock_setup

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
        with patch(logger_path, return_value=mock_logger):

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(mock_config,
                                                 logger=mock_logger)

            # Execute
            result = dataset._deserialize_records(buffer, 2)

            # Assert
            assert result == [
                (1, 2, 20.00, 50.00),
                (3, 1, 40.00, 79.04)
            ]

    def test_deserialize_records_empty(self, mock_setup, tmp_path):
        """Test records deserialization with zero records."""

        mock_config, mock_logger = mock_setup

        # Setup
        buffer = io.BytesIO(b'')
        nb_records = 0

        logger_path = "src.core.utils.logger.get_current_logger"
        with patch(logger_path, return_value=mock_logger):

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(mock_config,
                                                 logger=mock_logger)

            # Execute
            result = dataset._deserialize_records(buffer, nb_records)

            # Assert
            assert result == []

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

    def test_full_serialization_deserialization_cycle(
                                                        self,
                                                        setup_dataframe: pd.DataFrame,
                                                        dataset_instance: LumbarDicomTFRecordDataset
                                                       ) -> None:
        """
            Tests the complete serialization -> deserialization cycle (serialize-deserialize)
            to ensure data integrity field by field for a multi-record input.

            Steps:
            1. Select multiple records (multi-row DataFrame).
            2. Serialize the records (_serialize_payload).
            3. Deserialize the result (_deserialize_payload).
            4. Compare the initial DataFrame with the deserialized DataFrame.
        """
        # 1. Prepare the records (a multi-row DataFrame)
        mask = (
                    (setup_dataframe["study_id"] == 1) &
                    (setup_dataframe["series_id"] == 10) &
                    (setup_dataframe["instance_number"] == 100)
                )

        record_df = setup_dataframe[mask]

        # 2. Serialize the records (one feature dictionary output for the image)
        serialized_features = dataset_instance._serialize_metadata(record_df)

        # Verify that serialization produced a bytes object
        assert isinstance(serialized_features, bytes)

        # Verify the length of the serialized bytes
        assert len(serialized_features) == 31

        # 3. Deserialize the record
        # Pass the single feature dictionary output of serialization wrapped in a list
        deserialized_dict = dataset_instance._deserialize_metadata(serialized_features)
        deserialized_df = pd.DataFrame(deserialized_dict)

        # 4. Preparation for comparison: Ensure indices are reset and columns are correctly ordered
        deserialized_df = deserialized_df.reset_index(drop=True)

        # 5. Compare the original and deserialized DataFrames field by field
        assert deserialized_dict['nb_records'] == 2, (
            f"Expected 'nb_records' to be 2, but got {deserialized_dict['nb_records']}"
        )
        assert record_df.study_id.to_list() == deserialized_df.study_id.to_list(), (
            f"Study IDs do not match. Expected {record_df.study_id.to_list()}, "
            f"but got {deserialized_df.study_id.to_list()}"
        )
        assert record_df.series_id.to_list() == deserialized_df.series_id.to_list(), (
            f"Series IDs do not match. Expected {record_df.series_id.to_list()}, "
            f"but got {deserialized_df.series_id.to_list()}"
        )
        assert record_df.instance_number.to_list() == deserialized_df.instance_number.to_list(), (
            f"Instance Numbers do not match. Expected {record_df.instance_number.to_list()}, "
            f"but got {deserialized_df.instance_number.to_list()}"
        )
        assert record_df.condition.to_list() == deserialized_df.condition.to_list(), (
            f"Conditions do not match. Expected {record_df.condition.to_list()}, "
            f"but got {deserialized_df.condition.to_list()}"
        )
        assert record_df.series_description.to_list() == deserialized_df.description.to_list(), (
            f"Series Descriptions do not match. Expected {record_df.series_description.to_list()}, "
            f"but got {deserialized_df.description.to_list()}"
        )

        # Pre-calculate lists for the payload assertions to keep lines under 100 chars
        nb_records = deserialized_dict['nb_records']

        def get_records(idx, key):
            """
                Retrieves a specific value 'key' from the record at index 'idx'
                in the 'records' column of the deserialized DataFrame.
            """
            return deserialized_df.records.iloc[idx][key]

        level_list = [get_records(idx, 0) for idx in range(nb_records)]
        assert record_df.level.tolist() == level_list, (
            f"Level data does not match. Expected {record_df.level.tolist()}, "
            f"but got {level_list}"
        )

        severity_list = [get_records(idx, 1) for idx in range(nb_records)]
        assert record_df.severity.tolist() == severity_list, (
            f"Severity data does not match. Expected {record_df.severity.tolist()}, "
            f"but got {severity_list}"
        )

        x_list = [get_records(idx, 2) for idx in range(nb_records)]
        assert record_df.x.tolist() == x_list, (
            f"X coordinates do not match. Expected {record_df.x.tolist()}, "
            f"but got {x_list}"
        )

        y_list = [get_records(idx, 3) for idx in range(nb_records)]
        assert record_df.y.tolist() == y_list, (
            f"Y coordinates do not match. Expected {record_df.y.tolist()}, "
            f"but got {y_list}"
        )
