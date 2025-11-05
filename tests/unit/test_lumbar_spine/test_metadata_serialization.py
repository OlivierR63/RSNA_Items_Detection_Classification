# coding: utf-8

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
from pathlib import Path
from typing import Tuple, Any


class TestMetadataSerialization:
    @pytest.fixture
    def setup_dataframe(self):
        """Fixture to create a sample DataFrame for testing."""
        data = {
                    "study_id": [1, 1, 2, 2],
                    "series_id": [10, 10, 20, 20],
                    "instance_number": [100, 100, 200, 200],
                    "condition": [1, 1, 2, 2],
                    "series_description": [3, 3, 4, 4],
                    "level": [1, 2, 1, 2],
                    "severity": [2, 1, 2, 1],
                    "x": [1.23, 4.56, 7.89, 0.12],
                    "y": [2.34, 5.67, 8.90, 1.23]
                }
        return pd.DataFrame(data)

    @pytest.fixture
    def dataset_instance(
                            self,
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
            instance.logger = mock_logger
            instance._config = mock_config
        return instance

    def test_serialize_metadata(
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

        # Test with valid data
        result = dataset_instance._serialize_metadata(
                                                        "1",
                                                        "10",
                                                        "100",
                                                        setup_dataframe,
                                                        logger=mock_logger
                                                        )

        # Verify the result is a non-empty bytes object
        assert isinstance(result, bytes)
        assert len(result) > 0

        # Verify the header part. Explanation:
        #      - Header is first 15 bytes
        #      - Payload follows : 2 records * 8 bytes each = 16 bytes
        #      Total = 15 + 16 = 31 bytes
        assert len(result) == 31

        # Verify logger calls
        msg_str = "Starting function _serialize_metadata"
        mock_logger.info.assert_any_call(msg_str)

        msg_str = "Function _serialize_metadata completed successfully"
        mock_logger.info.assert_called_with(
                                            msg_str,
                                            extra={"status": "success"}
                                        )

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

                # Mock _filter_records to return the non-empty DataFrame
                patch.object(
                                LumbarDicomTFRecordDataset,
                                '_filter_records',
                                return_value=mock_data_df
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
                                                study_id_str="123",
                                                series_id_str="456",
                                                instance_number_str="1",
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

    def test_filter_records(
                                self,
                                setup_dataframe: pd.DataFrame,
                                dataset_instance: LumbarDicomTFRecordDataset,
                                mock_setup: Tuple[dict[str, Any], MagicMock],
                                tmp_path: Path
                            ) -> None:
        """
            Test the record filtering function.
        """
        _, mock_logger = mock_setup

        # Test with matching records
        result = dataset_instance._filter_records(
                                                    setup_dataframe,
                                                    "1",
                                                    "10",
                                                    "100",
                                                    mock_logger
                                                   )
        assert len(result) == 2  # Should return 2 matching records

        # Test with no matching records
        result = dataset_instance._filter_records(
                                                    setup_dataframe,
                                                    "99",
                                                    "99",
                                                    "999",
                                                    mock_logger
                                                  )
        assert result.empty
        mock_logger.warning.assert_called()

    def test_serialize_header(
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
        result = dataset_instance._serialize_header(records_df, "1", "10", "100")

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

    def test_serialize_payload(
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
        assert len(result) == 16  # 2 records * 8 bytes each

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
        result = dataset_instance._serialize_metadata(
                                                        "99",
                                                        "99",
                                                        "999",
                                                        setup_dataframe,
                                                        logger=mock_logger
                                                      )

        # Verify the result is an empty bytes object
        assert result == b''

        # Verify logger calls
        mock_logger.warning.assert_called()

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
                'condition': [1] * 26,
                'series_description': [2] * 26,
                'level': [10] * 26,
                'severity': [20] * 26,
                'x': [0.1] * 26,
                'y': [0.2] * 26,
            }
            records_dataframe = pd.DataFrame(data)

            # 2. Define expected inputs (not strictly used for the failure, but for context)
            mock_study_id = "123"
            mock_series_id = "456"
            mock_instance_number = "7"

            # 3. Assert that the ValueError is raised
            expected_error_message = "The number of records exceeds the limit of 25."

            with pytest.raises(ValueError, match=expected_error_message):
                dataset_obj._serialize_header(
                                                records_dataframe,
                                                mock_study_id,
                                                mock_series_id,
                                                mock_instance_number
                                                )
