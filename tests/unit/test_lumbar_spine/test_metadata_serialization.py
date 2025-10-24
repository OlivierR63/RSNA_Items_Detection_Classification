# coding: utf-8

import pytest
import pandas as pd
import logging
from unittest.mock import MagicMock, patch 
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import (
    LumbarDicomTFRecordDataset
)


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
    def mock_logger(self):
        """Fixture to create a mock logger."""
        return MagicMock(spec=logging.Logger)

    @pytest.fixture
    def dataset_instance(self, mock_logger):
        """
            Fixture to create an instance of LumbarDicomTFRecordDataset
            with a mock logger.
        """
        config = {
                    "output_dir": "tests/tmp",
                    "batch_size": 8,
                    "epochs": 2,
                    "model_3d": {"type": "mock"}
                  }
        with patch.object(LumbarDicomTFRecordDataset, '__init__', return_value=None):
            instance = LumbarDicomTFRecordDataset.__new__(LumbarDicomTFRecordDataset)
            instance.logger = mock_logger
            instance._config = config
        return instance

    def test_serialize_metadata(
                        self, setup_dataframe, dataset_instance, mock_logger):
        """Test the main serialization function."""

        # Test with valid data
        result = dataset_instance._serialize_metadata(
                        "1", "10", "100", setup_dataframe, logger=mock_logger)

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

    def test_filter_records(
                    self, setup_dataframe, dataset_instance, mock_logger):
        """Test the record filtering function."""

        # Test with matching records
        result = dataset_instance._filter_records(
                            setup_dataframe, "1", "10", "100", mock_logger)
        assert len(result) == 2  # Should return 2 matching records

        # Test with no matching records
        result = dataset_instance._filter_records(
                            setup_dataframe, "99", "99", "999", mock_logger)
        assert result.empty
        mock_logger.warning.assert_called()

    def test_serialize_header(self, setup_dataframe, dataset_instance):
        """Test the header serialization function."""

        # Filter the DataFrame to get records for:
        # study_id=1, series_id=10, instance_number=100
        records_df = setup_dataframe[
            (setup_dataframe["study_id"] == 1) &
            (setup_dataframe["series_id"] == 10) &
            (setup_dataframe["instance_number"] == 100)
        ]

        # Call the function
        result = dataset_instance._serialize_header(
                                            records_df, "1", "10", "100")

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

    def test_serialize_payload(self, setup_dataframe, dataset_instance):
        """Test the payload serialization function."""

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
                        self, setup_dataframe, dataset_instance, mock_logger):
        """Test the main serialization function with no matching records."""

        # Test with non-matching data
        result = dataset_instance._serialize_metadata(
                       "99", "99", "999", setup_dataframe, logger=mock_logger)

        # Verify the result is an empty bytes object
        assert result == b''

        # Verify logger calls
        mock_logger.warning.assert_called()
