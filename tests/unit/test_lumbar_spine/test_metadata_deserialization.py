# coding: utf-8

import io
import struct
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import (
    LumbarDicomTFRecordDataset
)

class TestMetadataDeserialization:
    """
        Unit tests for metadata deserialization in LumbarDicomTFRecordDataset.
    """
    
    @pytest.fixture(autouse=True)
    def setup(self, mock_config, mock_logger):
        """Fixture to initialize common attributes for all tests."""
        
        self.mock_config = mock_config
        self.mock_logger = mock_logger
        self.root_dir = Path(self.mock_config["dicom_root_dir"])
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

        with patch("src.core.utils.logger.get_current_logger",
                                            return_value=self.mock_logger):
            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                 logger=self.mock_logger)

            # Execute
            result = dataset._deserialize_metadata(metadata_bytes,
                                                        logger=logger)

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

        with patch("src.core.utils.logger.get_current_logger",
                                            return_value=self.mock_logger):
            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                 logger=self.mock_logger)

            # Execute & Assert
            with pytest.raises(ValueError,
                               match="Input byte sequence is empty"):
                dataset._deserialize_metadata(b'')

    def test_deserialize_metadata_invalid_input(self):
        """Test deserialization with invalid input."""
        
        with patch("src.core.utils.logger.get_current_logger",
                                            return_value=self.mock_logger):
            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                 logger=self.mock_logger)

            # Test with empty byte sequence
            with pytest.raises(ValueError,
                               match="Input byte sequence is empty"):
                dataset._deserialize_metadata(b'')

            # Test with non-byte sequence input
            with pytest.raises(ValueError,
                               match="Input must be a byte sequence"):
                dataset._deserialize_metadata("not a byte sequence")

            # Test with insufficient buffer length
            with pytest.raises(struct.error, match="Invalid buffer length"):
                dataset._deserialize_metadata(b'\x00\x00')  # Too short buffer


    def test_deserialize_metadata_too_short(self):
        """Test deserialization with too short byte sequence."""

        with patch("src.core.utils.logger.get_current_logger",
                                            return_value=self.mock_logger):
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

        with patch("src.core.utils.logger.get_current_logger",
                                            return_value=self.mock_logger):
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

        with patch("src.core.utils.logger.get_current_logger",
                                            return_value=self.mock_logger):
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

        with patch(
                    "src.core.utils.logger.get_current_logger",
                    return_value=self.mock_logger):

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                 logger=self.mock_logger)
            
            # Execute
            result = dataset._deserialize_records(buffer, nb_records)

            # Assert
            assert result == []
