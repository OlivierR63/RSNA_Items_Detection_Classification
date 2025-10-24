# coding: utf-8

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import (
    LumbarDicomTFRecordDataset
)
from pathlib import Path
from typing import Any

# Constants for test values

CONDITIONS = [
    "Spinal Canal Stenosis",
    "Right Neural Foraminal Narrowing",
    "Left Neural Foraminal Narrowing",
    "Right Subarticular Stenosis",
    "Left Subarticular Stenosis"
]

LEVELS = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]

SERIES_DESCRIPTIONS = ["Axial T2", "Sagittal T1", "Sagittal T2/STIR"]

SEVERITIES = ["Normal/Mild", "Severe", "Moderate"]


class TestEncodeDataFrame:
    """
        Unit tests for the _encode_dataframe method
        in LumbarDicomTFRecordDataset.
    """

    @pytest.fixture(autouse=True)
    def setup(self, mock_config: dict[str, Any], mock_logger: MagicMock):
        """
            Fixture to initialize common attributes for all tests.
        """

        self.mock_config = mock_config
        self.mock_logger = mock_logger
        self.root_dir = Path(self.mock_config["dicom_root_dir"])
        self.output_dir = Path(self.mock_config["tfrecord_dir"])
        self.metadata_df = pd.DataFrame({
            "study_id": [1, 1, 2],
            "series_id": [1, 2, 1],
            "instance_number": [1, 1, 1],
            "file_path": ["path1", "path2", "path3"],
            "metadata": ["meta1", "meta2", "meta3"]
        })

    @pytest.fixture(autouse=True)
    def test_df(self):
        """
            Fixture to provide a test DataFrame with realistic values.
        """

        series_list = SERIES_DESCRIPTIONS + [SERIES_DESCRIPTIONS[0]] + [SERIES_DESCRIPTIONS[1]]
        severities_list = SEVERITIES + [SEVERITIES[1]] + [SEVERITIES[0]]
        
        data_dict = {
            "condition": CONDITIONS,
            "level": LEVELS,
            "series_description": series_list,
            "severity": severities_list,
            "other_column": [1, 2, 3, 0, 4]
        }
        return pd.DataFrame(data_dict)

    def create_mock_mapper(self, values):
        # Create a mock mapping based on the input values
        mock_mapper = MagicMock()

        if not values:
            mock_mapper.mapping = {}
            return mock_mapper

        if any(val in CONDITIONS for val in values):
            mock_mapper.mapping = {condition: idx for idx, condition in enumerate(CONDITIONS)}

        elif any(val in LEVELS for val in values):
            mock_mapper.mapping = {level: idx for idx, level in enumerate(LEVELS)}

        elif any(val in SERIES_DESCRIPTIONS for val in values):
            mock_mapper.mapping = {desc: idx for idx, desc in enumerate(SERIES_DESCRIPTIONS)}

        elif any(val in SEVERITIES for val in values):
            mock_mapper.mapping = {severity: idx for idx, severity in enumerate(SEVERITIES)}

        else:
            mock_mapper.mapping = {value: idx for idx, value in enumerate(set(values))}

        return mock_mapper

    def test_encode_dataframe(self, test_df: dict[str, Any]):
        """
            Test the complete encoding process of a DataFrame with realistic values
        """

        # Mock the _create_string_to_int_mapper method
        csv_metadata_chain = (
                    "src.projects.lumbar_spine."
                    "lumbar_dicom_tfrecord_dataset.CSVMetadata"
        )

        with (
            patch.object(
                LumbarDicomTFRecordDataset,
                '_create_string_to_int_mapper',
                side_effect=self.create_mock_mapper
            ),
            patch(
                "src.core.utils.logger.get_current_logger",
                return_value=self.mock_logger
            ),
            patch(csv_metadata_chain) as mock_csv_metadata_class
        ):

            # Configure the mock CSVMetadata class
            mock_csv_metadata_instance = MagicMock()
            mock_csv_metadata_class.return_value = mock_csv_metadata_instance
            mock_csv_metadata_instance._merged_df = test_df

            # Mock get_current_logger to avoid side effects
            with patch("src.core.utils.logger.get_current_logger", return_value=self.mock_logger):

                # Initialize the dataset
                dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                     logger=self.mock_logger)

                # Call the method
                result_df = dataset._encode_dataframe(test_df)

                # Assertions
                # Check that categorical columns have been converted to integers
                condition_col = result_df["condition"]
                level_col = result_df["level"]
                series_desc_col = result_df["series_description"]
                severity_col = result_df["severity"]

                assert all(isinstance(x, int) for x in condition_col)
                assert all(isinstance(x, int) for x in level_col)
                assert all(isinstance(x, int) for x in series_desc_col)
                assert all(isinstance(x, int) for x in severity_col)

                # Check that non-categorical columns remain unchanged
                assert list(result_df["other_column"]) == [1, 2, 3, 0, 4]

                # Check specific values to ensure correct encoding
                # Spinal Canal Stenosis -> 0.
                assert result_df.iloc[0]["condition"] == 0

                # Right Neural Foraminal Narrowing -> 1
                assert result_df.iloc[1]["condition"] == 1

                # Left Neural Foraminal Narrowing -> 2
                assert result_df.iloc[2]["condition"] == 2

                # Right Subarticular Stenosis -> 3
                assert result_df.iloc[3]["condition"] == 3

                # Left Subarticular Stenosis -> 4
                assert result_df.iloc[4]["condition"] == 4

                # L1-L2 -> 0
                assert result_df.iloc[0]["level"] == 0

                # L3-L4 -> 1
                assert result_df.iloc[1]["level"] == 1

                # L2-L3 -> 2
                assert result_df.iloc[2]["level"] == 2

                # L5-S1 -> 3
                assert result_df.iloc[3]["level"] == 3

                # L4-L5 -> 4
                assert result_df.iloc[4]["level"] == 4

                # Axial T2 -> 0
                assert result_df.iloc[0]["series_description"] == 0

                # Sagittal T1 -> 1
                assert result_df.iloc[1]["series_description"] == 1

                # Sagittal T2/STIR -> 2
                assert result_df.iloc[2]["series_description"] == 2

                # Axial T2 -> 0
                assert result_df.iloc[3]["series_description"] == 0

                # Sagittal T1 -> 1
                assert result_df.iloc[4]["series_description"] == 1

                # Normal/Mild -> 0
                assert result_df.iloc[0]["severity"] == 0

                # Severe -> 1
                assert result_df.iloc[1]["severity"] == 1

                # Moderate -> 2
                assert result_df.iloc[2]["severity"] == 2

                # Severe -> 1
                assert result_df.iloc[3]["severity"] == 1

                # Normal/Mild -> 0
                assert result_df.iloc[4]["severity"] == 0

    def test_create_mappings(self, test_df: dict[str, Any]):
        """
            Test the creation of mapping dictionaries with realistic values.
        """

        columns_to_encode = ["condition", "level"]

        # Mock the _create_string_to_int_mapper method
        csv_metadata_chain = "src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset.CSVMetadata"
        with (
            patch.object(
                LumbarDicomTFRecordDataset,
                '_create_string_to_int_mapper',
                side_effect=self.create_mock_mapper
            ),
            patch(
                "src.core.utils.logger.get_current_logger",
                return_value=self.mock_logger
            ),
            patch(csv_metadata_chain) as mock_csv_metadata_class
        ):

            # Configure the mock CSVMetadata class
            mock_csv_metadata_instance = MagicMock()
            mock_csv_metadata_class.return_value = mock_csv_metadata_instance
            mock_csv_metadata_instance._merged_df = test_df

            # Mock _generate_tfrecord_files to avoid side effects
            with patch.object(LumbarDicomTFRecordDataset, '_generate_tfrecord_files'):

                # Initialize the dataset
                dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                     logger=self.mock_logger)

                # Call the method
                mappings = dataset._create_mappings(test_df,
                                                    columns_to_encode)

                # Assertions
                assert "condition" in mappings
                assert "level" in mappings

                expected_cond_map = {condition: idx for idx, condition in enumerate(CONDITIONS)}
                assert mappings["condition"] == expected_cond_map

                assert mappings["level"] == {level: idx for idx, level in enumerate(LEVELS)}

    def test_apply_encodings(self, test_df: dict[str, Any]):
        """
            Test the application of encoding mappings to a DataFrame
            with realistic values.
        """

        columns_to_encode = ["condition", "level"]

        # Define realistic mappings
        mappings = {
            "condition": {condition: idx for idx, condition in enumerate(CONDITIONS[:2])},
            "level": {level: idx for idx, level in enumerate(LEVELS[1:3])}
        }

        get_current_logger_chain = "src.core.utils.logger.get_current_logger"
        with patch(get_current_logger_chain, return_value=self.mock_logger):
            # Initialize the dataset
            dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                 logger=self.mock_logger)

            # Call the method on the instance
            result_df = dataset._apply_encodings(test_df,
                                                 columns_to_encode, mappings)

            # Assertions
            assert list(result_df["condition"]) == [0, 1, -1, -1, -1]
            assert list(result_df["level"]) == [-1, 0, 1, -1, -1]
            assert all(isinstance(x, int) for x in result_df["condition"])
            assert all(isinstance(x, int) for x in result_df["level"])

    def test_create_string_to_int_mapper(self):
        """Test the creation of a string to integer mapper."""
        # Test data with all expected values
        # Duplicate "Spinal Canal Stenosis" to test consistency
        test_values = CONDITIONS + ["Spinal Canal Stenosis"]

        # Mock _generate_tfrecord_files to avoid side effects
        get_current_logger_chain = "src.core.utils.logger.get_current_logger"
        with (
                patch.object(LumbarDicomTFRecordDataset, '_generate_tfrecord_files'),
                patch(get_current_logger_chain, return_value=self.mock_logger)
        ):

            # Initialize the dataset
            dataset = LumbarDicomTFRecordDataset(self.mock_config,
                                                 logger=self.mock_logger)

            # Call the method
            mapper = dataset._create_string_to_int_mapper(test_values)

            # Verifications
            # Check that the mapper is a dictionary-like object
            assert hasattr(mapper, 'mapping')
            assert isinstance(mapper.mapping, dict)

            # Check that all test values are in the mapping
            assert set(mapper.mapping.keys()) == set(CONDITIONS)
            assert(len(set(mapper.mapping.keys()))) == len(CONDITIONS)

            # Check that each value is an integer
            assert all(isinstance(value, int) for value in mapper.mapping.values())

            # Test the get method with existing keys
            for key in CONDITIONS:
                assert mapper.mapping.get(key) == mapper.mapping[key]

            # Test the get method with a non-existent key
            assert mapper.mapping.get("NonExistentValue") is None
