# coding: utf-8

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
from pathlib import Path
from typing import Tuple, Any

# Constants for test values (tuples to prevent accidental modifications)

CONDITIONS = (
    "Spinal Canal Stenosis",
    "Right Neural Foraminal Narrowing",
    "Left Neural Foraminal Narrowing",
    "Right Subarticular Stenosis",
    "Left Subarticular Stenosis"
)
LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")
SERIES_DESCRIPTIONS = ("Axial T2", "Sagittal T1", "Sagittal T2/STIR")
SEVERITIES = ("Normal/Mild", "Severe", "Moderate")


@pytest.fixture
def mock_setup(mock_config, mock_logger):
    """
        Fixture to initialize attributes common to all tests.
    """

    # Mock the get_current_logger function to return the mock_logger
    with patch("src.core.utils.logger.get_current_logger", return_value=mock_logger):
        with patch.object(LumbarDicomTFRecordDataset,
                          '_generate_tfrecord_files', return_value=None):
            yield mock_config, mock_logger


@pytest.fixture
def test_df():
    """
        Fixture to provide a test DataFrame with realistic values.
    """

    series_list = SERIES_DESCRIPTIONS + (SERIES_DESCRIPTIONS[0],) + (SERIES_DESCRIPTIONS[1],)
    severities_list = SEVERITIES + (SEVERITIES[1],) + (SEVERITIES[0],)

    data_dict = {
        "condition": CONDITIONS,
        "level": LEVELS,
        "series_description": series_list,
        "severity": severities_list,
        "other_column": [1, 2, 3, 0, 4]
    }
    return pd.DataFrame(data_dict)


class TestEncodeDataFrame:
    """
        Unit tests for the _encode_dataframe method in LumbarDicomTFRecordDataset.
    """

    def _create_mock_mapper(self, values):
        """
            Helper function : creates a mock mapper for string-to-integer encoding
            based on input values.
        """
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

    def test_encode_dataframe(
                                self,
                                mock_setup: Tuple[dict[str, Any], MagicMock],
                                test_df: pd.DataFrame
                              ) -> None:
        """
            Test the complete encoding process of a DataFrame with realistic values
            Verifies that categorical columns are converted to integers and non-categorical
            columns remain unchanged.
        """

        mock_config, mock_logger = mock_setup

        # Mock the _create_string_to_int_mapper method
        csv_metadata_chain = (
                                "src.projects.lumbar_spine."
                                "lumbar_dicom_tfrecord_dataset.CSVMetadata"
        )

        with (
                patch.object(
                    LumbarDicomTFRecordDataset,
                    '_create_string_to_int_mapper',
                    side_effect=self._create_mock_mapper
                ),

                patch(csv_metadata_chain) as mock_csv_metadata_class
             ):

            # Configure the mock CSVMetadata class
            mock_csv_metadata_instance = MagicMock()
            mock_csv_metadata_class.return_value = mock_csv_metadata_instance
            mock_csv_metadata_instance._merged_df = test_df

            # Initialize the dataset
            dataset = LumbarDicomTFRecordDataset(
                                                    mock_config,
                                                    logger=mock_logger
            )

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

    def test_encode_dataframe_exception(
                                            self,
                                            mock_setup: Tuple[dict[str, Any], MagicMock]
                                        ) -> None:
        """
            Tests that _encode_dataframe correctly handles exceptions raised by its dependencies
            (e.g., _create_mappings or _apply_encodings), logs the error, and re-raises it.
        """
        mock_config, mock_logger = mock_setup

        # Initialiser l'objet
        dataset_obj = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # Créer un DataFrame d'entrée (le contenu n'a pas d'importance ici)
        input_df = pd.DataFrame({"condition": ["A"]})

        # Définir l'erreur que nous allons simuler
        simulated_error_message = "Mocked encoding failure"

        # Patch _create_mappings pour lever une exception
        with patch.object(dataset_obj, '_create_mappings',
                          side_effect=Exception(simulated_error_message)):

            # 1. Vérifier que la fonction lève l'exception
            with pytest.raises(Exception) as excinfo:
                dataset_obj._encode_dataframe(input_df, logger=mock_logger)

            # 2. Vérifier que l'exception levée correspond à l'exception simulée
            assert str(excinfo.value) == simulated_error_message

            # 3. Vérifier les appels de logging

            # 3.1 Vérifier le message de début
            mock_logger.info.assert_any_call("Starting function _encode_dataframe")

            # 3.2 Vérifier que l'erreur a été loggée correctement
            expected_error_msg = f"Error in function _encode_dataframe: {simulated_error_message}"
            mock_logger.error.assert_called_with(
                expected_error_msg,
                exc_info=True,
                extra={"status": "failed", "error": simulated_error_message}
            )

    def test_create_mappings(
                                self,
                                mock_setup: Tuple[dict[str, Any], MagicMock],
                                test_df: dict[str, Any]
                             ) -> None:
        """
            Test the creation of mapping dictionaries with realistic values.
        """

        mock_config, mock_logger = mock_setup

        columns_to_encode = ["condition", "level"]

        # Mock the _create_string_to_int_mapper method
        csv_metadata_chain = "src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset.CSVMetadata"
        with (
                patch.object(
                    LumbarDicomTFRecordDataset,
                    '_create_string_to_int_mapper',
                    side_effect=self._create_mock_mapper
                ),
                patch(csv_metadata_chain) as mock_csv_metadata_class
              ):

            # Configure the mock CSVMetadata class
            mock_csv_metadata_instance = MagicMock()
            mock_csv_metadata_class.return_value = mock_csv_metadata_instance
            mock_csv_metadata_instance._merged_df = test_df

            # Initialize the dataset
            dataset = LumbarDicomTFRecordDataset(mock_config,
                                                 logger=mock_logger)

            # Call the method
            mappings = dataset._create_mappings(test_df,
                                                columns_to_encode)

            # Assertions
            assert "condition" in mappings
            assert "level" in mappings

            expected_cond_map = {condition: idx for idx, condition in enumerate(CONDITIONS)}
            assert mappings["condition"] == expected_cond_map

            assert mappings["level"] == {level: idx for idx, level in enumerate(LEVELS)}

    def test_apply_encodings(
                                self,
                                mock_setup: Tuple[dict[str, Any], MagicMock],
                                test_df: dict[str, Any]
                              ) -> None:
        """
            Test the application of encoding mappings to a DataFrame
            with realistic values.
        """

        mock_config, mock_logger = mock_setup
        columns_to_encode = ["condition", "level"]

        # Define realistic mappings
        mappings = {
            "condition": {condition: idx for idx, condition in enumerate(CONDITIONS[:2])},
            "level": {level: idx for idx, level in enumerate(LEVELS[1:3])}
        }

        # Initialize the dataset
        dataset = LumbarDicomTFRecordDataset(mock_config,
                                             logger=mock_logger)

        # Call the method on the instance
        result_df = dataset._apply_encodings(test_df,
                                             columns_to_encode, mappings)

        # Assertions
        assert list(result_df["condition"]) == [0, 1, -1, -1, -1]
        assert list(result_df["level"]) == [-1, 0, 1, -1, -1]
        assert all(isinstance(x, int) for x in result_df["condition"])
        assert all(isinstance(x, int) for x in result_df["level"])

    def test_create_string_to_int_mapper(
                                            self,
                                            mock_setup: Tuple[dict[str, Any], MagicMock]
                                         ) -> None:
        """
            Test the creation of a string to integer mapper.
        """

        mock_config, mock_logger = mock_setup

        # Test data with all expected values
        # Duplicate "Spinal Canal Stenosis" to test consistency
        test_values = CONDITIONS + ("Spinal Canal Stenosis",)

        # Initialize the dataset
        dataset = LumbarDicomTFRecordDataset(mock_config,
                                             logger=mock_logger)

        # Call the method
        mapper = dataset._create_string_to_int_mapper(test_values)

        # Verifications
        # Check that the mapper is a dictionary-like object
        assert hasattr(mapper, 'mapping')
        assert isinstance(mapper.mapping, dict)

        # Check that all test values are in the mapping
        assert set(mapper.mapping.keys()) == set(CONDITIONS)
        assert len(set(mapper.mapping.keys())) == len(CONDITIONS)

        # Check that each value is an integer
        assert all(isinstance(value, int) for value in mapper.mapping.values())

        # Test the get method with existing keys
        for key in CONDITIONS:
            assert mapper.mapping.get(key) == mapper.mapping[key]

        # Test the get method with a non-existent key
        assert mapper.mapping.get("NonExistentValue") is None

    def test_create_string_to_int_mapper_exception(
        self,
        mock_setup: Tuple[dict[str, Any], MagicMock],
        tmp_path: Path
    ) -> None:
        """
            Tests that _create_string_to_int_mapper correctly handles an Exception
            (specifically a TypeError from unhashable elements) by logging the error
            details and re-raising the original exception.
        """
        mock_config, mock_logger = mock_setup

        # Initialize the dataset object
        dataset_obj = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # Define input data that will cause a TypeError when attempting to create a dict.
        # A list cannot be used as a dictionary key (it's unhashable).
        input_data = ['valid_key', ['unhashable_key'], 'another_valid_key']

        # Use pytest.raises to assert that the function re-raises the exception
        # (TypeError in this case).
        with pytest.raises(TypeError) as excinfo:
            dataset_obj._create_string_to_int_mapper(input_data, logger=mock_logger)

        # Retrieve the original error message from the raised exception
        error_message = str(excinfo.value)

        # Assertions:

        # Verify the starting info message was logged
        mock_logger.info.assert_any_call("Starting function _create_string_to_int_mapper")

        # Verify that the error was logged correctly using the format from the except block
        expected_error_message = f"Error in function _create_string_to_int_mapper : {error_message}"

        mock_logger.error.assert_called_with(
                                                expected_error_message,
                                                exc_info=True,
                                                # The 'error' key in extra must match the simple
                                                # string representation of the error
                                                extra={"status": "failed", "error": error_message}
                                            )

        # Verify the type of exception re-raised is the expected TypeError
        assert excinfo.type is TypeError

    def test_create_string_to_int_mapper_unknown_key(
        self,
        mock_setup: Tuple[dict[str, Any], MagicMock],
        tmp_path: Path
    ) -> None:

        """
            Tests the case in _create_string_to_int_mapper where the mapper function
            is called with a string not present in the original list, ensuring it returns -1.
            This covers the 'return mapping.get(key, -1)' line's default case.
        """
        mock_config, mock_logger = mock_setup

        # Initialize the dataset object
        dataset_obj = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # 1. Define input strings and call the function to get the mapper
        input_strings = ['Normal', 'Stenosis', 'Degeneration']

        # Assuming _create_string_to_int_mapper is a method of dataset_obj
        mapper_func = dataset_obj._create_string_to_int_mapper(input_strings, logger=mock_logger)

        # 2. Test for an unknown key to cover the missed branch (-1)
        # The string 'UnknownCondition' should return -1.
        unknown_key_result = mapper_func('UnknownCondition')

        # Assert that the default return value is correctly triggered.
        assert unknown_key_result == -1
