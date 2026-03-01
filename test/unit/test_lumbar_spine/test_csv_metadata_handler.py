import pytest
import pandas as pd
import numpy as np
import logging
from src.projects.lumbar_spine.csv_metadata_handler import CSVMetadataHandler
from unittest.mock import patch


# class TestCSVMetadataHandler(CSVMetadataHandler):
#     def __init__(self, series_description, label_coordinates, train, mock_mock_logger):
#         super().__init__(series_description, label_coordinates, train, mock_logger)
#         # Force columns to be strings
#         self._series_desc_df = self._series_desc_df.astype(str)
#         self._label_coords_df = self._label_coords_df.astype(str)
#         self._train_df = self._train_df.astype(str)
#         self._merged_df = self._merged_df.astype(str)

def test_init(mock_config, setup_csv_files, mock_logger):
    """
    Test the initialization of CSVMetadataHandler.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        "",
        files["description"],
        files["label_coordinates"],
        files["train"],
        mock_logger
    )

    # Check that the DataFrames are loaded
    assert isinstance(csv_metadata_handler._series_desc_df, pd.DataFrame)
    assert isinstance(csv_metadata_handler._label_coords_df, pd.DataFrame)
    assert isinstance(csv_metadata_handler._train_df, pd.DataFrame)
    assert isinstance(csv_metadata_handler._merged_df, pd.DataFrame)


def test_init_exception(setup_csv_files, mock_logger, caplog):
    """
    Test that exceptions during initialization are handled and logged.
    """
    files = setup_csv_files

    # Mock pd.read_csv to raise an exception
    with patch('pandas.read_csv', side_effect=Exception("File not found")):
        with pytest.raises(Exception):
            _ = CSVMetadataHandler(
                "",
                files["description"],
                files["label_coordinates"],
                files["train"],
                mock_logger
            )

    # Check that the error was logged
    assert "Error initializing CSVMetadataHandler" in caplog.text


def test_merged_property(setup_csv_files, mock_logger):
    """
    Test the merged property.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        "",
        files["description"],
        files["label_coordinates"],
        files["train"],
        mock_logger
    )

    # Check that the merged property returns a DataFrame
    assert isinstance(csv_metadata_handler.merged, pd.DataFrame)


def test_train_df_property(setup_csv_files, mock_logger):
    """
    Test the train_df property.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        "",
        files["description"],
        files["label_coordinates"],
        files["train"],
        mock_logger
    )

    # Check that the train_df property returns a DataFrame
    assert isinstance(csv_metadata_handler.train_df, pd.DataFrame)


def test_merge_metadata(setup_csv_files, mock_logger):
    """
    Test the _merge_metadata method.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        "",
        files["description"],
        files["label_coordinates"],
        files["train"],
        mock_logger
    )

    # Call the private method _merge_metadata
    merged_df = csv_metadata_handler._merge_metadata()

    # Check that the result is a DataFrame
    assert isinstance(merged_df, pd.DataFrame)


def test_melt_and_clean_train_df(setup_csv_files, mock_logger):
    """
    Test the _melt_and_clean_train_df method.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        "",
        files["description"],
        files["label_coordinates"],
        files["train"],
        mock_logger
    )

    # Call the private method _melt_and_clean_train_df
    melted_df = csv_metadata_handler._melt_and_clean_train_df()

    # Check that the result is a DataFrame
    assert isinstance(melted_df, pd.DataFrame)

    # Check that the DataFrame has the expected columns
    assert 'study_id' in melted_df.columns
    assert 'condition' in melted_df.columns
    assert 'level' in melted_df.columns
    assert 'severity' in melted_df.columns


def test_melt_and_clean_train_df_exception(setup_csv_files, mock_logger, caplog):
    """
    Test that exceptions in _melt_and_clean_train_df are handled and logged.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        "",
        files["description"],
        files["label_coordinates"],
        files["train"],
        mock_logger
    )

    # Mock the melt method of the existing _train_df DataFrame to raise an exception
    with patch.object(csv_metadata_handler._train_df, 'melt', side_effect=Exception("Melt error")):
        with pytest.raises(Exception):
            csv_metadata_handler._melt_and_clean_train_df()

    # Check that the error was logged with the correct message
    assert "Error melting and cleaning training DataFrame" in caplog.text


def test_merge_with_label_coordinates(setup_csv_files, mock_logger):
    """
    Test the _merge_with_label_coordinates method.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        "",
        files["description"],
        files["label_coordinates"],
        files["train"],
        mock_logger
    )

    # Call the private method _melt_and_clean_train_df
    melted_df = csv_metadata_handler._melt_and_clean_train_df()

    # Call the private method _merge_with_label_coordinates
    merged_df = csv_metadata_handler._merge_with_label_coordinates(melted_df)

    # Check that the result is a DataFrame
    assert isinstance(merged_df, pd.DataFrame)

    # Check that the DataFrame has the expected columns
    assert 'x' in merged_df.columns
    assert 'y' in merged_df.columns


def test_merge_with_series_descriptions(setup_csv_files, mock_logger):
    """
    Test the _merge_with_series_descriptions method.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        "",
        files["description"],
        files["label_coordinates"],
        files["train"],
        mock_logger
    )

    # Call the private method _melt_and_clean_train_df
    melted_df = csv_metadata_handler._melt_and_clean_train_df()

    # Call the private method _merge_with_label_coordinates
    merged_df = csv_metadata_handler._merge_with_label_coordinates(melted_df)

    # Call the private method _merge_with_series_descriptions
    final_merged_df = csv_metadata_handler._merge_with_series_descriptions(merged_df)

    # Check that the result is a DataFrame
    assert isinstance(final_merged_df, pd.DataFrame)

    # Check that the DataFrame has the expected columns
    assert 'description' in final_merged_df.columns


def test_merge_metadata_exception(setup_csv_files, mock_logger, caplog):
    """
    Test that exceptions in _merge_metadata are handled and logged.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        "",
        files["description"],
        files["label_coordinates"],
        files["train"],
        mock_logger
    )

    # Mock the _melt_and_clean_train_df to raise an exception
    with patch.object(
                        csv_metadata_handler,
                        '_melt_and_clean_train_df',
                        side_effect=Exception("Merge error")
                       ):
        with pytest.raises(Exception):
            csv_metadata_handler._merge_metadata()

    # Check that the error was logged
    assert "Error merging metadata" in caplog.text


def test_normalize_identifier_types(setup_csv_files, mock_logger):
    """
    Test the _normalize_identifier_types method.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        "",
        files["description"],
        files["label_coordinates"],
        files["train"],
        mock_logger
    )

    # Call the private method _melt_and_clean_train_df
    melted_df = csv_metadata_handler._melt_and_clean_train_df()

    # Call the private method _merge_with_label_coordinates
    merged_df = csv_metadata_handler._merge_with_label_coordinates(melted_df)

    # Call the private method _merge_with_series_descriptions
    final_merged_df = csv_metadata_handler._merge_with_series_descriptions(merged_df)

    # Call the private method _normalize_identifier_types
    normalized_df = csv_metadata_handler._normalize_identifier_types(final_merged_df)

    # Check that the result is a DataFrame
    assert isinstance(normalized_df, pd.DataFrame)

    # Check that the identifier columns are of the correct type
    assert normalized_df['study_id'].dtype == np.int64
    assert normalized_df['series_id'].dtype == np.int64
    assert normalized_df['instance_number'].dtype == int
