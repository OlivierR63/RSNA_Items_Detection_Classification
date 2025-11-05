import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import os
from pathlib import Path
from src.projects.lumbar_spine.csv_metadata import CSVMetadata
from unittest.mock import patch


# class TestCSVMetadata(CSVMetadata):
#     def __init__(self, series_description, label_coordinates, train, logger):
#         super().__init__(series_description, label_coordinates, train, logger)
#         # Force columns to be strings
#         self._series_desc_df = self._series_desc_df.astype(str)
#         self._label_coords_df = self._label_coords_df.astype(str)
#         self._train_df = self._train_df.astype(str)
#         self._merged_df = self._merged_df.astype(str)

@pytest.fixture
def setup_csv_files(tmp_path):
    # Create test CSV files
    series_description_path = tmp_path / "series_description.csv"
    label_coordinates_path = tmp_path / "label_coordinates.csv"
    train_path = tmp_path / "train.csv"

    # Write series_description.csv
    series_description_path.write_text(
        "study_id,series_id,series_description\n"
        "'1','1','description_1'\n"
        "'2','2','description_2'"
    )

    # Write label_coordinates.csv
    label_coordinates_path.write_text(
        "study_id,series_id,instance_number,condition,level,x,y\n"
        "'1','1','1','condition_1','level_1','10','20'\n"
        "'2','2','2','condition_2','level_2','30','40'"
    )

    # Write train.csv
    train_path.write_text(
        "study_id,condition1_level1,condition2_level2\n"
        "'1','severity_1','severity_2',\n"
        "'2','severity_3','severity_2'"
    )

    return {
        "series_description": str(series_description_path),
        "label_coordinates": str(label_coordinates_path),
        "train": str(train_path)
    }


@pytest.fixture
def logger(caplog):
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    return logger


def test_init(mock_config, setup_csv_files, logger):
    """
    Test the initialization of CSVMetadata.
    """
    files = setup_csv_files
    csv_metadata = CSVMetadata(
                                "",
                                files["series_description"],
                                files["label_coordinates"],
                                files["train"],
                                logger
                                )

    # Check that the DataFrames are loaded
    assert isinstance(csv_metadata._series_desc_df, pd.DataFrame)
    assert isinstance(csv_metadata._label_coords_df, pd.DataFrame)
    assert isinstance(csv_metadata._train_df, pd.DataFrame)
    assert isinstance(csv_metadata._merged_df, pd.DataFrame)


def test_init_exception(setup_csv_files, logger, caplog):
    """
    Test that exceptions during initialization are handled and logged.
    """
    files = setup_csv_files

    # Mock pd.read_csv to raise an exception
    with patch('pandas.read_csv', side_effect=Exception("File not found")):
        with pytest.raises(Exception):
            csv_metadata = CSVMetadata(
                "",
                files["series_description"],
                files["label_coordinates"],
                files["train"],
                logger
            )

    # Check that the error was logged
    assert "Error initializing CSVMetadata" in caplog.text


def test_merged_property(setup_csv_files, logger):
    """
    Test the merged property.
    """
    files = setup_csv_files
    csv_metadata = CSVMetadata(
                                "",
                                files["series_description"],
                                files["label_coordinates"],
                                files["train"],
                                logger
                                )

    # Check that the merged property returns a DataFrame
    assert isinstance(csv_metadata.merged, pd.DataFrame)


def test_train_df_property(setup_csv_files, logger):
    """
    Test the train_df property.
    """
    files = setup_csv_files
    csv_metadata = CSVMetadata(
                                "",     
                                files["series_description"],
                                files["label_coordinates"],
                                files["train"],
                                logger
                                )

    # Check that the train_df property returns a DataFrame
    assert isinstance(csv_metadata.train_df, pd.DataFrame)


def test_merge_metadata(setup_csv_files, logger):
    """
    Test the _merge_metadata method.
    """
    files = setup_csv_files
    csv_metadata = CSVMetadata(
                                "",
                                files["series_description"],
                                files["label_coordinates"],
                                files["train"],
                                logger
                                )

    # Call the private method _merge_metadata
    merged_df = csv_metadata._merge_metadata()

    # Check that the result is a DataFrame
    assert isinstance(merged_df, pd.DataFrame)


def test_melt_and_clean_train_df(setup_csv_files, logger):
    """
    Test the _melt_and_clean_train_df method.
    """
    files = setup_csv_files
    csv_metadata = CSVMetadata(
                                "",
                                files["series_description"],
                                files["label_coordinates"],
                                files["train"],
                                logger
                                )

    # Call the private method _melt_and_clean_train_df
    melted_df = csv_metadata._melt_and_clean_train_df()

    # Check that the result is a DataFrame
    assert isinstance(melted_df, pd.DataFrame)

    # Check that the DataFrame has the expected columns
    assert 'study_id' in melted_df.columns
    assert 'condition' in melted_df.columns
    assert 'level' in melted_df.columns
    assert 'severity' in melted_df.columns


def test_melt_and_clean_train_df_exception(setup_csv_files, logger, caplog):
    """
    Test that exceptions in _melt_and_clean_train_df are handled and logged.
    """
    files = setup_csv_files
    csv_metadata = CSVMetadata(
                                "",
                                files["series_description"],
                                files["label_coordinates"],
                                files["train"],
                                logger
                               )

    # Mock the melt method of the existing _train_df DataFrame to raise an exception
    with patch.object(csv_metadata._train_df, 'melt', side_effect=Exception("Melt error")):
        with pytest.raises(Exception):
            csv_metadata._melt_and_clean_train_df()

    # Check that the error was logged with the correct message
    assert "Error melting and cleaning training DataFrame" in caplog.text


def test_merge_with_label_coordinates(setup_csv_files, logger):
    """
    Test the _merge_with_label_coordinates method.
    """
    files = setup_csv_files
    csv_metadata = CSVMetadata(
                                "",
                                files["series_description"],
                                files["label_coordinates"],
                                files["train"],
                                logger
                               )

    # Call the private method _melt_and_clean_train_df
    melted_df = csv_metadata._melt_and_clean_train_df()

    # Call the private method _merge_with_label_coordinates
    merged_df = csv_metadata._merge_with_label_coordinates(melted_df)

    # Check that the result is a DataFrame
    assert isinstance(merged_df, pd.DataFrame)

    # Check that the DataFrame has the expected columns
    assert 'x' in merged_df.columns
    assert 'y' in merged_df.columns


def test_merge_with_series_descriptions(setup_csv_files, logger):
    """
    Test the _merge_with_series_descriptions method.
    """
    files = setup_csv_files
    csv_metadata = CSVMetadata(
                                "",
                                files["series_description"],
                                files["label_coordinates"],
                                files["train"],
                                logger
                               )

    # Call the private method _melt_and_clean_train_df
    melted_df = csv_metadata._melt_and_clean_train_df()

    # Call the private method _merge_with_label_coordinates
    merged_df = csv_metadata._merge_with_label_coordinates(melted_df)

    # Call the private method _merge_with_series_descriptions
    final_merged_df = csv_metadata._merge_with_series_descriptions(merged_df)

    # Check that the result is a DataFrame
    assert isinstance(final_merged_df, pd.DataFrame)

    # Check that the DataFrame has the expected columns
    assert 'series_description' in final_merged_df.columns


def test_merge_metadata_exception(setup_csv_files, logger, caplog):
    """
    Test that exceptions in _merge_metadata are handled and logged.
    """
    files = setup_csv_files
    csv_metadata = CSVMetadata(
                                "",
                                files["series_description"],
                                files["label_coordinates"],
                                files["train"],
                                logger
                               )

    # Mock the _melt_and_clean_train_df to raise an exception
    with patch.object(
                        csv_metadata,
                        '_melt_and_clean_train_df',
                        side_effect=Exception("Merge error")
                       ):
        with pytest.raises(Exception):
            csv_metadata._merge_metadata()

    # Check that the error was logged
    assert "Error merging metadata" in caplog.text


def test_normalize_identifier_types(setup_csv_files, logger):
    """
    Test the _normalize_identifier_types method.
    """
    files = setup_csv_files
    csv_metadata = CSVMetadata(
                                "",
                                files["series_description"],
                                files["label_coordinates"],
                                files["train"],
                                logger
                               )

    # Call the private method _melt_and_clean_train_df
    melted_df = csv_metadata._melt_and_clean_train_df()

    # Call the private method _merge_with_label_coordinates
    merged_df = csv_metadata._merge_with_label_coordinates(melted_df)

    # Call the private method _merge_with_series_descriptions
    final_merged_df = csv_metadata._merge_with_series_descriptions(merged_df)

    # Call the private method _normalize_identifier_types
    normalized_df = csv_metadata._normalize_identifier_types(final_merged_df)

    # Check that the result is a DataFrame
    assert isinstance(normalized_df, pd.DataFrame)

    # Check that the identifier columns are of the correct type
    assert normalized_df['study_id'].dtype == np.int64
    assert normalized_df['series_id'].dtype == np.int64
    assert normalized_df['instance_number'].dtype == int


# def test_to_tf_lookup(setup_csv_files, logger):
#     """
#     test the to_tf_lookup method.
#     """
#     files = setup_csv_files
#     csv_metadata = testcsvmetadata(
#                                 files["series_description"],
#                                 files["label_coordinates"],
#                                 files["train"],
#                                 logger
#                                )

#     # call the to_tf_lookup method
#     lookup_table = csv_metadata.to_tf_lookup()

#     # check that the result is a tf.lookup.statichashtable
#     assert isinstance(lookup_table, tf.lookup.statichashtable)


# def test_to_tf_lookup_exception(setup_csv_files, logger, caplog):
#     """
#     Test that exceptions in to_tf_lookup are handled and logged.
#     """
#     files = setup_csv_files
#     csv_metadata = CSVMetadata(
#                                 files["series_description"],
#                                 files["label_coordinates"],
#                                 files["train"],
#                                 logger
#                                )

#     # Mock the merged property to raise an exception
#     with patch.object(csv_metadata, 'merged', side_effect=Exception("Lookup error")):
#         with pytest.raises(Exception):
#             csv_metadata.to_tf_lookup()

#     # Check that the error was logged
#     assert "Error creating lookup table" in caplog.text