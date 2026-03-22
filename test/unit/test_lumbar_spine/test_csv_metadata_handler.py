import pytest
import pandas as pd
import numpy as np
from src.projects.lumbar_spine.csv_metadata_handler import CSVMetadataHandler
from unittest.mock import patch, MagicMock
from contextlib import ExitStack


# class TestCSVMetadataHandler(CSVMetadataHandler):
#     def __init__(self, series_description, label_coordinates, train, mock_mock_logger):
#         super().__init__(series_description, label_coordinates, train, mock_logger)
#         # Force columns to be strings
#         self._series_desc_df = self._series_desc_df.astype(str)
#         self._label_coords_df = self._label_coords_df.astype(str)
#         self._train_df = self._train_df.astype(str)
#         self._merged_df = self._merged_df.astype(str)

def test_init_isolated(mock_config, mock_logger, tmp_path):
    """
    Initialization logic is tested by patching internal method calls.
    """
    # Internal methods of the CSVMetadataHandler class are patched
    with ExitStack() as stack:
        mock_setup = stack.enter_context(
            patch.object(CSVMetadataHandler, "_setup_paths")
        )
        mock_load = stack.enter_context(
            patch.object(CSVMetadataHandler, "_load_and_cleanse_data")
        )

        handler = CSVMetadataHandler(
            dicom_studies_dir=str(tmp_path / "dicom"),
            series_description="desc.csv",
            label_coordinates="coords.csv",
            label_enriched="enriched.csv",
            train="train.csv",
            config=mock_config,
            logger=mock_logger,
        )

        # Basic assignments are verified
        assert handler._config == mock_config
        assert str(handler._dicom_studies_dir) == str(tmp_path / "dicom")

        # Internal methods are expected to be called exactly once
        mock_setup.assert_called_once_with(
            "desc.csv", "coords.csv", "enriched.csv", "train.csv"
        )
        mock_load.assert_called_once()


def test_init_raises_exception(mock_config, mock_logger):
    """
    Initialization handling of data cleansing failures is tested.
    """
    # Internal methods are patched to simulate a failure
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(CSVMetadataHandler, "_setup_paths")
        )

        stack.enter_context(
            patch.object(
                CSVMetadataHandler,
                "_load_and_cleanse_data",
                side_effect=Exception("Cleansing Failed")
            )
        )
        with pytest.raises(Exception) as excinfo:
            CSVMetadataHandler(
                dicom_studies_dir="",
                series_description="",
                label_coordinates="",
                label_enriched="",
                train="",
                config=mock_config,
                logger=mock_logger,
            )
        assert "Cleansing Failed" in str(excinfo.value)


def test_setup_paths(mock_config, mock_logger, tmp_path):
    """
    Path construction logic is tested for both relative and absolute paths.
    """
    # Initialize handler without triggering the full loading process
    with ExitStack() as stack:
        stack.enter_context(patch.object(CSVMetadataHandler, "_setup_paths"))
        stack.enter_context(patch.object(CSVMetadataHandler, "_load_and_cleanse_data"))

        handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description="desc.csv",
            label_coordinates=str(tmp_path / "path/coords.csv"),
            label_enriched="enriched.csv",
            train="train.csv",
            config=mock_config,
            logger=mock_logger,
        )

    # Manually trigger the method to test its internal logic
    handler._setup_paths(
        series_description="desc.csv",
        label_coordinates=str(tmp_path / "path/coords.csv"),
        label_enriched="enriched.csv",
        train="train.csv"
    )

    # Assertions for relative paths (should be joined with root_dir)
    assert handler._paths_dict['series_description'] == tmp_path / "desc.csv"
    assert handler._paths_dict['train'] == tmp_path / "train.csv"

    # Assertion for absolute path (should remain unchanged)
    # Pathlib's / operator ignores the left side if the right side is absolute
    assert handler._paths_dict['label_raw'] == tmp_path / "path/coords.csv"


def test_load_and_cleanse_data_integration(mock_config, setup_csv_files, mock_logger):
    """
    Integration test for data loading using real temporary CSV files.
    """
    def side_effect_scale(df):
        # Simulate the real method by adding the required column
        # using the string format that the subsequent ast.literal_eval will parse
        df["actual_file_format"] = "(640, 640)"
        return df

    with ExitStack() as stack:
        # 1. Setup the mock with a side_effect to return the input DataFrame
        # This prevents handler._label_coords_df from becoming a MagicMock object
        mock_scale = stack.enter_context(
            patch.object(CSVMetadataHandler, "_scale_series_format_locations")
        )
        mock_scale.side_effect = side_effect_scale

        # Force the handler to believe the enriched file does not exist
        # (which is false actually: the file was created by the fixture setup_csv_files).
        # This triggers the scaling logic and the subsequent save.
        stack.enter_context(
            patch("pathlib.Path.is_file", return_value=False)
        )

        # Prevent actual writing to disk during the test
        stack.enter_context(patch("pandas.DataFrame.to_csv"))
        stack.enter_context(patch("pathlib.Path.mkdir"))

        handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=setup_csv_files["series_description"],
            label_coordinates=setup_csv_files["label_coordinates"],
            label_enriched=setup_csv_files["label_enriched"],
            train=setup_csv_files["train"],
            config=mock_config,
            logger=mock_logger
        )

    # 2. Assertions - Data integrity and type conversion
    # Verify study_id is now an integer (converted from CSV string)
    assert handler._series_desc_df["study_id"].iloc[0] == 1
    assert pd.api.types.is_integer_dtype(handler._series_desc_df["study_id"])

    # Verify coordinates are floats
    assert isinstance(handler._label_coords_df["x"].iloc[0], float)

    # Verify tuple reconstruction for 'actual_file_format'
    # The string "(640, 640)" in CSV should become the tuple (640, 640)
    expected_tuple = (640, 640)
    assert handler._label_coords_df["actual_file_format"].iloc[0] == expected_tuple
    assert isinstance(handler._label_coords_df["actual_file_format"].iloc[0], tuple)

    # Verify that the scaling logic was triggered
    # because the enriched file was missing or processed for the first time
    mock_scale.assert_called_once()

    # Check that the scale function received the DataFrame
    # (The first argument of the call should be a pandas DataFrame)
    args, _ = mock_scale.call_args
    assert isinstance(args[0], pd.DataFrame)


def test_load_from_enriched_cache(mock_config, setup_csv_files, mock_logger):
    """
    Test that the handler loads data directly from the enriched file
    and skips scaling logic when the file already exists.
    """
    with ExitStack() as stack:
        # 1. Patch the scaling method to ensure it is NOT called
        mock_scale = stack.enter_context(
            patch.object(CSVMetadataHandler, "_scale_series_format_locations")
        )

        # 2. Force is_file to True to simulate an existing cache
        stack.enter_context(
            patch("pathlib.Path.is_file", return_value=True)
        )

        # 3. Prevent any accidental writes during the test
        stack.enter_context(patch("pandas.DataFrame.to_csv"))

        # 4. Instantiate the handler
        # Note: label_enriched in handler will point to setup_csv_files["label_enriched"]
        handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=setup_csv_files["series_description"],
            label_coordinates=setup_csv_files["label_coordinates"],
            label_enriched=setup_csv_files["label_enriched"],
            train=setup_csv_files["train"],
            config=mock_config,
            logger=mock_logger
        )

    # Assertions
    # Verify that data was loaded (from the enriched file created by the fixture)
    assert handler._label_coords_df is not None
    assert "actual_file_format" in handler._label_coords_df.columns

    # CRITICAL: The scaling logic must NOT have been triggered
    mock_scale.assert_not_called()

    # Verify that conversion to tuple still happens for the cached data
    # (The fixture setup_csv_files already contains "(640, 640)" in the enriched file)
    assert isinstance(handler._label_coords_df["actual_file_format"].iloc[0], tuple)


def test_scale_triggers_inspection_when_column_missing(mock_config, mock_logger):
    """
    Scenario: Input DataFrame is missing the 'actual_file_format' column.
    Expectation: The handler must trigger the parallel inspection logic
                 by calling _get_file_formats for each record.
    """
    with ExitStack() as stack:
        # Bypass initialization side effects
        stack.enter_context(patch.object(CSVMetadataHandler, "_setup_paths"))
        stack.enter_context(patch.object(CSVMetadataHandler, "_load_and_cleanse_data"))

        handler = CSVMetadataHandler("", "", "", "", "", config=mock_config, logger=mock_logger)

    # Input DataFrame without the format column
    input_df = pd.DataFrame({
        'series_id': [100],
        'x': [10.0],
        'y': [10.0]
    })

    # Patch the inspection method to monitor its execution
    with patch.object(handler, '_get_file_formats', return_value=(512, 512)) as mock_inspect:
        handler._scale_series_format_locations(input_df)

    # Assertions: Verify that the inspection was indeed triggered
    mock_inspect.assert_called()
    assert mock_inspect.call_count == len(input_df), "Should call inspection once per row"


def test_scale_skips_inspection_when_column_exists(mock_config, mock_logger):
    """
    Scenario: Input DataFrame already contains the 'actual_file_format' column.
    Expectation: The handler should use existing data and NEVER call
                 the expensive _get_file_formats method.
    """
    with ExitStack() as stack:
        stack.enter_context(patch.object(CSVMetadataHandler, "_setup_paths"))
        stack.enter_context(patch.object(CSVMetadataHandler, "_load_and_cleanse_data"))

        handler = CSVMetadataHandler("", "", "", "", "", config=mock_config, logger=mock_logger)

    # Input DataFrame with pre-existing format data (e.g., loaded from enriched CSV)
    input_df = pd.DataFrame({
        'series_id': [100, 100],
        'x': [10.0, 10.0],
        'y': [10.0, 10.0],
        'actual_file_format': [(640, 640), (640, 640)]
    })

    # Patch the inspection method to ensure it is NOT invoked
    with patch.object(handler, '_get_file_formats') as mock_inspect:
        output_df = handler._scale_series_format_locations(input_df)

    # Assertions: Verify that the shortcut was taken
    mock_inspect.assert_not_called()

    # Verify that the output structure remains consistent
    assert 'actual_file_format' in output_df.columns
    assert output_df.iloc[0]['actual_file_format'] == (640, 640)


def test_train_df_property(setup_csv_files, mock_logger, mock_config):
    """
    Test the train_df property.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        dicom_studies_dir="",
        series_description=files["series_description"],
        label_coordinates=files["label_coordinates"],
        label_enriched=files["label_enriched"],
        train=files["train"],
        config=mock_config,
        logger=mock_logger
    )

    # Check that the train_df property returns a DataFrame
    assert isinstance(csv_metadata_handler.train_df, pd.DataFrame)


def test_merge_metadata_integration(setup_csv_files, mock_logger, mock_config, caplog):
    """
    Test the complete metadata merge pipeline.
    Ensures that the final DataFrame contains all necessary features
    and represents the correct intersection of input files.
    """
    files = setup_csv_files

    # 1. Initialize the handler
    # Note: initialization triggers _load_and_cleanse_data automatically
    handler = CSVMetadataHandler(
        dicom_studies_dir="",
        series_description=files["series_description"],
        label_coordinates=files["label_coordinates"],
        label_enriched=files["label_enriched"],
        train=files["train"],
        config=mock_config,
        logger=mock_logger
    )

    # 2. Execute the merge process
    merged_df = handler._merge_metadata()

    # --- ASSERTIONS ---

    # A. Structural Integrity
    assert isinstance(merged_df, pd.DataFrame)
    assert not merged_df.empty, "The merged DataFrame should not be empty."

    # B. Column Presence Verification
    # These columns are the result of merging train, coordinates, and series descriptions
    required_columns = [
        "study_id", "condition_level", "severity",
        "series_id", "instance_number", "x", "y"
    ]
    for col in required_columns:
        assert col in merged_df.columns, f"Merged DataFrame is missing column: {col}"

    # C. Data Consistency Logic
    # Verify that study_id is normalized as integer
    assert pd.api.types.is_integer_dtype(merged_df["study_id"])

    # Verify that severity text is cleaned (lowercase and stripped)
    sample_severity = merged_df["severity"].iloc[0]
    assert sample_severity == sample_severity.lower().strip()

    # D. Verification of the Merge Intersection (The "Shape" Check)
    # We manually simulate the expected intersection to compare counts.
    # We expect only rows that exist in BOTH melted train data AND coordinates.
    melted_train = handler._train_df.melt(
        id_vars="study_id",
        var_name="condition_level",
        value_name="severity"
    ).dropna()

    # Force study_id to int32 to match handler._label_coords_df
    melted_train["study_id"] = melted_train["study_id"].astype(np.int32)

    # Note: condition_level in label_coords is already formatted by the handler
    expected_count = len(pd.merge(
        melted_train,
        handler._label_coords_df,
        on=["study_id", "condition_level"],
        how="inner"
    ))

    assert merged_df.shape[0] == expected_count, (
        f"Row count mismatch. Expected {expected_count} rows (inner join), "
        f"but got {merged_df.shape[0]}."
    )

    # E. Successful Termination Check
    # Confirming the logger reached the final success message
    expected_message = "Metadata merge completed successfully"
    failure_message = f"The success message '{expected_message}' was not found in the logs."
    assert any(expected_message in record.message for record in caplog.records), failure_message


def test_melt_and_clean_train_df(setup_csv_files, mock_logger, mock_config):
    """
    Test the _melt_and_clean_train_df method.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        dicom_studies_dir="",
        series_description=files["series_description"],
        label_coordinates=files["label_coordinates"],
        label_enriched=files["label_enriched"],
        train=files["train"],
        config=mock_config,
        logger=mock_logger
    )

    # Call the private method _melt_and_clean_train_df
    melted_df = csv_metadata_handler._melt_and_clean_train_df()

    # Check that the result is a DataFrame
    assert isinstance(melted_df, pd.DataFrame)

    # Check that the DataFrame has the expected columns
    assert 'study_id' in melted_df.columns
    assert 'condition_level' in melted_df.columns
    assert 'severity' in melted_df.columns


def test_melt_and_clean_train_df_exception(setup_csv_files, mock_logger, mock_config, caplog):
    """
    Test that exceptions in _melt_and_clean_train_df are handled and logged.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        dicom_studies_dir="",
        series_description=files["series_description"],
        label_coordinates=files["label_coordinates"],
        label_enriched=files["label_enriched"],
        train=files["train"],
        config=mock_config,
        logger=mock_logger
    )

    # Mock the melt method of the existing _train_df DataFrame to raise an exception
    with patch.object(csv_metadata_handler._train_df, 'melt', side_effect=Exception("Melt error")):
        with pytest.raises(Exception):
            csv_metadata_handler._melt_and_clean_train_df()

    # Check that the error was logged with the correct message
    assert "Error melting and cleaning training DataFrame" in caplog.text


def test_merge_with_label_coordinates(setup_csv_files, mock_logger, mock_config):
    """
    Test the _merge_with_label_coordinates method.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        dicom_studies_dir="",
        series_description=files["series_description"],
        label_coordinates=files["label_coordinates"],
        label_enriched=files["label_enriched"],
        train=files["train"],
        config=mock_config,
        logger=mock_logger
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


def test_merge_with_series_descriptions(setup_csv_files, mock_logger, mock_config):
    """
    Test the _merge_with_series_descriptions method.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        dicom_studies_dir="",
        series_description=files["series_description"],
        label_coordinates=files["label_coordinates"],
        label_enriched=files["label_enriched"],
        train=files["train"],
        config=mock_config,
        logger=mock_logger
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
    assert 'series_description' in final_merged_df.columns


def test_merge_metadata_exception(setup_csv_files, mock_logger, mock_config, caplog):
    """
    Test that exceptions in _merge_metadata are handled and logged.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        dicom_studies_dir="",
        series_description=files["series_description"],
        label_coordinates=files["label_coordinates"],
        label_enriched=files["label_enriched"],
        train=files["train"],
        config=mock_config,
        logger=mock_logger
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


def test_normalize_identifier_types(setup_csv_files, mock_logger, mock_config):
    """
    Test the _normalize_identifier_types method.
    """
    files = setup_csv_files
    csv_metadata_handler = CSVMetadataHandler(
        dicom_studies_dir="",
        series_description=files["series_description"],
        label_coordinates=files["label_coordinates"],
        label_enriched=files["label_enriched"],
        train=files["train"],
        config=mock_config,
        logger=mock_logger
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


class TestEncodeDataFrame:
    """
    Test suite for _encode_dataframe method.
    Focuses on the orchestration of mappings and encodings applications.
    """

    def test_encode_dataframe_success(
            self,
            setup_csv_files,
            mock_metadata,
            mock_config,
            mock_logger,
            caplog,
            tmp_path
    ):
        """
        Test successful encoding of all target columns.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        # Define expected columns to be processed
        target_columns = ["condition_level", "series_description", "severity"]

        # Create dummy mappings for the mock
        mock_mappings = {
            col: {
                    val: idx for idx, val in enumerate(mock_metadata[col].unique())
            } for col in target_columns
        }

        # Create a version of DataFrame where values are replaced by dummy ints
        encoded_df = mock_metadata.copy()
        for col in target_columns:
            encoded_df[col] = 0  # Simplified mock return

        with ExitStack() as stack:
            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = csv_metadata_handler._logger

            # Mock internal calls to isolate _encode_dataframe logic
            mock_create = stack.enter_context(
                patch.object(
                    csv_metadata_handler,
                    '_create_mappings',
                    return_value=mock_mappings
                )
            )

            mock_apply = stack.enter_context(
                patch.object(
                    csv_metadata_handler,
                    '_apply_encodings',
                    return_value=encoded_df
                )
            )

            result = csv_metadata_handler._encode_dataframe(mock_metadata)

            # Verifications
            assert isinstance(result, pd.DataFrame)
            mock_create.assert_called_once_with(mock_metadata, target_columns)
            mock_apply.assert_called_once_with(mock_metadata, target_columns, mock_mappings)

            # Verify that the result is indeed the one returned by _apply_encodings
            assert (result[target_columns] == 0).all().all()

    def test_encode_dataframe_missing_column(
        self,
        setup_csv_files,
        mock_config,
        mock_logger,
        caplog,
        tmp_path
    ):
        """
        Test behavior when a required column is missing from the input DataFrame.
        Note: Depending on _create_mappings implementation, this should raise an error.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        # DF missing 'severity' column
        incomplete_df = pd.DataFrame({
            "condition_level": ["A"],
            "series_description": ["Desc"]
        })

        with ExitStack() as stack:
            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = csv_metadata_handler._logger

            stack.enter_context(
                patch.object(
                    csv_metadata_handler,
                    '_create_mappings',
                    side_effect=KeyError("severity")
                )
            )

            with pytest.raises(KeyError):
                csv_metadata_handler._encode_dataframe(incomplete_df)

            # Correct way to check for ERROR level in caplog records
            assert any(record.levelname == "ERROR" for record in caplog.records)

            # Check for the specific message in the text
            assert "severity" in caplog.text

    def test_encode_dataframe_empty_input(
        self,
        setup_csv_files,
        mock_config,
        mock_logger,
        caplog,
        tmp_path
    ):
        """
        Test behavior with an empty DataFrame (but with correct columns).
        """
        empty_df = pd.DataFrame(columns=["condition_level", "series_description", "severity"])

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        with ExitStack() as stack:
            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = csv_metadata_handler._logger

            stack.enter_context(
                patch.object(
                    csv_metadata_handler,
                    '_create_mappings',
                    return_value={}
                )
            )

            stack.enter_context(
                patch.object(
                    csv_metadata_handler,
                    '_apply_encodings',
                    return_value=empty_df
                )
            )

            with pytest.raises(ValueError) as exc_info:
                csv_metadata_handler._encode_dataframe(empty_df)

            # Check the error message
            assert "Empty DataFrame" in str(exc_info.value)

            # Check that the error was actually logged
            assert any(record.levelname == "ERROR" and "Empty DataFrame" in record.message
                       for record in caplog.records)

    def test_encode_dataframe_exception_handling(
        self,
        setup_csv_files,
        mock_config,
        mock_logger,
        mock_metadata,
        caplog,
        tmp_path
    ):
        """
        Test that any exception during the process is caught, logged, and re-raised.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        # Mock sub-calls to avoid heavy processing
        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = csv_metadata_handler._logger

            stack.enter_context(
                patch.object(
                    csv_metadata_handler,
                    '_create_mappings',
                    side_effect=RuntimeError("Unexpected error")
                )
            )

            with pytest.raises(RuntimeError) as exc_info:
                csv_metadata_handler._encode_dataframe(mock_metadata)

            # Check the exception message itself
            assert "Unexpected error" in str(exc_info.value)

            # Verify the formatted error message in logs
            expected_log_msg = (
                "Fatal error in function CSVMetadataHandler._encode_dataframe: "
                "Unexpected error"
            )
            assert expected_log_msg in caplog.text

            # Verify that exc_info=True was used (the stacktrace is not None) :
            assert any(record.exc_info for record in caplog.records if record.levelname == "ERROR")


class TestCreateMappings:
    """
    Test suite for the _create_mappings method.
    Covers nominal cases, missing columns, sorting, and error propagation.
    """

    def test_create_mappings_success(
        self,
        setup_csv_files,
        mock_config,
        mock_logger,
        mock_metadata,
        caplog,
        tmp_path
    ):
        """
        Nominal case: verify that mappings are created for all requested columns.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        # Define columns to encode
        columns_to_encode = ["condition_level", "severity"]

        # Mock the mapper object that _create_string_to_int_mapper is expected to return
        mock_mapper = MagicMock()
        mock_mapper.mapping = {"dummy_key": 0}

        # Enter each context manager through the stack
        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = csv_metadata_handler._logger

            mock_method = stack.enter_context(
                patch.object(
                    csv_metadata_handler,
                    '_create_string_to_int_mapper',
                    return_value=mock_mapper
                )
            )

            result = csv_metadata_handler._create_mappings(mock_metadata, columns_to_encode)

            # Check result structure
            assert isinstance(result, dict)
            assert "condition_level" in result
            assert "severity" in result
            assert result["condition_level"] == {"dummy_key": 0}
            assert result["severity"] == {"dummy_key": 0}

            # Verify that mapper was called for each column
            assert mock_method.call_count == len(columns_to_encode)

    def test_create_mappings_sorting_and_uniqueness(
        self,
        setup_csv_files,
        mock_config,
        mock_logger,
        caplog,
        tmp_path
    ):
        """
        Verify that values are sorted and duplicates/NaNs are handled before mapping.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        # Create specific data with unsorted values and NaNs
        df = pd.DataFrame({
            "target_col": ["Beta", "Alpha", "Beta", None, "Gamma"]
        })
        columns = ["target_col"]

        mock_mapper = MagicMock()
        mock_mapper.mapping = {}

        # Enter each context manager through the stack
        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = csv_metadata_handler._logger

            mock_method = stack.enter_context(
                patch.object(
                    csv_metadata_handler,
                    '_create_string_to_int_mapper',
                    return_value=mock_mapper
                )
            )
            csv_metadata_handler._create_mappings(df, columns)

            # Capture the argument passed to the mapper
            # The list should be sorted: ['Alpha', 'Beta', 'Gamma'] (NaN removed)
            args, _ = mock_method.call_args
            passed_list = args[0]

            assert passed_list == ["Alpha", "Beta", "Gamma"]
            assert passed_list == sorted(passed_list)

    def test_create_mappings_missing_column_warning(
        self,
        setup_csv_files,
        mock_config,
        mock_logger,
        mock_metadata,
        caplog,
        tmp_path
    ):
        """
        Test that missing columns in the DataFrame trigger a warning but don't stop the process.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        # 'unknown_column' does not exist in mock_metadata
        columns_to_encode = ["condition_level", "unknown_column"]

        mock_mapper = MagicMock()
        mock_mapper.mapping = {"a": 1}

        # Enter each context manager through the stack
        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = csv_metadata_handler._logger

            stack.enter_context(
                patch.object(
                    csv_metadata_handler,
                    '_create_string_to_int_mapper',
                    return_value=mock_mapper
                )
            )
            result = csv_metadata_handler._create_mappings(mock_metadata, columns_to_encode)

            # Should only contain 'condition'
            assert "condition_level" in result
            assert "unknown_column" not in result

            # Verify warning was logged
            assert any(record.levelname == "WARNING" for record in caplog.records)
            assert "Column 'unknown_column' not found" in caplog.text

    def test_create_mappings_exception_handling(
        self,
        setup_csv_files,
        mock_config,
        mock_logger,
        mock_metadata,
        caplog,
        tmp_path
    ):
        """
        Verify that exceptions are caught, logged with exc_info, and re-raised.
        """
        columns = ["condition_level"]

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        # Enter each context manager through the stack
        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = csv_metadata_handler._logger

            # Force an error during the mapping process
            stack.enter_context(
                patch.object(
                    csv_metadata_handler,
                    '_create_string_to_int_mapper',
                    side_effect=Exception("Mapper failure")
                )
            )

            with pytest.raises(Exception) as exc_info:
                csv_metadata_handler._create_mappings(mock_metadata, columns)

            assert "Mapper failure" in str(exc_info.value)

            # Verify error logging logic
            assert any(record.exc_info for record in caplog.records if record.levelname == "ERROR")

            # Verify the specific error message format from the source code
            assert "while processing column 'condition_level'" in caplog.text


class TestApplyEncodings:
    """
    Test suite for _apply_encodings method.
    Verifies data transformation, handling of missing values, and type casting.
    """

    def test_apply_encodings_success(
        self,
        setup_csv_files,
        mock_config, mock_logger,
        mock_metadata
    ):
        """
        Nominal case: verify that values are correctly mapped and types are cast to int.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        # Define columns and mappings based on mock_metadata values
        columns_to_encode = ["condition_level", "severity"]
        mappings = {
            "condition_level": {
                val: idx for idx, val in enumerate(mock_metadata["condition_level"].unique())
            },
            "severity": {val: idx for idx, val in enumerate(mock_metadata["severity"].unique())}
        }

        result = csv_metadata_handler._apply_encodings(
            mock_metadata,
            columns_to_encode,
            mappings
        )

        # Verifications
        assert result["condition_level"].dtype in [np.int32, np.int64]
        assert result["severity"].dtype in [np.int32, np.int64]

        # Check that no NaN remain after mapping
        assert not result[columns_to_encode].isnull().any().any()

    def test_apply_encodings_preserves_original_df(
        self,
        setup_csv_files,
        mock_config,
        mock_logger,
        mock_metadata
    ):
        """
        Verify that the input DataFrame is not modified (deep copy) during encoding.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        # Take a snapshot of the original data
        df_input = mock_metadata.copy()
        df_input_original = df_input.copy()

        columns = ["condition_level", "severity"]

        # Simple mapping to force a change in the result
        mappings = {
            "condition_level": {val: 1 for val in df_input["condition_level"].unique()},
            "severity": {val: 1 for val in df_input["severity"].unique()}
        }

        # Act
        result_df = csv_metadata_handler._apply_encodings(df_input, columns, mappings)

        # Verify result is transformed
        assert (result_df[columns] == 1).all().all()

        # Verify original input is UNTOUCHED (integrity check)
        pd.testing.assert_frame_equal(df_input, df_input_original)
        assert df_input.loc[0, "condition_level"] == df_input_original.loc[0, "condition_level"]
        assert isinstance(df_input.loc[0, "condition_level"], str)

    def test_apply_encodings_with_unknown_values(self, setup_csv_files, mock_config, mock_logger):
        """
        Verify that values not present in the mapping are filled with -1.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        df = pd.DataFrame({"condition_level": ["Known", "Unknown"]})
        columns = ["condition_level"]
        mappings = {"condition_level": {"Known": 5}}

        result = csv_metadata_handler._apply_encodings(df, columns, mappings)

        assert result["condition_level"].tolist() == [5, -1]
        assert np.issubdtype(result["condition_level"].dtype, np.integer)

    def test_apply_encodings_with_nans(self, setup_csv_files, mock_config, mock_logger):
        """
        Verify that existing NaN values in the source DataFrame are handled as -1.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        df = pd.DataFrame({"severity": ["Normal", np.nan]})
        columns = ["severity"]
        mappings = {"severity": {"Normal": 0}}

        result = csv_metadata_handler._apply_encodings(df, columns, mappings)

        assert result["severity"].tolist() == [0, -1]

    def test_apply_encodings_partial_columns(self, setup_csv_files, mock_config, mock_logger):
        """
        Verify that only the specified columns are encoded, leaving others untouched.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        df = pd.DataFrame({
            "condition_level": ["A"],
            "other_info": ["StaySame"]
        })
        columns = ["condition_level"]
        mappings = {"condition_level": {"A": 1}}

        result = csv_metadata_handler._apply_encodings(df, columns, mappings)

        assert result.loc[0, "condition_level"] == 1
        assert result.loc[0, "other_info"] == "StaySame"


class TestCreateStringToIntMapper:
    """
    Test suite for the mapper factory function.
    Verifies the generated callable and its attached metadata.
    """

    def test_mapper_creation_success(self, setup_csv_files, mock_config, mock_logger):
        """
        Nominal case: Verify that the returned mapper correctly handles known and unknown keys.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        pathologies = ["Normal", "Moderate", "Severe"]

        # Passing None will cause enumerate() to raise a TypeError
        with patch('src.core.utils.logger.get_current_logger') as mock_get_log:
            # Reuse the mock_logger from the manager
            mock_get_log.return_value = csv_metadata_handler._logger

            # Generate the mapper function
            mapper = csv_metadata_handler._create_string_to_int_mapper(pathologies)

        # 1. Test the callable behavior
        assert callable(mapper)
        assert mapper("Normal") == 0
        assert mapper("Severe") == 2
        assert mapper("Unknown") == -1  # Test default value for missing keys

        # 2. Test the attached attributes
        assert hasattr(mapper, 'mapping')
        assert hasattr(mapper, 'reverse_mapping')
        assert mapper.mapping["Moderate"] == 1
        assert mapper.reverse_mapping[1] == "Moderate"

    def test_mapper_with_empty_list(self, setup_csv_files, mock_config, mock_logger):
        """
        Boundary case: Verify behavior when an empty list is provided.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        # Passing None will cause enumerate() to raise a TypeError
        with patch('src.core.utils.logger.get_current_logger') as mock_get_log:
            # Reuse the mock_logger from the manager
            mock_get_log.return_value = csv_metadata_handler._logger

            mapper = csv_metadata_handler._create_string_to_int_mapper([])

        assert mapper("Any") == -1
        assert mapper.mapping == {}
        assert mapper.reverse_mapping == {}

    def test_mapper_idempotency_and_order(self, setup_csv_files, mock_config, mock_logger):
        """
        Ensure that the mapping follows the exact order of the input list.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        input_list = ["C", "A", "B"]

        # Passing None will cause enumerate() to raise a TypeError
        with patch('src.core.utils.logger.get_current_logger') as mock_get_log:
            # Reuse the mock_logger from the manager
            mock_get_log.return_value = csv_metadata_handler._logger

            mapper = csv_metadata_handler._create_string_to_int_mapper(input_list)

        # Verify that indices match the list position
        assert mapper("C") == 0
        assert mapper("A") == 1
        assert mapper("B") == 2

    def test_mapper_exception_handling(self, setup_csv_files, mock_config, mock_logger, caplog):
        """
        Verify that exceptions (e.g., passing None instead of a list) are logged and raised.
        """

        files = setup_csv_files
        csv_metadata_handler = CSVMetadataHandler(
            dicom_studies_dir="",
            series_description=files["series_description"],
            label_coordinates=files["label_coordinates"],
            label_enriched=files["label_enriched"],
            train=files["train"],
            config=mock_config,
            logger=mock_logger
        )

        # Passing None will cause enumerate() to raise a TypeError
        with patch('src.core.utils.logger.get_current_logger') as mock_get_log:
            # Reuse the mock_logger from the manager
            mock_get_log.return_value = csv_metadata_handler._logger

            # Passing None will cause enumerate() to raise a TypeError
            with pytest.raises(TypeError):
                csv_metadata_handler._create_string_to_int_mapper(None)

        # Verify that the logger captured the failure with traceback
        assert any(record.exc_info for record in caplog.records if record.levelname == "ERROR")

        # Verify the 'extra' metadata (stored in the record object)
        error_record = next(r for r in caplog.records if r.levelname == "ERROR")
        assert error_record.status == "failed"
