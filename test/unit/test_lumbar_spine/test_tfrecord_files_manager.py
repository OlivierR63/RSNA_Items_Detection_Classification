# coding utf-8

import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from src.projects.lumbar_spine.tfrecord_files_manager import (
    _int64_feature,
    _float_list_feature,
    _bytes_feature,
    TFRecordFilesManager,
    EmptyDirectoryError
)
from unittest.mock import patch, MagicMock
from pathlib import Path
from contextlib import ExitStack


def test_int64_feature():
    """
    Verify that a single integer is correctly converted to an Int64List feature.
    """
    test_val = 42
    feature = _int64_feature(test_val)

    # Check if the feature is an instance of tf.train.Feature
    assert isinstance(feature, tf.train.Feature)

    # Verify the value inside the proto structure
    assert feature.int64_list.value[0] == test_val


def test_float_list_feature():
    """
    Verify that a list of floats is correctly converted to a FloatList feature.
    """
    test_vals = [1.5, 2.7, 3.14]
    feature = _float_list_feature(test_vals)

    # Verify it returns the correct feature type
    assert isinstance(feature, tf.train.Feature)

    # Compare the lists (using almost equal for float precision)
    assert len(feature.float_list.value) == len(test_vals)

    assert list(feature.float_list.value) == pytest.approx(test_vals, abs=1e-5)


def test_bytes_feature():
    """
    Verify that a byte string is correctly converted to a BytesList feature.
    """
    test_bytes = b"sample_data_string"
    feature = _bytes_feature(test_bytes)

    # Ensure the output is a Feature object
    assert isinstance(feature, tf.train.Feature)

    # Check if the byte value matches
    assert feature.bytes_list.value[0] == test_bytes


def test_bytes_feature_with_string():
    """
    Edge case: Ensure that a regular string (if passed) is handled or raises error.
    Note: tf.train.BytesList expects raw bytes.
    """
    test_str = "not bytes"

    # In practice, users should encode strings before passing them
    with pytest.raises(TypeError):
        _bytes_feature(test_str)


class TestTFRecordFilesManager():

    def test_generate_tfrecord_files_nominal(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        setup_dicom_tree_structure,
        caplog,
        tmp_path
    ):
        """
        Test the generation process using internal state.
        Verifies that the function iterates over internal data and writes to disk.
        """

        # Enter each context manager through the stack
        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            mock_tf_writer = stack.enter_context(patch('tensorflow.io.TFRecordWriter'))
            mock_writer_instance = mock_tf_writer.return_value.__enter__.return_value

            # Link the CSVMetadataHandler mock to the mock_metadata fixture (the DataFrame)
            # Directly set the 'merged' attribute on the mock's return instance
            csv_func_path = 'src.projects.lumbar_spine.tfrecord_files_manager.CSVMetadataHandler'
            mock_csv = stack.enter_context(patch(csv_func_path))
            instance = mock_csv.return_value
            instance.generate_metadata_dataframe.return_value = mock_encoded_metadata

            # Mock SimpleITK image and its conversion to numpy
            sitk_func_path = 'src.projects.lumbar_spine.tfrecord_files_manager.sitk'
            mock_sitk = stack.enter_context(patch(sitk_func_path))
            mock_sitk.ReadImage.return_value = MagicMock(spec=sitk.Image)

            # Create a fake 2D image array (e.g., 512*512)
            fake_array = np.zeros((512, 512), dtype=np.uint16)
            mock_sitk.GetArrayFromImage.return_value = fake_array

            # Mock TFRecordFilesManager._get_series_stats
            stack.enter_context(
                patch.object(mock_tfrecord_files_manager, '_get_series_stats')
            ).return_value = (0, 256)

            # Mock calculate_max_series_depth

            # Execute the function with the real signature
            mock_tfrecord_files_manager.generate_tfrecord_files()

            # Check that the writer was initialized 3 times (once per study)
            assert mock_tf_writer.call_count == 3

            # Verify the maximum number of files per series ("depth"):
            # With a 95 % chosen percentile rate, le series depth is 5 files per series.
            assert mock_tfrecord_files_manager.get_series_depth() == 5

            # Verify that the writer was called for each DICOM file instance
            # As a reminder : there are 3 series per study. ALWAYS.
            # Total expected calls:
            #   First study: 3 series * (3 images + 2 padding files)
            #   Second study: 3 series * (4 images + 1 padding file per series)
            #   Third study: 3 series * 5 images per series
            #   Total : 45 images
            assert mock_writer_instance.write.call_count == 45

    def test_generate_tfrecord_files_with_logger(
        self,
        mock_tfrecord_files_manager,
        mock_metadata,
        caplog,
        tmp_path
    ):
        """
        Verify that the logger is correctly used if provided.
        """

        # Enter each context manager through the stack
        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            stack.enter_context(patch('tensorflow.io.TFRecordWriter'))

            # Link the CSVMetadataHandler mock to the mock_metadata fixture (the DataFrame)
            # Directly set the 'merged' attribute on the mock's return instance
            csv_func_path = 'src.projects.lumbar_spine.tfrecord_files_manager.CSVMetadataHandler'
            stack.enter_context(patch(csv_func_path)).return_value.merged = mock_metadata

            # Mock SimpleITK image and its conversion to numpy
            sitk_func_path = 'src.projects.lumbar_spine.tfrecord_files_manager.sitk'
            mock_sitk = stack.enter_context(patch(sitk_func_path))
            mock_sitk.ReadImage.return_value = MagicMock(spec=sitk.Image)

            # Create a fake 2D image array (e.g., 512*512)
            fake_array = np.zeros((512, 512), dtype=np.uint16)
            mock_sitk.GetArrayFromImage.return_value = fake_array

            # Mock TFRecordFilesManager._get_series_stats
            stack.enter_context(
                patch.object(mock_tfrecord_files_manager, '_get_series_stats')
            ).return_value = (0, 256)

            # Call with a mock logger to check for interaction
            mock_tfrecord_files_manager.generate_tfrecord_files()

            # Check that the final success message was logged, using caplog
            # inspections
            assert "DICOM to TFRecord conversion completed." in caplog.text

    def test_generate_tfrecord_files_empty_state(
        self,
        mock_tfrecord_files_manager,
        mock_metadata,
        caplog,
        tmp_path
    ):
        """
        Test behavior when there are no studies in the internal state.
        """
        # Enter each context manager through the stack
        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            mock_tf_writer = stack.enter_context(patch('tensorflow.io.TFRecordWriter'))

            # Link the CSVMetadataHandler mock to the mock_metadata fixture (the DataFrame)
            # Directly set the 'merged' attribute on the mock's return instance
            csv_func_path = 'src.projects.lumbar_spine.tfrecord_files_manager.CSVMetadataHandler'
            stack.enter_context(patch(csv_func_path)).return_value.merged = mock_metadata

            # Mock SimpleITK image and its conversion to numpy
            sitk_func_path = 'src.projects.lumbar_spine.tfrecord_files_manager.sitk'
            mock_sitk = stack.enter_context(patch(sitk_func_path))
            mock_sitk.ReadImage.return_value = MagicMock(spec=sitk.Image)

            # Create a fake 2D image array (e.g., 512*512)
            fake_array = np.zeros((512, 512), dtype=np.uint16)
            mock_sitk.GetArrayFromImage.return_value = fake_array

            # Mock TFRecordFilesManager._get_series_stats
            stack.enter_context(
                patch.object(mock_tfrecord_files_manager, '_get_series_stats')
            ).return_value = (0, 256)

            # Initialize manager and simulate internal state
            # We update the mock_config toward a place with no study directory
            empty_studies_dir = tmp_path/"root/dicom_empty"
            mock_tfrecord_files_manager._config["paths"]["dicom_studies"] = empty_studies_dir
            empty_studies_dir.mkdir(parents=True, exist_ok=True)

            with pytest.raises(EmptyDirectoryError) as excinfo:
                mock_tfrecord_files_manager.generate_tfrecord_files()

            # Ensure no writing occurred if no data was present
            mock_tf_writer.assert_not_called()

            # Check the logs (Pytest / caplog style)
            assert f"No studies found in {empty_studies_dir}" in caplog.text

            # Access the string representation of the exception
            exception_msg = str(excinfo.value)

            # Check the exception message
            assert exception_msg == (
                f"No studies found in {empty_studies_dir}. "
                "Process stops immediately."
            )


class TestConvertDicomToTFrecords:

    # ---  Nominal case ---
    def test_convert_dicom_to_tfrecords_nominal(
        self,
        mock_tfrecord_files_manager,
        mock_config,
        mock_encoded_metadata,
        caplog,
        tmp_path
    ):
        # Use real structure created by setup_dicom_tree_structure
        paths_cfg = mock_config.get('paths')

        if paths_cfg is None:
            error_msg = (
                "Fatal error: the parameter 'paths' "
                "is required but was not found. "
                "Please check your YAML file structure."
            )
            raise ValueError(error_msg)

        studies_dir = paths_cfg.get("dicom_studies")
        tfrecord_dir = paths_cfg.get("tfrecord")

        if None in (studies_dir, tfrecord_dir):
            error_msg = (
                "Fatal error: the setting variables 'paths -> dicom_studies' "
                "and/or 'paths -> tfrecord' are required but at least one is missing. "
                "Please check your YAML file structure."
            )
            raise ValueError(error_msg)

        # Mock sub-calls to avoid heavy processing
        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            mock_setup_dir = stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_setup_tfrecord_directory'
                )
            )
            mock_setup_dir.return_value = Path(tfrecord_dir)

            mock_process = stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_process_study'
                )
            )

            mock_tfrecord_files_manager._convert_dicom_to_tfrecords(
                studies_dir,
                mock_encoded_metadata,
                str(tfrecord_dir)
            )

            # We expect 3 studies (1010, 1020, 1030) as defined in conftest.py
            assert mock_process.call_count == 3
            mock_setup_dir.assert_called_once()

    # --- Empty root folder ---
    def test_convert_dicom_to_tfrecords_empty_root(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        caplog,
        tmp_path
    ):
        # Create a truly empty directory
        empty_dir = tmp_path / "very_empty"
        empty_dir.mkdir()

        # Patch get_current_logger so the @log_method decorator doesn't crash
        with patch('src.core.utils.logger.get_current_logger') as mock_get_log:
            # Reuse the mock_logger from the manager
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            with pytest.raises(EmptyDirectoryError, match="No studies found") as excinfo:
                mock_tfrecord_files_manager._convert_dicom_to_tfrecords(
                    empty_dir,
                    mock_encoded_metadata,
                    "dummy_out"
                )

            # Verify critical log was sent: the log history keeps a trace of the mishaps
            assert f"No studies found in {empty_dir}. Process stops immediately" in caplog.text
            critical_record = next(r for r in caplog.records if r.levelname == "CRITICAL")
            assert critical_record.status == "failure"

        # Access the string representation of the exception
        exception_msg = str(excinfo.value)

        # Verify that the function actually stopped the process
        # and sent the proper message to the user.
        assert exception_msg == f"No studies found in {empty_dir}. Process stops immediately."

    # --- Empty study folder (skip it) ---
    def test_convert_dicom_to_tfrecords_skips_empty_study(
        self,
        mock_tfrecord_files_manager,
        mock_config,
        mock_encoded_metadata,
        caplog,
        tmp_path
    ):

        studies_dir = mock_config["paths"]["dicom_studies"]

        # Add an empty study folder
        empty_dir = studies_dir / "9999"
        empty_dir.mkdir()

        with ExitStack() as stack:
            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            # Enter other context managers through the stack
            stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_setup_tfrecord_directory'
                )
            )

            mock_process = stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_process_study'
                )
            )

            mock_tfrecord_files_manager._convert_dicom_to_tfrecords(
                studies_dir,
                mock_encoded_metadata,
                "out"
            )

            # Should still be 3 (the 9999 is skipped because any().iterdir is False)
            assert mock_process.call_count == 3

            assert_msg = f"Skipping empty directory {empty_dir} in root directory {studies_dir}"
            assert assert_msg in caplog.text

            warning_record = next(
                r for r in caplog.records if "Skipping empty directory" in r.message
            )
            assert warning_record.status == "Skipped studies"

    # --- Non folder item (skip it) ---
    def test_convert_dicom_to_tfrecords_skips_file(
        self,
        mock_tfrecord_files_manager,
        mock_config,
        mock_encoded_metadata,
        caplog,
        tmp_path
    ):

        studies_dir = mock_config["paths"]["dicom_studies"]

        # Add a parasite file
        (studies_dir / "garbage.txt").write_text("not a directory")

        with ExitStack() as stack:
            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            # Enter other context managers through the stack
            stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_setup_tfrecord_directory'
                )
            )

            mock_process = stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_process_study'
                )
            )

            mock_tfrecord_files_manager._convert_dicom_to_tfrecords(
                studies_dir,
                mock_encoded_metadata,
                "out"
            )

            # Verification of the warning log.
            # English comment: Filter records to find warnings about non-directory items
            warning_records = [
                r for r in caplog.records
                if r.levelname == "WARNING" and "Skipping non-directory item" in r.message
            ]

            # English comment: Ensure at least one such warning was logged
            assert len(warning_records) > 0, "Warning log for non-directory item was not found."

            # Ensure process_study was not called for the file
            assert mock_process.call_count == 3  # Should be 3 (from nominal structure in conftest)

    # --- Missing metadata ---
    def test_convert_dicom_to_tfrecords_missing_metadata(
        self,
        mock_tfrecord_files_manager,
        mock_config,
        caplog,
        tmp_path
    ):

        studies_dir = mock_config["paths"]["dicom_studies"]

        # Provide a DataFrame that doesn't contain study 1010
        partial_metadata = pd.DataFrame({"study_id": [2000, 3000]})

        # Enter each context manager through the stack
        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            stack.enter_context(
                patch.object(mock_tfrecord_files_manager, '_setup_tfrecord_directory')
            )

            mock_process = stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_process_study'
                )
            )
            mock_tfrecord_files_manager._convert_dicom_to_tfrecords(
                studies_dir,
                partial_metadata,
                "out"
            )

            # No studies should match
            mock_process.assert_not_called()

            # Verify that at least one warning was logged
            assert any(record.levelname == "WARNING" for record in caplog.records)

    # --- Critical failure (Exception raised) ---
    def test_convert_dicom_to_tfrecords_exception(
        self,
        mock_tfrecord_files_manager,
        mock_config,
        mock_encoded_metadata,
        caplog,
        tmp_path
    ):
        studies_dir = mock_config["paths"]["dicom_studies"]

        # Force _setup_tfrecord_directory to crash
        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_setup_tfrecord_directory',
                    side_effect=RuntimeError("Disk full")
                )
            )

            with pytest.raises(RuntimeError, match="Disk full") as excinfo:
                mock_tfrecord_files_manager._convert_dicom_to_tfrecords(
                    studies_dir,
                    mock_encoded_metadata,
                    "out"
                )

            # Verify error was logged before re-raising
            assert any(record.levelname == "ERROR" for record in caplog.records)

            # Access the string representation of the exception
            exception_msg = str(excinfo.value)

            # Exact equality check
            assert "Disk full" in exception_msg


class TestSetupTfrecordDirectory:

    # --- Nominal Cases ---

    def test_setup_tfrecord_directory_creates_new_folder(
        self,
        mock_tfrecord_files_manager,
        caplog,
        tmp_path
    ):
        """
        Verify that the directory is created if it does not exist.
        """
        new_dir = tmp_path / "output_data"

        # Ensure it doesn't exist before
        assert not new_dir.exists()

        result_path = mock_tfrecord_files_manager._setup_tfrecord_directory(str(new_dir))

        assert result_path.exists()
        assert result_path.is_dir()
        assert result_path == new_dir

    def test_setup_tfrecord_directory_already_exists(
        self,
        mock_tfrecord_files_manager,
        caplog,
        tmp_path
    ):
        """
        Verify that it doesn't crash if the directory already exists.
        """
        existing_dir = tmp_path / "existing_folder"
        existing_dir.mkdir()

        # Should not raise FileExistsError thanks to exist_ok=True
        result_path = mock_tfrecord_files_manager._setup_tfrecord_directory(str(existing_dir))

        assert result_path.exists()
        assert result_path == existing_dir

    def test_setup_tfrecord_directory_nested_creation(
        self,
        mock_tfrecord_files_manager,
        caplog,
        tmp_path
    ):
        """
        Verify that parents are created if they don't exist.
        """
        nested_dir = tmp_path / "level1" / "level2" / "target"

        result_path = mock_tfrecord_files_manager._setup_tfrecord_directory(str(nested_dir))

        assert result_path.exists()
        assert result_path.is_dir()

    # --- Edge Cases & Errors ---

    def test_setup_tfrecord_directory_path_is_a_file(
        self,
        mock_tfrecord_files_manager,
        caplog,
        tmp_path
    ):
        """
        Verify error if the path points to an existing file instead of a directory.
        """
        file_path = tmp_path / "im_a_file.txt"
        file_path.write_text("dummy content")

        # Pathlib.mkdir raises FileExistsError if a file exists with the same name
        with pytest.raises(FileExistsError):
            mock_tfrecord_files_manager._setup_tfrecord_directory(str(file_path))

    def test_setup_tfrecord_directory_permission_denied(
        self,
        mock_tfrecord_files_manager,
        caplog,
        tmp_path
    ):
        """
        Verify behavior when there are no write permissions (Simulation).
        """
        # Note: Handling permissions on Windows/Linux can be tricky in tests.
        # We can mock mkdir to raise a PermissionError.
        with patch.object(Path, 'mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError, match="Permission denied"):
                mock_tfrecord_files_manager._setup_tfrecord_directory("any_path")


class TestProcessStudy:

    # --- Nominal Case ---
    def test_process_study_nominal(
        self,
        mock_tfrecord_files_manager,
        setup_dicom_tree_structure,
        mock_encoded_metadata,
        caplog
    ):
        """
        Verify the successful end-to-end flow of the _process_study method.
        This test ensures that:
        1. The study directory is correctly identified from the DICOM tree.
        2. The TFRecordWriter is initialized without errors.
        3. The internal series processing method is called the expected number of times.
        4. Success logs and 'extra' metadata (status='success') are correctly recorded.
        """
        studies_dir = setup_dicom_tree_structure

        # Use study 1010 which is present in mock_encoded_metadata
        study_path = studies_dir / "1010"

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    'src.core.utils.logger.get_current_logger',
                    return_value=mock_tfrecord_files_manager._logger
                )
            )
            stack.enter_context(patch('tensorflow.io.TFRecordWriter'))

            # Mock successful processing of the series
            mock_p = stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_process_single_series_instance'
                )
            )

            mock_p.side_effect = [
                (5, 0, 3, 2),
                (5, 0, 4, 1),
                (5, 0, 5, 0)
            ]

            mock_tfrecord_files_manager._process_study(
                study_path,
                mock_encoded_metadata,
                studies_dir / "out"
            )

            assert mock_p.call_count == 3
            success_record = next(
                rec for rec in caplog.records
                if rec.levelname == "INFO" and "completed successfully" in rec.message
            )
            assert success_record.status == "success"

    # 2. --- Skips Files (Parasites) ---
    def test_process_study_skips_files(
        self,
        mock_tfrecord_files_manager,
        setup_dicom_tree_structure,
        mock_encoded_metadata,
        caplog
    ):
        """
        Verify that non-directory items within a study folder are ignored.

        This test ensures that:
        1. Files (like 'parasite.txt') do not cause the processing to crash.
        2. A specific warning log is generated for each non-directory item encountered.
        3. The function continues its execution for valid sub-directories.
        """
        studies_dir = setup_dicom_tree_structure
        study_path = studies_dir / "1010"
        (study_path / "parasite.txt").touch()

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    'src.core.utils.logger.get_current_logger',
                    return_value=mock_tfrecord_files_manager._logger
                )
            )
            stack.enter_context(patch('tensorflow.io.TFRecordWriter'))
            stack.enter_context(
                patch.object(
                    TFRecordFilesManager,
                    '_get_series_stats',
                    return_value=(0, 255)
                )
            )
            stack.enter_context(
                patch.object(
                    TFRecordFilesManager,
                    '_process_single_series_instance',
                    return_value=(5, 0, 3, 2)
                )
            )

            mock_tfrecord_files_manager._process_study(
                study_path,
                mock_encoded_metadata,
                studies_dir / "out"
            )

            # Verify warning for non-directory item
            assert any(
                "Skipping non-directory" in rec.message
                for rec in caplog.records
                if rec.levelname == "ERROR"
            )

    # 3. --- Writer Error (Disk Full) ---
    def test_process_study_writer_error(
        self,
        mock_tfrecord_files_manager,
        setup_dicom_tree_structure,
        mock_encoded_metadata,
        caplog
    ):

        """
        Verify that I/O errors during TFRecord writing are correctly caught, logged, and re-raised.
        This test ensures that:
        1. Critical system errors (like a full disk) stop the process for the current study.
        2. The error is logged with 'ERROR' level including the study ID.
        3. The log record contains the correct metadata (status='failed').
        4. The exception propagates upwards to be handled by the main pipeline caller.
        """

        studies_dir = setup_dicom_tree_structure
        study_path = studies_dir / "1010"

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    'src.core.utils.logger.get_current_logger',
                    return_value=mock_tfrecord_files_manager._logger
                )
            )
            stack.enter_context(
                patch(
                    'tensorflow.io.TFRecordWriter',
                    side_effect=IOError("Disk Full")
                )
            )

            # Track if processing started despite the early writer error
            mock_p = stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_process_single_series_instance'
                )
            )

            with pytest.raises(IOError, match="Disk Full"):
                mock_tfrecord_files_manager._process_study(
                    study_path,
                    mock_encoded_metadata,
                    studies_dir / "out"
                )

        # Validate the specific error record exists
        error_record = next(
            rec for rec in caplog.records
            if rec.levelname == "ERROR"
        )

        # Validate error message and critical 'extra' metadata for traceability
        assert "Failed to process study 1010" in error_record.message
        assert error_record.status == "failed"
        assert error_record.study_id == "1010"

        # Ensure the exception stack trace was captured (exc_info=True)
        assert error_record.exc_info is not None

        # Verify that no series were processed because the writer failed at init
        mock_p.assert_not_called()

    # 4. --- Empty Study Directory ---
    def test_process_study_empty_study_dir(
            self,
            mock_tfrecord_files_manager,
            setup_dicom_tree_structure,
            mock_encoded_metadata,
            caplog
    ):
        """
        Verify that a study directory with no sub-directories is handled as a failure.
        This test ensures that:
        1. An empty directory (no series) doesn't cause a crash during iteration.
        2. The system identifies this as a 'failure' status because no data can be processed.
        3. The log message specifically mentions the empty study ID for easier debugging.
        4. Counters (nb_series, skipped_series) are correctly set to zero.
        """
        studies_dir = setup_dicom_tree_structure

        # Create a new, completely empty study directory
        empty_study_path = studies_dir / "9999"
        empty_study_path.mkdir(parents=True, exist_ok=True)

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    'src.core.utils.logger.get_current_logger',
                    return_value=mock_tfrecord_files_manager._logger
                )
            )
            stack.enter_context(patch('tensorflow.io.TFRecordWriter'))

            with pytest.raises(EmptyDirectoryError, match="is empty"):
                mock_tfrecord_files_manager._process_study(
                    empty_study_path,
                    mock_encoded_metadata,
                    studies_dir / "out"
                )

            # Target the specific log for empty studies
            record = next(
                rec for rec in caplog.records
                if "Study 9999 is empty" in rec.message and hasattr(rec, 'status')
            )
            assert record.status == "failure"
            assert record.nb_series == 0
            assert record.skipped_series == 0  # Since there was nothing to skip

    # 5. --- General Exception Handling ---
    def test_process_study_general_exception(
        self,
        mock_tfrecord_files_manager,
        setup_dicom_tree_structure,
        mock_encoded_metadata,
        caplog
    ):
        """
        Verify that any unexpected exception is caught, logged with traceback, and re-raised.

        This test ensures that:
        1. Unexpected errors (e.g., PermissionError) are handled by the general 'except' block.
        2. The error is logged with 'exc_info=True' (stack trace).
        3. The 'status' metadata is set to 'failed'.
        4. The exception is propagated to the caller.
        """
        studies_dir = setup_dicom_tree_structure
        study_path = studies_dir / "1010"

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    'src.core.utils.logger.get_current_logger',
                    return_value=mock_tfrecord_files_manager._logger
                )
            )

            # Simulate a random system error during directory iteration
            stack.enter_context(
                patch.object(
                    Path,
                    'iterdir',
                    side_effect=RuntimeError("Unexpected OS Error")
                )
            )

            # Verify the exception is re-raised
            with pytest.raises(RuntimeError, match="Unexpected OS Error"):
                mock_tfrecord_files_manager._process_study(
                    study_path,
                    mock_encoded_metadata,
                    studies_dir / "out"
                )

        # Verify the log from the 'except Exception as e' block
        error_record = next(
            rec for rec in caplog.records
            if "Failed to process study" in rec.message
        )

        assert error_record.levelname == "ERROR"
        assert error_record.status == "failed"
        assert error_record.study_id == "1010"

        # Verify that exc_info is present (traceback)
        assert error_record.exc_info is not None


class TestProcessSingleSeriesInstance:

    def test_process_single_series_success(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        mock_writer,
        caplog,
        tmp_path
    ):
        """
        Test a successful run using real directory paths from setup_dicom_tree_structure.
        Verifies that a valid study/series combination from the mock data
        is correctly identified and processed.
        """
        # Use the first series from our generated structure
        # Study 1010, Series 10110
        series_dir = tmp_path / "dicom/1010/10110"

        # Extract input features and target labels subsets for this study
        mask_1 = ['study_id', 'series_id', 'series_description']
        input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

        mask_2 = ['condition_level', 'severity', 'x', 'y']
        labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

        # Mock the final processing step
        # Note: logic is 'is_successful = not _process_series', so False means success
        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            mock_final = stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_process_series',
                    return_value=(5, 0, 3, 2)
                )
            )

            nb_success, nb_failures, nb_dicom_files, nb_padding_instances = (
                mock_tfrecord_files_manager._process_single_series_instance(
                    series_dir,
                    input_features_df,
                    labels_df,
                    mock_writer
                )
            )

            assert nb_success == 5
            assert nb_failures == 0
            assert nb_dicom_files == 3
            assert nb_padding_instances == 2

            mock_final.assert_called_once()

    def test_process_single_series_missing_metadata(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        mock_writer,
        caplog,
        tmp_path
    ):
        """
        Test behavior when a series exists on disk but is absent from the CSV metadata.
        Verifies that the process skips the series and returns False.
        """
        # Series 10110 exists, but we filter input_features_df to exclude it
        series_dir = tmp_path / "dicom/1010/10110"
        mask = (mock_encoded_metadata['series_id'] != 10110)
        metadata_df = mock_encoded_metadata[mask]
        input_features_df = (
            metadata_df[['study_id', 'series_id', 'series_description']]  # Force removal
            .drop_duplicates()
        )

        labels_df = (
            mock_encoded_metadata[['condition_level', 'severity', 'x', 'y']]
            .drop_duplicates()
        )

        with patch('src.core.utils.logger.get_current_logger') as mock_get_log:
            mock_get_log.return_value = mock_tfrecord_files_manager._logger
            nb_success, nb_failures, nb_files = (
                mock_tfrecord_files_manager._process_single_series_instance(
                    series_dir,
                    input_features_df,
                    labels_df,
                    mock_writer
                )
            )

            assert nb_success == 0
            assert nb_failures == 0
            assert nb_files == 3

        # Ensure the specific missing metadata status was logged
        assert any(
            "No matching metadata found" in rec.message
            for rec in caplog.records
            if rec.levelname == "WARNING"
        )

    def test_process_single_series_is_not_dir(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        mock_writer,
        caplog,
        tmp_path
    ):
        """
        Test behavior when the path provided is a file instead of a directory.
        Verifies that the function correctly identifies the item type and skips it.
        """
        # Use an existing DICOM file from our structure as the 'series_path'
        fake_series_path = tmp_path / "root/dicom/1010/10110/1.dcm"

        mask_1 = ['study_id', 'series_id', 'series_description']
        input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

        mask_2 = ['condition_level', 'severity', 'x', 'y']
        labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

        with patch('src.core.utils.logger.get_current_logger') as mock_get_log:
            mock_get_log.return_value = mock_tfrecord_files_manager._logger
            nb_success, nb_failures, nb_files = (
                mock_tfrecord_files_manager._process_single_series_instance(
                    fake_series_path,
                    input_features_df,
                    labels_df,
                    mock_writer
                )
            )

            assert nb_success == 0
            assert nb_failures == 0
            assert nb_files == 0

        assert any(
            "Skipping non-directory item" in rec.message
            for rec in caplog.records
            if rec.levelname == "WARNING"
        )

    def test_process_single_series_invalid_name(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        mock_writer,
        caplog,
        tmp_path
    ):
        """
        Test behavior when the directory name is not a valid integer.
        Verifies that the casting to int(series_path.name) raises a ValueError.
        """
        # Create a folder that doesn't follow the numeric ID convention
        invalid_dir = tmp_path / "root/dicom/1010/not_an_id"
        invalid_dir.mkdir(parents=True)

        mask_1 = ['study_id', 'series_id', 'series_description']
        input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

        mask_2 = ['condition_level', 'severity', 'x', 'y']
        labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

        with patch('src.core.utils.logger.get_current_logger') as mock_get_log:
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            with pytest.raises(ValueError):
                mock_tfrecord_files_manager._process_single_series_instance(
                    invalid_dir, input_features_df, labels_df, mock_writer
                )


class TestProcessSeries:

    def test_process_series_full_success(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        mock_writer,
        caplog,
        tmp_path
    ):
        """
        Test the case where all DICOM files in a series are processed successfully.
        Verifies that the returned tuple matches (5, 0, 3, 2) for the 10110 series.
        """
        # Use Study 1010, Series 10110 (contains 3 .dcm files)
        series_dir = tmp_path / "dicom/1010/10110"

        with ExitStack() as stack:
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            # Mock dependencies to avoid real DICOM heavy lifting
            stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_get_series_stats',
                    return_value=(0, 255)
                )
            )

            # Mock individual DICOM processing to always succeed (return_value=True)
            mock_single = stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_process_single_dicom_instance',
                    return_value=True
                )
            )

            mask_1 = ['study_id', 'series_id', 'series_description']
            input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

            mask_2 = ['condition_level', 'severity', 'x', 'y']
            labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

            success, failures, nb_dicom_files, nb_padding_instances = (
                mock_tfrecord_files_manager._process_series(
                    series_dir,
                    input_features_df,
                    labels_df,
                    mock_writer
                )
            )

            # There are 3 DICOM files, however 2 more padding files must be generated,
            # because config['series_depth'] = 5
            assert success == 5
            assert failures == 0
            assert nb_dicom_files == 3
            assert nb_padding_instances == 2

            assert mock_single.call_count == 5

            # Search for the log message using a partial match to avoid punctuation issues
            start_log = next(
                (
                    rec for rec in caplog.records
                    if "processing completed successfully" in rec.message
                ),
                None  # Means: start_log = None if the expected log is missing
            )

            # Ensure the record was actually found before asserting attributes
            assert_msg = f"Success log for series {series_dir.name} was not found in caplog"
            assert start_log is not None, assert_msg
            assert start_log.levelname == "INFO"
            assert start_log.status == "success"

    def test_process_series_partial_success(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        mock_writer,
        caplog,
        tmp_path
    ):
        """
        Test the case where 1 file fails and 2 succeed.
        Verifies that the returned tuple is (4, 1, 4, 1).
        """
        series_dir = tmp_path / "dicom/1020/10210"  # 3 files

        with ExitStack() as stack:
            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_get_series_stats',
                    return_value=(0, 255)
                )
            )

            # Mock side_effect: first file fails (False), next four succeed (True)
            mock_single = stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_process_single_dicom_instance',
                    side_effect=[False, True, True, True, True]
                )
            )

            mask_1 = ['study_id', 'series_id', 'series_description']
            input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

            mask_2 = ['condition_level', 'severity', 'x', 'y']
            labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

            success, failures, nb_dicom_files, nb_padding_instances = (
                mock_tfrecord_files_manager._process_series(
                    series_dir,
                    input_features_df,
                    labels_df,
                    mock_writer
                )
            )

            assert success == 4
            assert failures == 1
            assert nb_dicom_files == 4
            assert nb_padding_instances == 1
            assert mock_single.call_count == 5

            # Filter caplog records to find warnings containing the specific substring
            warning_records = [
                rec for rec in caplog.records
                if rec.levelname == "WARNING" and "partially completed" in rec.message
            ]

            # Verify that at least one such warning was logged
            assert len(warning_records) == 1
            assert warning_records[0].failed_processing == 1

    def test_process_series_complete_failure(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        mock_writer,
        caplog,
        tmp_path
    ):
        """
        Test the case where all files fail (3/3).
        Verifies that the returned tuple is (0, 5, 4, 1).
        """
        series_dir = tmp_path / "dicom/1020/10210"

        with ExitStack() as stack:
            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_get_series_stats',
                    return_value=(0, 255)
                )
            )

            # All files fail (return_value = False)
            mock_single = stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_process_single_dicom_instance',
                    return_value=False
                )
            )

            mask_1 = ['study_id', 'series_id', 'series_description']
            input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

            mask_2 = ['condition_level', 'severity', 'x', 'y']
            labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

            success, failures, nb_dicom_files, nb_padding_instances = (
                mock_tfrecord_files_manager._process_series(
                    series_dir,
                    input_features_df,
                    labels_df,
                    mock_writer
                )
            )

            # Result is True because it IS a complete failure
            assert mock_single.call_count == 5
            assert success == 0
            assert failures == 5
            assert nb_dicom_files == 4
            assert nb_padding_instances == 1

            # Define the expected error message substring
            error_msg = f"Series {series_dir.name} processing failed"

            # Find the specific error record in caplog
            error_record = next(
                (
                    rec for rec in caplog.records
                    if error_msg in rec.message and rec.levelname == "ERROR"
                ),
                None
            )

            # Validate that the log exists and contains the correct metadata
            assert error_record is not None, f"Error log for series {series_dir.name} was not found"
            assert "All files failed during processing" in error_record.message
            assert error_record.status == "failed"

            # Verify that exception info (traceback) was captured
            assert error_record.exc_info is not None

    def test_process_series_no_dicom_files(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        mock_writer,
        caplog,
        tmp_path
    ):
        """
        Test behavior when a series directory exists but contains no .dcm files.
        Verifies that this is treated as a complete failure.
        """
        # Create an empty directory
        empty_series = tmp_path / "dicom/empty_series"
        empty_series.mkdir(parents=True)

        with ExitStack() as stack:
            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_get_series_stats',
                    return_value=(0, 0)
                )
            )

            mock_single = stack.enter_context(
                patch.object(mock_tfrecord_files_manager, '_process_single_dicom_instance')
            )

            mask_1 = ['study_id', 'series_id', 'series_description']
            input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

            mask_2 = ['condition_level', 'severity', 'x', 'y']
            labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

            success, failures, nb_dicom_files, nb_padding_instances = (
                mock_tfrecord_files_manager._process_series(
                    empty_series,
                    input_features_df,
                    labels_df,
                    mock_writer
                )
            )

            assert success == 5
            assert failures == 0
            assert nb_dicom_files == 0
            assert nb_padding_instances == 5
            assert mock_single.call_count == 5

    # 5. --- General Exception Handling ---
    def test_process_series_general_exception(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        mock_writer,
        caplog,
        tmp_path
    ):
        """
        Verify that an unexpected exception during series processing is caught,
        logged, and re-raised.
        """
        series_dir = tmp_path / "root/dicom/1010/10110"

        with ExitStack() as stack:
            # Standard logger patch
            stack.enter_context(
                patch(
                    'src.core.utils.logger.get_current_logger',
                    return_value=mock_tfrecord_files_manager._logger
                )
            )

            # Simulate a critical failure during statistics calculation
            stack.enter_context(patch.object(
                mock_tfrecord_files_manager,
                '_get_series_stats',
                side_effect=RuntimeError("Hardware Read Error")
            ))

            mask_1 = ['study_id', 'series_id', 'series_description']
            input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

            mask_2 = ['condition_level', 'severity', 'x', 'y']
            labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

            # Check that the exception propagates to the caller
            with pytest.raises(RuntimeError, match="Hardware Read Error"):
                mock_tfrecord_files_manager._process_series(
                    series_dir,
                    input_features_df,
                    labels_df,
                    mock_writer
                )

        # Verify that the error was logged with appropriate metadata
        # Note: This assumes you have a try/except block in your _process_series source code
        error_record = next(
            (rec for rec in caplog.records if "Failed to process series" in rec.message),
            None
        )

        assert error_record is not None
        assert error_record.levelname == "ERROR"
        assert error_record.status == "failed"
        assert error_record.exc_info is not None


class TestGetSeriesStats:
    """
    Test suite for the _get_series_stats method, covering path handling,
    DICOM discovery, and pixel intensity aggregation.
    """

    def test_get_series_stats_nominal_aggregation(
        self,
        mock_tfrecord_files_manager,
        caplog,
        tmp_path
    ):
        """
        Tests that min and max are correctly aggregated across multiple DICOM files.
        """
        # Setup a dummy directory with 2 empty dcm files
        series_dir = tmp_path / "study/series_1"
        series_dir.mkdir(parents=True)
        (series_dir / "instance_1.dcm").touch()
        (series_dir / "instance_2.dcm").touch()

        # Mock first file: range [10, 100]
        mock_ds_1 = MagicMock()
        mock_ds_1.pixel_array = np.array([[10, 50], [50, 100]])

        # Mock second file: range [5, 150] -> global min should be 5
        mock_ds_2 = MagicMock()
        mock_ds_2.pixel_array = np.array([[5, 120], [80, 150]])

        with ExitStack() as stack:
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            stack.enter_context(patch('pydicom.dcmread', side_effect=[mock_ds_1, mock_ds_2]))

            global_min, global_max = mock_tfrecord_files_manager._get_series_stats(series_dir)

            assert global_min == 5
            assert global_max == 150
            assert isinstance(global_min, int)

    def test_get_series_stats_empty_directory_raises_error(
        self,
        mock_tfrecord_files_manager,
        caplog,
        tmp_path
    ):
        """
        Tests that FileNotFoundError is raised and logged when no .dcm files exist.
        """
        empty_dir = tmp_path / "empty_series"
        empty_dir.mkdir()

        with patch('src.core.utils.logger.get_current_logger') as mock_get_log:
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            with pytest.raises(FileNotFoundError):
                mock_tfrecord_files_manager._get_series_stats(empty_dir)

        # Verify the ERROR level record is present in caplog
        error_record = next((rec for rec in caplog.records if rec.levelname == "ERROR"), None)
        assert error_record is not None
        assert "No DICOM files found" in error_record.message

    def test_get_series_stats_corrupt_dicom_raises_exception(
        self,
        mock_tfrecord_files_manager,
        caplog,
        tmp_path
    ):
        """
        Tests that any exception during DICOM reading is re-raised and logged.
        """
        series_dir = tmp_path / "corrupt_series"
        series_dir.mkdir()
        (series_dir / "broken.dcm").touch()

        with ExitStack() as stack:
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            # Simulate a generic exception during dcmread
            stack.enter_context(patch('pydicom.dcmread', side_effect=Exception("Format error")))

            with pytest.raises(Exception):
                mock_tfrecord_files_manager._get_series_stats(series_dir)

        # Verify that the log contains exc_info (traceback)
        error_record = next((rec for rec in caplog.records if rec.levelname == "ERROR"), None)
        assert error_record is not None
        assert "Function TFRecordFilesManager._get_series_stats failed" in error_record.message
        assert error_record.exc_info is not None

    def test_get_series_stats_constant_volume(self, mock_tfrecord_files_manager, caplog, tmp_path):
        """
        Tests the edge case where all pixels in all files have the same value.
        """
        series_dir = tmp_path / "constant_series"
        series_dir.mkdir()
        (series_dir / "flat.dcm").touch()

        mock_ds = MagicMock()
        mock_ds.pixel_array = np.full((5, 5), 500)

        with ExitStack() as stack:
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            stack.enter_context(patch('pydicom.dcmread', return_value=mock_ds))
            s_min, s_max = mock_tfrecord_files_manager._get_series_stats(series_dir)

            assert s_min == 500
            assert s_max == 500

    def test_get_series_stats_accepts_string_path(
        self,
        mock_tfrecord_files_manager,
        caplog,
        tmp_path
    ):
        """
        Verifies that the method handles string input by converting it to a Path object.
        """
        series_dir = tmp_path / "string_path_test"
        series_dir.mkdir()
        (series_dir / "test.dcm").touch()

        mock_ds = MagicMock()
        mock_ds.pixel_array = np.array([0, 1])

        with ExitStack() as stack:
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            stack.enter_context(patch('pydicom.dcmread', return_value=mock_ds))

            # Passing the path as a string instead of Path object
            s_min, s_max = mock_tfrecord_files_manager._get_series_stats(str(series_dir))

            assert s_min == 0
            assert s_max == 1

    def test_get_series_stats_general_exception(
        self,
        mock_tfrecord_files_manager,
        caplog,
        tmp_path
    ):
        """
        Verify that any unexpected exception is caught, logged with traceback, and re-raised.
        """

        # Setup directory with at least one file to enter the loop
        series_dir = tmp_path / "crash_series"
        series_dir.mkdir()
        (series_dir / "trigger.dcm").touch()

        with ExitStack() as stack:
            # Mock the logger to capture through caplog
            stack.enter_context(patch('src.core.utils.logger.get_current_logger',
                                      return_value=mock_tfrecord_files_manager._logger))

            # Simulate a critical failure (e.g., MemoryError or OS-level issue)
            # during the pydicom read process
            stack.enter_context(patch('pydicom.dcmread',
                                      side_effect=RuntimeError("Critical IO Failure")))

            # Ensure the exception propagates
            with pytest.raises(RuntimeError, match="Critical IO Failure"):
                mock_tfrecord_files_manager._get_series_stats(series_dir)

        # Verify that the error was logged before the exception was re-raised
        error_record = next((rec for rec in caplog.records if rec.levelname == "ERROR"), None)
        assert error_record is not None
        assert "Function TFRecordFilesManager._get_series_stats failed" in error_record.message

        # Check that exc_info is present (this confirms the use of exc_info=True)
        assert error_record.exc_info is not None


class TestProcessSingleDicomInstance:
    """
    Test suite for _process_single_dicom_instance.
    Focuses on orchestration between metadata extraction and TFRecord writing.
    """

    def test_process_single_dicom_success(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        mock_writer,
        caplog,
        tmp_path
    ):
        """
        Test a successful processing flow.
        Verifies that features are serialized and written, and returns True.
        """
        dicom_path = tmp_path / "mock/path/slice1.dcm"

        # Mocking the feature object returned by _process_dicom_file_with_metadata
        mock_features = MagicMock()
        mock_features.SerializeToString.return_value = b"serialized_proto_data"

        with ExitStack() as stack:
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            mock_process = stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_process_dicom_file_with_metadata',
                    return_value=mock_features
                )
            )

            mask_1 = ['study_id', 'series_id', 'series_description']
            input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

            mask_2 = ['condition_level', 'severity', 'x', 'y']
            labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

            result = mock_tfrecord_files_manager._process_single_dicom_instance(
                series_path=dicom_path.parent,
                series_min=0,
                series_max=2000,
                input_features_df=input_features_df,
                labels_df=labels_df,
                writer=mock_writer,
                instance_num=dicom_path.stem,
                is_padding=False
            )

            # Verify method calls
            mock_process.assert_called_once_with(
                dicom_path, 0, 2000, input_features_df, labels_df
            )
            mock_writer.write.assert_called_once_with(b"serialized_proto_data")

            # According to docstring, True means success
            assert result is True

    def test_process_single_dicom_extraction_failure(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        mock_writer,
        caplog,
        tmp_path
    ):
        """
        Test failure during the metadata extraction phase.
        Verifies that an exception is caught, logged, and returns False.
        """
        dicom_path = Path("/mock/path/corrupt.dcm")

        with ExitStack() as stack:
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            # Simulate an exception during extraction (e.g. pydicom error)
            stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_process_dicom_file_with_metadata',
                    side_effect=Exception("Extraction Error")
                )
            )

            mask_1 = ['study_id', 'series_id', 'series_description']
            input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

            mask_2 = ['condition_level', 'severity', 'x', 'y']
            labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

            result = mock_tfrecord_files_manager._process_single_dicom_instance(
                series_path=dicom_path.parent,
                series_min=0,
                series_max=1000,
                input_features_df=input_features_df,
                labels_df=labels_df,
                writer=mock_writer,
                instance_num=dicom_path.stem,
                is_padding=False
            )

            # Verify failure handling via caplog
            assert result is False

            error_log = next((rec for rec in caplog.records if rec.levelname == "ERROR"), None)
            assert error_log is not None
            assert "Extraction Error" in error_log.message

            # Writer should not have been called
            mock_writer.write.assert_not_called()

    def test_process_single_dicom_write_failure(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        mock_writer,
        caplog,
        tmp_path
    ):
        """
        Test failure during the TFRecord writing phase.
        Verifies that even if extraction works, a write error returns False.
        """
        dicom_path = Path("/mock/path/slice1.dcm")
        mock_features = MagicMock()

        # Extraction succeeds, but writer.write raises an OSError (e.g. disk full)
        mock_writer.write.side_effect = Exception("IO Error")

        with ExitStack() as stack:
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_process_dicom_file_with_metadata',
                    return_value=mock_features
                )
            )

            mask_1 = ['study_id', 'series_id', 'series_description']
            input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

            mask_2 = ['condition_level', 'severity', 'x', 'y']
            labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

            result = mock_tfrecord_files_manager._process_single_dicom_instance(
                series_path=dicom_path.parent,
                series_min=0,
                series_max=1000,
                input_features_df=input_features_df,
                labels_df=labels_df,
                writer=mock_writer,
                instance_num=dicom_path.stem,
                is_padding=False
            )

            assert result is False

            error_log = next((rec for rec in caplog.records if rec.levelname == "ERROR"), None)
            assert error_log is not None
            assert "IO Error" in error_log.message

    def test_process_single_dicom_with_custom_logger(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        mock_writer,
        caplog,
        tmp_path
    ):
        """
        Verifies that a custom logger passed as argument is used instead of the default one.
        """
        custom_logger = MagicMock()
        dicom_path = tmp_path / "path/error.dcm"

        with ExitStack() as stack:
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_process_dicom_file_with_metadata',
                    side_effect=Exception("Log Test")
                )
            )

            mask_1 = ['study_id', 'series_id', 'series_description']
            input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

            mask_2 = ['condition_level', 'severity', 'x', 'y']
            labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

            _ = mock_tfrecord_files_manager._process_single_dicom_instance(
                series_path=dicom_path.parent,
                series_min=0,
                series_max=1000,
                input_features_df=input_features_df,
                labels_df=labels_df,
                writer=mock_writer,
                instance_num=dicom_path.stem,
                is_padding=False,
                logger=custom_logger
            )

            # The custom logger should receive the error call
            # Custom loggers passed as Mocks bypass the standard logging
            # hierarchy, so assert_called() is the only way to verify interaction as
            # pytest's caplog fixture will not capture these calls.
            custom_logger.error.assert_called()

            # The default manager logger should not be used
            assert not any(rec.levelname == "ERROR" for rec in caplog.records)

    def test_process_single_dicom_instance_general_exception(
        self,
        mock_tfrecord_files_manager,
        caplog,
        tmp_path
    ):
        """
        Test that an unexpected exception during DICOM processing is caught and logged.
        Verifies the function returns False instead of crashing.
        """
        # Setup dummy path and inputs
        dicom_path = tmp_path / "1.dcm"
        mock_writer = MagicMock()

        with ExitStack() as stack:
            # Inject logger for caplog capture
            stack.enter_context(patch('src.core.utils.logger.get_current_logger',
                                      return_value=mock_tfrecord_files_manager._logger))

            # Simulate a failure in the delegation method (e.g., pydicom error)
            stack.enter_context(patch.object(
                mock_tfrecord_files_manager,
                '_process_dicom_file_with_metadata',
                side_effect=RuntimeError("Unexpected processing error")
            ))

            # Call the method and verify it returns False (as per docstring)
            result = mock_tfrecord_files_manager._process_single_dicom_instance(
                series_path=dicom_path.parent,
                series_min=0,
                series_max=255,
                input_features_df=pd.DataFrame(),
                labels_df=pd.DataFrame(),
                writer=mock_writer,
                instance_num=dicom_path.stem,
                is_padding=False
            )

            assert result is False

        # Verify the error log contains metadata and traceback
        error_record = next((rec for rec in caplog.records if rec.levelname == "ERROR"), None)
        assert error_record is not None
        assert "DicomProcessingError" in str(error_record.__dict__)
        assert "Unexpected processing error" in error_record.message
        assert error_record.exc_info is not None


class TestProcessDicomFileWithMetadata:
    """
    Test suite for _process_dicom_file_with_metadata.
    Verifies image processing, metadata merging, and TF Feature preparation.
    """

    def test_process_dicom_success(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        mock_writer,
        caplog,
        tmp_path
    ):
        """
        Test the successful conversion of a DICOM file to a tf.train.Example.
        """
        # Setup mock file structure and metadata
        dicom_path = tmp_path / "dicom/1010/10110/555.dcm"
        series_id = 10110
        study_id = 1010

        # Prepare mock image data (SimpleITK)
        mock_img = MagicMock()
        mock_array = np.zeros((512, 512), dtype=np.uint16)

        # Prepare features dictionary to be returned by _prepare_tf_features
        # Use the actual helper functions to create valid Protobuf objects
        # This avoids the TypeError: Parameter to CopyFrom() must be instance of same class

        mock_features_dict = {
            'image': _bytes_feature(b"fake_image_data"),
            'study_id': _int64_feature(1010),
            'series_id': _int64_feature(10110),
            'series_min': _int64_feature(15),
            'series_max': _int64_feature(550),
            'instance_number': _int64_feature(1),
            'img_height': _int64_feature(224),
            'img_width': _int64_feature(224),
            'series_description': _int64_feature(0)
        }

        with ExitStack() as stack:

            # Mock SimpleITK reading and array conversion
            stack.enter_context(patch('SimpleITK.ReadImage', return_value=mock_img))
            stack.enter_context(patch('SimpleITK.GetArrayFromImage', return_value=mock_array))

            # Mock the feature preparation and logger
            mock_prepare = stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_prepare_tf_features',
                    return_value=mock_features_dict
                )
            )

            stack.enter_context(
                patch(
                    'src.core.utils.logger.get_current_logger',
                    return_value=mock_tfrecord_files_manager._logger
                )
            )

            mask_1 = ['study_id', 'series_id', 'actual_file_format', 'series_description']
            input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

            mask_2 = ['condition_level', 'severity', 'x', 'y']
            labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

            result = mock_tfrecord_files_manager._process_single_dicom_instance(
                series_path=dicom_path.parent,
                series_min=0,
                series_max=2000,
                input_features_df=input_features_df,
                labels_df=labels_df,
                writer=mock_writer,
                instance_num=dicom_path.stem,
                is_padding=False
            )

            # Check if the orchestration was correct
            assert result is True
            mock_prepare.assert_called_once()

            # Verify metadata extraction from path and dataframe
            args, kwargs = mock_prepare.call_args
            assert kwargs['series_id'] == series_id
            assert kwargs['instance_id'] == 555
            assert kwargs['study_id'] == study_id

    def test_process_dicom_missing_description_error(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        caplog,
        tmp_path
    ):
        """
        Test that a ValueError is raised when the key 'series_description'
        is missing for the series.
        """

        # Series 99999 not in mock_encoded_metadata
        dicom_path = tmp_path / "dicom/1010/99999/1.dcm"

        with ExitStack() as stack:
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            stack.enter_context(
                patch(
                    'SimpleITK.ReadImage',
                    return_value=MagicMock()
                )
            )

            stack.enter_context(
                patch(
                    'SimpleITK.GetArrayFromImage',
                    return_value=np.zeros((10, 10))
                )
            )

            mask_1 = ['study_id', 'series_id', 'actual_file_format', 'series_description']
            input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

            mask_2 = ['condition_level', 'severity', 'x', 'y']
            labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

            with pytest.raises(ValueError) as exc_info:
                mock_tfrecord_files_manager._process_dicom_file_with_metadata(
                    dicom_path=dicom_path,
                    series_min=0,
                    series_max=1000,
                    input_features_df=input_features_df,
                    labels_df=labels_df
                )

            assert "No matching data in file" in str(exc_info.value)

            # Verify the log message contains the IDs that failed the lookup
            error_record = next((rec for rec in caplog.records if rec.levelname == "ERROR"), None)
            assert error_record is not None
            assert "No matching data in file" in error_record.message

    def test_process_dicom_read_exception(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        caplog,
        tmp_path
    ):
        """
        Test that exceptions during SimpleITK read are logged and re-raised.
        """
        dicom_path = tmp_path / "dicom/1010/10110/1.dcm"

        with ExitStack() as stack:
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            stack.enter_context(
                patch(
                    'SimpleITK.ReadImage',
                    side_effect=Exception("SITK Read Error")
                )
            )

            mask_1 = ['study_id', 'series_id', 'actual_file_format', 'series_description']
            input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

            mask_2 = ['condition_level', 'severity', 'x', 'y']
            labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

            with pytest.raises(Exception) as exc_info:
                mock_tfrecord_files_manager._process_dicom_file_with_metadata(
                    dicom_path=dicom_path,
                    series_min=0,
                    series_max=1000,
                    input_features_df=input_features_df,
                    labels_df=labels_df
                )

            assert "SITK Read Error" in str(exc_info.value)

            # Verify that the error was logged with the full message
            error_record = next((rec for rec in caplog.records if rec.levelname == "ERROR"), None)
            assert error_record is not None
            assert "_process_dicom_file_with_metadata" in error_record.message
            assert "SITK Read Error" in error_record.message

    def test_process_dicom_data_type_conversion(
        self,
        mock_tfrecord_files_manager,
        mock_encoded_metadata,
        caplog,
        tmp_path
    ):
        """
        Ensures that the image array is correctly cast to uint16 before byte conversion.
        """
        dicom_path = tmp_path / "dicom/1010/10110/1.dcm"

        # Simulate float array from SITK (common with some rescaled DICOMs)
        float_array = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)

        with ExitStack() as stack:
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_tfrecord_files_manager._logger

            stack.enter_context(patch('SimpleITK.ReadImage', return_value=MagicMock()))
            stack.enter_context(patch('SimpleITK.GetArrayFromImage', return_value=float_array))
            mock_prepare = stack.enter_context(
                patch.object(
                    mock_tfrecord_files_manager,
                    '_prepare_tf_features',
                    return_value={}
                )
            )

            mask_1 = ['study_id', 'series_id', 'actual_file_format', 'series_description']
            input_features_df = mock_encoded_metadata[mask_1].drop_duplicates()

            mask_2 = ['condition_level', 'severity', 'x', 'y']
            labels_df = mock_encoded_metadata[mask_2].drop_duplicates()

            mock_tfrecord_files_manager._process_dicom_file_with_metadata(
                dicom_path, 0, 100, input_features_df, labels_df
            )

            # Check image_bytes in _prepare_tf_features call
            passed_bytes = mock_prepare.call_args[1]['image_bytes']
            expected_bytes = float_array.astype(np.uint16).tobytes()
            assert passed_bytes == expected_bytes

    def test_process_dicom_file_general_exception(
        self,
        mock_tfrecord_files_manager,
        caplog,
        tmp_path
    ):
        """
        Test that any unexpected exception (e.g., SimpleITK failure) is caught,
        logged with traceback, and then re-raised.
        """
        # Setup minimal directory structure to allow ID extraction
        dicom_path = tmp_path / "1010/10110/1.dcm"
        dicom_path.parent.mkdir(parents=True)
        dicom_path.touch()

        with ExitStack() as stack:
            # Inject logger for caplog capture
            stack.enter_context(
                patch(
                    'src.core.utils.logger.get_current_logger',
                    return_value=mock_tfrecord_files_manager._logger
                )
            )

            # Simulate a critical failure during SimpleITK image reading
            stack.enter_context(
                patch(
                    'SimpleITK.ReadImage',
                    side_effect=RuntimeError("SITK Critical Error")
                )
            )

            # Verify that the exception is re-raised to the caller
            with pytest.raises(RuntimeError, match="SITK Critical Error"):
                mock_tfrecord_files_manager._process_dicom_file_with_metadata(
                    dicom_path=dicom_path,
                    series_min=0,
                    series_max=255,
                    input_features_df=pd.DataFrame(),  # Not reached due to SITK error
                    labels_df=pd.DataFrame()
                )

        # Verify the ERROR log contains the custom message and the traceback
        error_record = next((rec for rec in caplog.records if rec.levelname == "ERROR"), None)
        assert error_record is not None

        assert_msg = "The DICOM file read/conversion/serialization process for 1.dcm failed"
        assert assert_msg in error_record.message
        assert error_record.status == "failed"
        assert error_record.error_type == "DicomProcessingError"

        # Crucial: ensure exc_info is present for debugging
        assert error_record.exc_info is not None


class TestPrepareTfFeatures:
    """
    Test suite for _prepare_tf_features.
    Verifies fixed-position mapping and padding via reindexing.
    """

    def test_prepare_tf_features_fixed_positions(
        self,
        mock_tfrecord_files_manager,
        caplog,
        tmp_path
    ):
        """
        Nominal case: Verify that records are placed at specific indices
        based on their condition_level value, not their position in the DF.
        """
        # Providing level 1 and 3 out of order
        labels_df = pd.DataFrame({
            'condition_level': [3.0, 1.0],
            'severity': [1.0, 0.5],
            'x': [300.0, 100.0],
            'y': [400.0, 200.0]
        })

        nb_max_records = 5  # Total floats: 5 * 4 = 20

        result = mock_tfrecord_files_manager._prepare_tf_features(
            image_bytes=b"fake_image",
            study_id=1010,
            series_id=10110,
            series_min=0,
            series_max=2000,
            instance_id=1,
            img_height=512,
            img_width=512,
            description=3,
            labels_df=labels_df,
            nb_max_records=nb_max_records,
            is_padding=False
        )

        records_values = result['records'].float_list.value
        assert len(records_values) == 20

        # Level 0 (indices 0-3) should be 0.0 (filled by reindex)
        assert all(v == 0.0 for v in records_values[0:4])

        # Level 1 (indices 4-7) should contain the data for level 1.0
        assert records_values[4:8] == [1.0, 0.5, 100, 200]

        # Level 2 (indices 8-11) should be 0.0
        assert records_values[8:12] == [2, 0, 0, 0]

        # Level 3 (indices 12-15) should contain data for level 3.0
        assert records_values[12:16] == [3, 1, 300, 400]

        # Level 4 (indices 16-19) should be 0.0
        assert records_values[16:20] == [4, 0, 0, 0]

    def test_prepare_tf_features_out_of_range_selection(
        self,
        mock_tfrecord_files_manager,
        caplog,
        tmp_path
    ):
        """
        Verify that levels outside the 0 to (nb_max_records-1) range are excluded.
        """
        # Level 10 is beyond nb_max_records (5)
        labels_df = pd.DataFrame({
            'condition_level': [1.0, 10.0],
            'severity': [0, 0],
            'x': [0, 0],
            'y': [0, 0]
        })

        nb_max_records = 5

        result = mock_tfrecord_files_manager._prepare_tf_features(
            image_bytes=b"...",
            study_id=1,
            series_id=1,
            series_min=0,
            series_max=1,
            instance_id=1,
            img_height=1,
            img_width=1,
            description=1,
            labels_df=labels_df,
            nb_max_records=nb_max_records,
            is_padding=False
        )

        records_values = result['records'].float_list.value

        # Still 20 values, but level 10 is ignored
        assert len(records_values) == 20

        # nb_records still reflects original input size (2)
        assert result['nb_records'].int64_list.value[0] == 2

    def test_prepare_tf_features_missing_column_error(
        self,
        mock_tfrecord_files_manager,
        mock_logger,
        caplog,
        tmp_path
    ):
        """
        Verify that a KeyError is raised and logged if 'condition_level' is missing.
        """
        mock_tfrecord_files_manager._logger = mock_logger

        # Missing 'condition_level' entirely
        invalid_labels_df = pd.DataFrame({'severity': [0.0], 'x': [0.0], 'y': [0.0]})

        with pytest.raises(KeyError):
            mock_tfrecord_files_manager._prepare_tf_features(
                image_bytes=b"",
                study_id=0,
                series_id=0,
                series_min=0,
                series_max=0,
                instance_id=0,
                img_height=0,
                img_width=0,
                description=0,
                labels_df=invalid_labels_df,
                nb_max_records=1,
                is_padding=False
            )

        # Check that error was logged with traceback (exc_info=True)
        error_records = [rec for rec in caplog.records if rec.levelname == "ERROR"]
        assert len(error_records) == 1
        assert "condition_level" in error_records[0].message
        assert error_records[0].exc_info is not None

    def test_prepare_tf_features_general_exception(self, mock_tfrecord_files_manager, caplog):
        """
        Verify that any unexpected exception during feature preparation is caught,
        logged with traceback, and re-raised.
        """
        # Using a malformed DataFrame that will trigger an error
        # during the set_index or reindex operation (e.g., missing column)
        bad_labels_df = pd.DataFrame({'wrong_column': [1.0, 2.0]})

        with ExitStack() as stack:
            # Ensure we capture logs via caplog
            stack.enter_context(patch('src.core.utils.logger.get_current_logger',
                                      return_value=mock_tfrecord_files_manager._logger))

            # We expect a KeyError because 'condition_level' is missing
            with pytest.raises(KeyError):
                mock_tfrecord_files_manager._prepare_tf_features(
                    image_bytes=b"bytes",
                    study_id=1,
                    series_id=1,
                    series_min=0,
                    series_max=1,
                    instance_id=1,
                    img_height=1,
                    img_width=1,
                    description=1,
                    labels_df=bad_labels_df,
                    nb_max_records=5,
                    is_padding=False
                )

        # Verify the error was logged with the function name and traceback
        error_record = next((rec for rec in caplog.records if rec.levelname == "ERROR"), None)
        assert error_record is not None
        assert "Function TFRecordFilesManager._prepare_tf_features failed" in error_record.message

        # This confirms that logger.error(..., exc_info=True) was called
        assert error_record.exc_info is not None
