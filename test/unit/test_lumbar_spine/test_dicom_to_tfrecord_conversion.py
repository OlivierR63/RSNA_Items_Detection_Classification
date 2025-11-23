# coding: utf-8

import pytest
from unittest.mock import patch, MagicMock
import tensorflow as tf
import pandas as pd
from pathlib import Path
from typing import Tuple, Any
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset


@pytest.fixture
def mock_setup(mock_config, mock_logger):
    """
        Fixture to initialize common attributes for all tests.
    """

    dicom_study_dir = Path(mock_config["root_dir"]) / mock_config["dicom_study_dir"]
    tfrecord_dir = Path(mock_config["root_dir"]) / mock_config["tfrecord_dir"]

    metadata_df = pd.DataFrame(
                                    {
                                        "study_id": ['1', '1', '2', '4003253', '123456789'],
                                        "series_id": ['1', '2', '1', '1', '1234567'],
                                        "instance_number": ['1', '1', '1', '1', '1'],
                                        "file_path": ["path1", "path2", "path3", "path4", 'path5'],
                                        "metadata": ["meta1", "meta2", "meta3", "meta4", 'meta5']
                                     }
    )

    # Mock the get_current_logger function to return the mock_logger
    with patch("src.core.utils.logger.get_current_logger", return_value=mock_logger):
        yield mock_config, mock_logger, dicom_study_dir, tfrecord_dir, metadata_df


# Helper function to create a dummy TFRecord file
def generate_dummy_tfrecord(tfrecord_dir: Path) -> None:

    # Create an empty file
    dummy_file = tfrecord_dir / "4003253.TFRecord"
    dummy_file.touch()

    # Write some dummy data to the file
    with open(dummy_file, 'wb') as f:
        f.write(b'dummy_data')


class TestDicomToTFRecordConversion:
    """
        Unit tests for DICOM to TFRecord conversion in Lumbar Spine project.
    """

    def test_convert_dicom_to_tfrecords(
        self,
        mock_setup: Tuple[dict[str, Any], MagicMock, Path, Path, pd.DataFrame],
        tmp_path: Path
    ) -> None:
        """
            Test the main function for converting DICOM to TFRecord.
        """

        mock_config, mock_logger, dicom_study_dir, tfrecord_dir, metadata_df = mock_setup

        mock_csv_path = "src.projects.lumbar_spine.csv_metadata.CSVMetadata"

        with (
                  patch(mock_csv_path) as mock_csv_metadata_class,

                  patch.object(
                      LumbarDicomTFRecordDataset,
                      '_encode_dataframe'
                  ) as mock_encode_dataframe,

                  patch.object(
                      LumbarDicomTFRecordDataset,
                      '_generate_tfrecord_files'
                  ) as mock_generate_tfrecord_files
              ):

            # Mock _generate_tfrecord_files to create a dummy file instead of actual processing
            def generate_dummy_tfrecord_wrapper(*args, **kwargs):
                generate_dummy_tfrecord(tfrecord_dir)

            mock_generate_tfrecord_files.side_effect = generate_dummy_tfrecord_wrapper

            # Configure the mock CSVMetadata class
            mock_csv_metadata_instance = MagicMock()
            mock_csv_metadata_class.return_value = mock_csv_metadata_instance
            mock_csv_metadata_instance._merged_df = metadata_df

            # Mock _encode_dataframe to return the same DataFrame without processing
            mock_encode_dataframe.return_value = metadata_df

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

            # Now patch methods on the instance
            with (
                    patch.object(dataset,
                                 '_setup_tfrecord_directory') as mock_setup_tfrecord_directory,
                    patch.object(dataset, '_process_study') as mock_process_study
                  ):

                mock_setup_tfrecord_directory.return_value = tfrecord_dir
                mock_process_study.return_value = None

                # Call the function
                str_study_dir = str(dicom_study_dir)
                dataset._convert_dicom_to_tfrecords(str_study_dir, metadata_df, str(tfrecord_dir))

                # Verifications
                mock_setup_tfrecord_directory.assert_called_once_with(str(tfrecord_dir))
                mock_process_study.assert_called()

                mock_logger.info.assert_any_call(
                    "Starting DICOM to TFRecord conversion",
                    extra={"action": "convert_dicom", "dicom_study_dir": str_study_dir}
                )

                mock_logger.info.assert_called_with(
                    "DICOM to TFRecord conversion completed successfully",
                    extra={"status": "success"}
                )

                # Check that the expected TFRecord file has been created
                expected_tfrecord_file = tfrecord_dir / "4003253.TFRecord"
                assert (
                    expected_tfrecord_file.exists()
                ), f"Expected TFRecord file {expected_tfrecord_file} was not created"

                # Check that the TFRecord file is not empty
                assert (
                    expected_tfrecord_file.stat().st_size > 0
                ), f"TFRecord file {expected_tfrecord_file} is empty"

    def test_setup_tfrecord_directory(
        self,
        mock_setup: Tuple[dict[str, Any], MagicMock, Path, Path, pd.DataFrame],
        tmp_path: Path
    ) -> None:

        """
            Test the creation of the tfrecord directory.
        """

        mock_config, mock_logger, _, tfrecord_dir, metadata_df = mock_setup

        mock_csv_path = "src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset.CSVMetadata"

        with (
                    patch(mock_csv_path) as mock_csv_metadata_class,

                    patch.object(
                        LumbarDicomTFRecordDataset,
                        '_generate_tfrecord_files'
                    ) as mock_generate_tfrecord_files
                ):

            # Mock _generate_tfrecord_files to avoid side effects
            mock_generate_tfrecord_files.return_value = None

            # Configure the mock CSVMetadata class
            mock_csv_metadata_instance = MagicMock()
            mock_csv_metadata_class.return_value = mock_csv_metadata_instance
            mock_csv_metadata_instance._merged_df = metadata_df

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

            # Call the function
            test_dir = dataset._setup_tfrecord_directory(str(tmp_path / "tfrecords"))

            # Verifications
            assert test_dir == tfrecord_dir
            assert test_dir.is_dir()

    def test_process_study(
        self,
        mock_setup: Tuple[dict[str, Any], MagicMock, Path, Path, pd.DataFrame],
        tmp_path: Path
    ) -> None:

        """
            Test the processing of a study.
        """

        mock_config, mock_logger, _, tfrecord_dir, metadata_df = mock_setup

        study_path = tmp_path / "123456789"
        study_path.mkdir()
        series_path = study_path / "1234567"
        series_path.mkdir()
        (series_path / "1.dcm").write_text("dummy")

        mock_csv_path = "src.projects.lumbar_spine.csv_metadata.CSVMetadata"
        with (
                patch(mock_csv_path) as mock_csv_metadata_class,

                patch.object(
                    LumbarDicomTFRecordDataset,
                    '_generate_tfrecord_files'
                ) as mock_generate_tfrecord_files
              ):

            # Mock _generate_tfrecord_files to avoid side effects
            mock_generate_tfrecord_files.return_value = None

            # Configure the mock CSVMetadata class
            mock_csv_metadata_instance = MagicMock()
            mock_csv_metadata_class.return_value = mock_csv_metadata_instance
            mock_csv_metadata_instance._merged_df = metadata_df

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

            with patch.object(dataset, '_process_series') as mock_process_series:
                mock_process_series.return_value = None

                tfrecord_dir = dataset._setup_tfrecord_directory(tmp_path / "tfrecords")

                # 5. Call the function under test
                # Remark: the logger SHALL be passed as a keyword argument (logger=mock_logger)
                # to avoid the "got multiple values for argument 'logger'" TypeError,
                # which occurs when the 'log_method' decorator tries to inject the logger
                # while it is simultaneously passed positionally.
                dataset._process_study(
                                       study_path=study_path,
                                       metadata_df=metadata_df,
                                       tfrecord_dir=tfrecord_dir,
                                       logger=mock_logger
                                    )

            # Verify that the TFRecord file has been actually created
            tfrecord_file = tfrecord_dir / "123456789.tfrecord"
            assert tfrecord_file.is_file()

            # Verifications
            mock_process_series.assert_called_once()

    def test_process_series(
        self,
        mock_setup: Tuple[dict[str, Any], MagicMock, Path, Path, pd.DataFrame],
        tmp_path: Path
    ) -> None:

        """
            Test the processing of a serie.
        """

        mock_config, mock_logger, _, _, metadata_df = mock_setup

        series_path = tmp_path / "123456789"
        series_path.mkdir()
        (series_path / "1.dcm").write_text("dummy")

        mock_csv_path = "src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset.CSVMetadata"
        with (
                patch(mock_csv_path) as mock_csv_metadata_class,

                patch.object(
                    LumbarDicomTFRecordDataset,
                    '_generate_tfrecord_files'
                ) as mock_generate_tfrecord_files
              ):

            # Mock _generate_tfrecord_files to avoid side effects
            mock_generate_tfrecord_files.return_value = None

            # Configure the mock CSVMetadata class
            mock_csv_metadata_instance = MagicMock()
            mock_csv_metadata_class.return_value = mock_csv_metadata_instance
            mock_csv_metadata_instance._merged_df = metadata_df

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

            with patch.object(dataset, '_process_dicom_file') as mock_process_dicom_file:
                mock_process_dicom_file.return_value = (b"img_bytes", b"metadata_bytes")
                mock_writer = MagicMock(spec=tf.io.TFRecordWriter)

                # Call the function under test
                # Remark: the logger SHALL be passed as a keyword argument (logger=mock_logger)
                # to avoid the "got multiple values for argument 'logger'" TypeError,
                # which occurs when the 'log_method' decorator tries to inject the logger
                # while it is simultaneously passed positionally.
                dataset._process_series(
                                            series_path=series_path,
                                            metadata_df=metadata_df,
                                            writer=mock_writer,
                                            logger=mock_logger
                                        )

                # Verifications
                mock_process_dicom_file.assert_called_once()
                mock_writer.write.assert_called_once()

    def test_process_series_missing_dicom_metadata(
        self,
        mock_setup: Tuple[dict[str, Any], MagicMock, Path, Path, pd.DataFrame],
        tmp_path: Path
    ) -> None:

        """
            Tests the 'if dicom_metadata_df.empty: continue' branch in _process_series.

            Simulates a DICOM file for which no instance-specific metadata is available,
            and asserts that:
            1. The file processing is skipped (_process_dicom_file is NOT called).
            2. The four expected warnings are logged sequentially.
        """
        mock_config, mock_logger, _, _, _ = mock_setup

        # Initialize the dataset object
        # Note: We must patch _generate_tfrecord_files to prevent side effects during init
        with patch.object(
                            LumbarDicomTFRecordDataset,
                            '_generate_tfrecord_files',
                            return_value=None
                          ):
            dataset_object = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # 1. Setup Mock Paths and IDs
        series_id = "SERIES_12345"
        dicom_file_stem = "99"  # Instance number (file stem) that will not be found
        dicom_file_name = f"{dicom_file_stem}.dcm"
        series_path_str = str(tmp_path / series_id)

        # 2. Setup Input Data (metadata_df)
        # DataFrame contains metadata for a DIFFERENT instance number (e.g., '100'). This ensures
        # that the filter 'metadata df[... == dicom path.stem]' will return an empty DataFrame.
        mock_metadata_df = pd.DataFrame({
                                          'study_id': ["STUDY_999"],
                                          'series_id': [series_id],
                                          'instance_number': ["100"],  # Does not correspond to '99'
                                          'metadata': ['meta1']
                                        })

        # 3. Mock Series Path and its contents

        # Mock: The path to the series directory
        mock_series_path = MagicMock(spec=Path)
        mock_series_path.name = series_id

        # Mock: The path to the DICOM file whose metadata is missing
        mock_dicom_path = MagicMock(spec=Path)
        mock_dicom_path.name = dicom_file_name
        mock_dicom_path.stem = dicom_file_stem  # La valeur utilisee pour le filtrage
        mock_dicom_path.__str__.return_value = f"{series_path_str}/{dicom_file_name}"

        # Mock the glob("*.dcm") call to return only the missing file
        mock_series_path.glob.return_value = [mock_dicom_path]

        # 4. Patch Dependencies
        with (
                # We patch the functions that SHOULD NOT be called.
                patch.object(
                                dataset_object,
                                '_process_dicom_file'
                             ) as mock_process_dicom_file,
                patch.object(
                                dataset_object,
                                '_write_tfrecord_example'
                             ) as mock_write_tfrecord_example,
               ):

            # Reset calls to the logger
            mock_logger.warning.reset_mock()

            # Mock the TFRecordWriter
            mock_writer = MagicMock(spec=tf.io.TFRecordWriter)

            # 5. Call the function under test
            # Remark: the logger SHALL be passed as a keyword argument (logger=mock_logger)
            # to avoid the "got multiple values for argument 'logger'" TypeError,
            # which occurs when the 'log_method' decorator tries to inject the logger
            # while it is simultaneously passed positionally.
            dataset_object._process_series(
                                            series_path=mock_series_path,
                                            metadata_df=mock_metadata_df,
                                            writer=mock_writer,
                                            logger=mock_logger
                                           )

            # 6. Assertions

            # 6.1. Assert that the main processing functions have NOT been called ('continue' logic)
            mock_process_dicom_file.assert_not_called()
            mock_write_tfrecord_example.assert_not_called()

            # 6.2. Assert that the warning message was logged EXACTLY FOUR times
            assert mock_logger.warning.call_count == 4

            # 6.3. Assert the content of the 4 calls to the logger

            # Expected messages for the 4 calls
            warning_msg = f"No metadata found for DICOM file {dicom_file_name} "
            expected_warning_1_start = warning_msg

            warning_msg = "This file will not be considered during training or evaluation."
            expected_warning_2_exact = warning_msg

            warning_msg = "This may be due to missing or inconsistent records in the CSV files."
            expected_warning_3_exact = warning_msg

            warning_msg = "Please check the CSV files and ensure they contain the relevant records."
            expected_warning_4_exact = warning_msg

            # Extract arguments from all warning calls
            warning_calls = mock_logger.warning.call_args_list

            # Verify the first warning (the most detailed)
            first_warning_msg = warning_calls[0][0][0]
            assert first_warning_msg.startswith(expected_warning_1_start)
            assert f"in series {series_id}. Skipping this file." in first_warning_msg

            # Verify the next messages
            assert warning_calls[1][0][0] == expected_warning_2_exact
            assert warning_calls[2][0][0] == expected_warning_3_exact

            # Verify the mast message message (User instruction)
            assert warning_calls[3][0][0] == expected_warning_4_exact

    def test_process_dicom_file(
        self,
        mock_setup: Tuple[dict[str, Any], MagicMock, Path, Path, pd.DataFrame],
        tmp_path: Path
    ) -> None:

        mock_config, mock_logger, _, _, metadata_df = mock_setup

        """Test the processing of a DICOM file."""
        dicom_path = tmp_path / "file1.dcm"
        dicom_path.write_text("dummy")

        with (
                    patch("SimpleITK.ReadImage") as mock_read_image,

                    patch.object(
                        LumbarDicomTFRecordDataset,
                        '_serialize_metadata',
                    ) as mock_serialize_metadata,

                    patch.object(
                        LumbarDicomTFRecordDataset,
                        '_generate_tfrecord_files',
                    ) as mock_generate_tfrecord_files,
               ):

            # Mock _generate_tfrecord_files to avoid side effects
            mock_generate_tfrecord_files.return_value = None

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

            # Configure the mock for SimpleITK.ReadImage
            mock_img = MagicMock()
            mock_img.GetPixelIDValue.return_value = 2  # sitkUInt16
            mock_read_image.return_value = mock_img

            # Mock sitk.GetArrayFromImage to return a numpy array
            with patch("SimpleITK.GetArrayFromImage") as mock_get_array_from_image:
                mock_img_array = [[1, 2], [3, 4]]
                mock_get_array_from_image.return_value = mock_img_array

                mock_serialize_metadata.return_value = b"metadata_bytes"

                # Call the function
                img_bytes, metadata_bytes = dataset._process_dicom_file(
                                                                         dicom_path,
                                                                         metadata_df
                                                                        )

                # Verifications
                mock_read_image.assert_called_once_with(str(dicom_path))
                mock_get_array_from_image.assert_called_once_with(mock_img)
                mock_serialize_metadata.assert_called_once_with(metadata_df)
                assert isinstance(img_bytes, bytes)
                assert metadata_bytes == b"metadata_bytes"

    def test_write_tfrecord_example(
        self,
        mock_setup: Tuple[dict[str, Any], MagicMock, Path, Path, pd.DataFrame],
        tmp_path: Path
    ) -> None:

        """
            Test the writing of a TFRecord example.
        """

        mock_config, mock_logger, _, _, _ = mock_setup

        mock_writer = MagicMock(spec=tf.io.TFRecordWriter)
        img_bytes = b"img_bytes"
        metadata_bytes = b"metadata_bytes"

        # Initialize the dataset with the mock logger
        dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # Call the function
        dataset._write_tfrecord_example(img_bytes, metadata_bytes, mock_writer)

        # Verifications
        mock_writer.write.assert_called_once()
        args, _ = mock_writer.write.call_args
        assert isinstance(args[0], bytes)
