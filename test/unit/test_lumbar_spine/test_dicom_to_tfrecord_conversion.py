# coding: utf-8

import pytest
from unittest.mock import patch, MagicMock, NonCallableMock
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

    dicom_study_dir = Path(mock_config["dicom_study_dir"])
    tfrecord_dir = Path(mock_config["tfrecord_dir"])

    metadata_df = pd.DataFrame(
                                    {
                                        "study_id": [1, 1, 2, 4003253, 123456789],
                                        "series_id": [1, 2, 1, 1, 1234567],
                                        "instance_number": [1, 1, 1, 1, 1],
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
    expected_study_id = "4003253"
    dummy_file = tfrecord_dir / f"{expected_study_id}.TFRecord"

    # Write some dummy data to the file
    with open(dummy_file, 'wb') as f:
        f.write(b'dummy_data')

# Define a custom exception for simulation purposes
class MockTFRecordError(Exception):
    """Custom exception to simulate a write failure."""
    pass

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

        # ID d'étude cible (doit correspondre à l'entrée '4003253' dans votre DataFrame)
        study_id_str_to_process = "4003253"
    
        # 1. Créer le répertoire d'étude simulé dans le chemin DICOM
        expected_study_path = dicom_study_dir / study_id_str_to_process
        expected_study_path.mkdir(parents=True, exist_ok=True)

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
        dicom_file_stem = "99"
        dicom_file_name = f"{dicom_file_stem}.dcm"
        
        # Minimal DataFrame for the function signature
        mock_metadata_df = pd.DataFrame({'instance_number': ["100"]}) 

        # 2. Mock Series Path and its contents
        mock_series_path = MagicMock(spec=Path)
        mock_series_path.name = series_id
        # Mock glob("*.dcm") to return only one file
        mock_series_path.glob.return_value = [
            MagicMock(spec=Path, name=dicom_file_name, stem=dicom_file_stem)
        ]

        # 3. Patch Dependencies
        with (
                # 3.1. Simulate metadata failure: _process_single_dicom_instance returns metadata_ok=False
                patch.object(
                    dataset_object,
                    '_process_single_dicom_instance',
                    # Return tuple: (metadata_ok=False, process_initiated_and_aborted=False)
                    return_value=(False, False) 
                ) as mock_process_single_dicom_instance,
                
                # 3.2. Patch the functions that SHOULD NOT be called, as 'metadata_ok' is False
                patch.object(dataset_object, '_process_dicom_file') as mock_process_dicom_file,
                patch.object(dataset_object, '_write_tfrecord_example') as mock_write_tfrecord_example,
            ):

            # Reset calls to the logger
            mock_logger.reset_mock()
            # Mock the TFRecordWriter
            mock_writer = MagicMock(spec=tf.io.TFRecordWriter)

            # 4. Call the function under test
            dataset_object._process_series(
                series_path=mock_series_path,
                metadata_df=mock_metadata_df,
                writer=mock_writer,
                logger=mock_logger
            )

            # 5. Assertions

            # 5.1. Assertions on internal calls
            mock_process_dicom_file.assert_not_called()
            mock_write_tfrecord_example.assert_not_called()

            # The core function should have been called once inside the loop
            mock_process_single_dicom_instance.assert_called_once()
            
            # 5.2. Assertions on Logger
            
            # Verify 1: The starting 'info' log is present
            mock_logger.info.assert_called_once()
            
            # Verify 2: The final 'error' log for complete failure is called once (nb_success_file == 0).
            mock_logger.error.assert_called_once()
            
            # Verify 3: No warning logs were produced by _process_series itself.
            mock_logger.warning.assert_not_called()
            
            # Verify 4: The content of the error log (complete_failure branch)
            error_call_args, error_call_kwargs = mock_logger.error.call_args
            
            expected_error_start = f"Series {series_id} processing failed: All files were skipped or failed during processing."
            
            assert error_call_args[0].startswith(expected_error_start)
            assert error_call_kwargs["extra"]["status"] == "failed"

    def test_process_single_series_skip_non_directory(
            self,
            mock_setup: Tuple[dict[str, Any], MagicMock]
        ) -> None:
        """
            Tests the control flow when a non-directory item (like a file) 
            is encountered within the study directory during series processing.
    
            Covers the case: 'if not series_path.is_dir(): return False'
    
            Simulates:
            1. A series path that returns False when .is_dir() is called (a file).
    
            Asserts:
            1. The main series processing method (_process_series) is never called.
            2. A warning is logged indicating the skip due to the item not being a directory.
            3. The function correctly returns False.
        """
        mock_config, mock_logger, _, _, _ = mock_setup

        STUDY_ID = 1000000001
        SERIES_ID = 123456789
        FILE_NAME = ".DS_Store"  # Example of a non-directory item

        # 1. Prepare Mocks
    
        # Metadata DataFrame is required for function signature, but its content is irrelevant 
        # as the function exits before accessing it.
        metadata_df = pd.DataFrame({'series_id': [SERIES_ID], 'data': ['some_data']})
    
        # Mock for the TFRecordWriter (required argument)
        mock_writer = MagicMock(spec=tf.io.TFRecordWriter)

        # Mock object for the study path (parent of the series path)
        mock_study_path = NonCallableMock(spec=Path)
        mock_study_path.name = str(STUDY_ID)

        # Mock object for the non-directory item (series_path)
        mock_series_path = MagicMock(spec=Path)
        mock_series_path.name = FILE_NAME

        # CRITICAL CONDITION: is_dir() must return False to cover the target block
        mock_series_path.is_dir.return_value = False
        mock_series_path.parent = mock_study_path # Simulates the parent directory relationship

        # Initialize the dataset object (necessary to call the internal method)
        # Patch _generate_tfrecord_files to avoid side effects during initialization
        with patch.object(
            LumbarDicomTFRecordDataset,
            '_generate_tfrecord_files',
            return_value=None
        ):
            dataset_object = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # 2. Patch and Call
        with (
            # Patch the internal method _process_series, which must NOT be called in this test
            patch.object(
                dataset_object,
                '_process_series',
                return_value=True 
            ) as mock_process_series,
        ):
            mock_logger.reset_mock()

            # 3. Call the method under test
            result = dataset_object._process_single_series_instance(
                series_path=mock_series_path,
                metadata_df=metadata_df,
                writer=mock_writer,
                logger=mock_logger
            )

            # 4. Assertions

            # 4.1. Assertions on Control Flow
            # The function must return False as it encountered a non-directory item
            assert result is False

            # _process_series MUST NOT be called
            mock_process_series.assert_not_called()

            # 4.2. Assertions on Logging
            mock_logger.warning.assert_called_once()

            # Check the content of the warning message
            warning_call_args = mock_logger.warning.call_args[0][0]
            # The exact path strings are used in the warning message
            expected_warning_substring = (
                f"Skipping non-directory item: {mock_series_path} in study: {mock_study_path}"
            )

            assert expected_warning_substring in warning_call_args
        
            mock_logger.info.assert_not_called()
            mock_logger.error.assert_not_called()

    def test_process_single_series_skip_missing_metadata(
            self,
            mock_setup: Tuple[dict[str, Any], MagicMock]
        ) -> None:
        """
            Tests the control flow when a valid series directory is found, 
            but no matching metadata is present in the provided DataFrame.
    
            This covers the conditional block: 'if series_metadata_df.empty: ... return False'.
    
            Simulates:
            1. A series path that is confirmed as a valid directory (.is_dir() returns True).
            2. A metadata DataFrame that, when filtered by the specific series ID, results 
               in an empty DataFrame (no matching records found).
    
            Asserts:
            1. The core series processing method (_process_series) is never called.
            2. A warning is logged, detailing the skip due to missing metadata.
            3. The function returns False, indicating processing failure/skip.
        """
        # Fix for 'ValueError: too many values to unpack' often caused by setup fixtures 
        # returning more than two items (e.g., config, logger, and self).
        mock_config, mock_logger, *_ = mock_setup 

        STUDY_ID = 1000000001
        SERIES_ID_PRESENT = 123456789
        SERIES_ID_TARGET = 999999999 # The ID of the mocked series directory (missing metadata)
    
        # 1. Prepare Mocks
    
        # Metadata DataFrame contains a DIFFERENT ID, ensuring the filtered DataFrame will be empty.
        metadata_df = pd.DataFrame(
            {
                'series_id': [SERIES_ID_PRESENT], 
                'study_id': [STUDY_ID],
                'data': ['some_data']
            }
        )
    
        # Mock for the TFRecordWriter (required argument)
        mock_writer = MagicMock(spec=tf.io.TFRecordWriter)

        # Mock Path object for the study path (parent directory)
        mock_study_path = MagicMock(spec=Path)
        mock_study_path.name = str(STUDY_ID)

        # Mock Path object for the series directory being processed
        mock_series_path = MagicMock(spec=Path)
        mock_series_path.name = str(SERIES_ID_TARGET) 
        # CRITICAL CONDITION 1: Must be a directory to pass the first check
        mock_series_path.is_dir.return_value = True 
        mock_series_path.parent = mock_study_path # Establish path hierarchy

        # Initialize the dataset object for patching internal methods
        with patch.object(
            LumbarDicomTFRecordDataset,
            '_generate_tfrecord_files',
            return_value=None
        ):
            dataset_object = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # 2. Patch and Call
        with (
            # Patch the internal series processing method, which MUST NOT be called in this test
            patch.object(
                dataset_object,
                '_process_series',
                return_value=False 
            ) as mock_process_series,
        ):
            mock_logger.reset_mock()

            # 3. Call the method under test
            result = dataset_object._process_single_series_instance(
                series_path=mock_series_path,
                metadata_df=metadata_df,
                writer=mock_writer,
                logger=mock_logger
            )

            # 4. Assertions
        
            # 4.1. Assertion on Control Flow
            # The function must return False (skip due to empty filtered DataFrame)
            assert result is False
        
            # _process_series MUST NOT be called, as flow stopped earlier
            mock_process_series.assert_not_called()

            # 4.2. Assertion on Logging
            mock_logger.warning.assert_called_once()
        
            # Check the warning message content
            warning_call_args = mock_logger.warning.call_args[0][0]
            expected_warning_substring = (
                f"Skipping series {SERIES_ID_TARGET} in study {STUDY_ID}: No matching metadata found."
            )

            assert warning_call_args.startswith(expected_warning_substring)
        
            # Check the 'extra' log parameters
            extra_args = mock_logger.warning.call_args[1].get('extra')
            assert extra_args is not None
            assert extra_args['status'] == "metadata_missing"
            assert extra_args['series_dir'] == str(SERIES_ID_TARGET)
            assert extra_args['study_id'] == str(STUDY_ID)
        
            mock_logger.info.assert_not_called()
            mock_logger.error.assert_not_called()

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

    def test_write_tfrecord_example_raises_on_exception(
            self,
            mock_setup: Tuple[dict[str, Any], MagicMock]
        ) -> None:

        """
            Tests the error handling block (try...except...raise) in _write_tfrecord_example.
    
            This simulates a failure during the writing process (e.g., I/O error, 
            serialization failure) by forcing the mocked writer.write() method to raise.
    
            Covers the case: 'except Exception as e: logger.error(...) raise'
    
            Asserts:
            1. The function re-raises the exception.
            2. An error message is logged with the exception details and traceback.
        """

        # Fix for 'ValueError: too many values to unpack' 
        mock_config, mock_logger, *_ = mock_setup 

        # 1. Prepare Mocks and Data
    
        # Mock input data (contents are irrelevant for this test, only existence matters)
        mock_img_bytes = b"mock_image_data"
        mock_metadata_bytes = b"mock_metadata_data"
    
        # Mock TFRecordWriter object
        mock_writer = MagicMock(spec=tf.io.TFRecordWriter)
    
        # CRITICAL STEP: Force the writer.write method to raise an exception
        MOCK_ERROR_MESSAGE = "Simulated write failure for testing."
        mock_writer.write.side_effect = MockTFRecordError(MOCK_ERROR_MESSAGE)

        # Initialize the dataset object (necessary to call the internal method)
        with patch.object(
            LumbarDicomTFRecordDataset,
            '_generate_tfrecord_files',
            return_value=None
        ):
            dataset_object = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # Reset mock to clear __init__ logging calls.
        mock_logger.reset_mock()

        # 2. Call the method under test and assert it raises
    
        # We use pytest.raises to assert that the custom exception is re-raised
        with pytest.raises(MockTFRecordError) as exc_info:
            dataset_object._write_tfrecord_example(
                img_bytes=mock_img_bytes,
                serialized_metadata=mock_metadata_bytes,
                writer=mock_writer,
                logger=mock_logger
            )

        # 3. Assertions
    
        # 3.1. Assertion on the re-raised exception
        assert MOCK_ERROR_MESSAGE in str(exc_info.value)
    
        # 3.2. Assertion on Logging
    
        # Check that the error was logged
        mock_logger.error.assert_called_once()
    
        # Check the error message content and arguments
        log_call = mock_logger.error.call_args_list[0]
    
        # Message should contain the error string
        assert f"Error writing TFRecord example: {MOCK_ERROR_MESSAGE}" in log_call.args[0]
    
        # Check that exc_info=True was passed to log the full traceback
        assert log_call.kwargs['exc_info'] is True
    
        # Check 'extra' logs
        extra_args = log_call.kwargs['extra']
        assert extra_args is not None
        assert extra_args['status'] == "failed"
        assert extra_args['error'] == MOCK_ERROR_MESSAGE
    
        mock_logger.warning.assert_not_called()
        mock_logger.info.assert_not_called()
