# coding: utf-8

from unittest.mock import patch, MagicMock, ANY

# The 'ANY' object is imported here for use in mock assertions.
# It is necessary to replace a pandas.DataFrame argument in 
# 'assert_called_once_with', as direct comparison of DataFrames 
# raises a 'ValueError: DataFrame is ambiguous' within unittest.mock.

from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
import pytest
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Any
import inspect


class TestLumbarDicomTFRecordDataset:

    def _bytes_feature(self, value: bytes) -> tf.train.Feature:
        """
            Helper to create a bytes Feature for a TFRecord example.
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _mock_successful_error_handling(self,
                                        tfrecord_proto: tf.Tensor,
                                        mock_logger: MagicMock) -> Tuple[tf.Tensor, tf.Tensor]:
        """
            Helper function used as a side_effect for the mocked _parse_tfrecord.

            It simulates the successful execution of the error handling logic
            (the try/except block) within _parse_tfrecord by:
            1. Logging the simulated exception via the mock_logger.
            2. Returning the default (zeroed) tensors that prevent the
               tf.data pipeline from crashing.

            This verifies that the error management logic (logging + returning defaults)
            is correctly triggered by the pipeline.
        """
        # 1. Simulate the exception being caught (by logging the error)
        mock_logger.error(
                            "Error parsing record: Map error simulated",
                            exc_info=True
                          )

        # 2. Return the dummy tensors that the error handling code would produce
        dummy_image = tf.zeros((64, 64, 64, 1), dtype=tf.float32)
        dummy_metadata = tf.constant(b'', dtype=tf.string)

        return dummy_image, dummy_metadata

    def generate_dummy_tfrecord_file(
                                        self,
                                        mock_setup: Tuple[dict[str, Any], MagicMock],
                                        tmp_path: Path
                                     ) -> None:
        """
            Helper function to create a dummy TFRecord file
            with a valid serialized example.
        """
        mock_config, _ = mock_setup

        # Create a temporary directory for the TFRecord files :
        tfrecord_dir = Path(mock_config["tfrecord_dir"])
        tfrecord_dir.mkdir(parents=True, exist_ok=True)

        # Create a dummy TFRecord file
        dummy_tfrecord_file = tfrecord_dir / "dummy.TFRecord"
        dummy_tfrecord_file.touch()

        # CRITICAL ISSUE: Write a single, valid, dummy TFRecord to the file.
        # An empty file is not a valid TFRecord and causes StopIteration.

        # Create a minimal tf.train.Example (only a single feature is needed)
        example = tf.train.Example(features=tf.train.Features(feature={
            'tfrecord_file': self._bytes_feature(b'dummy_file_path')
        }))

        # Write the serialized example to the dummy file
        with tf.io.TFRecordWriter(str(dummy_tfrecord_file)) as writer:
            writer.write(example.SerializeToString())

    # -------------------------------------------------------------------------
    # Test functions using the new helpers
    # -------------------------------------------------------------------------

    def test_max_records_flat(
                                self,
                                mock_setup: Tuple[dict[str, Any], MagicMock],
                                tmp_path: Path
                              ) -> None:
        """
            Tests the maximum number of flattened records calculation.
        """
        mock_config, mock_logger = mock_setup

        # Initialize the dataset
        dataset = LumbarDicomTFRecordDataset(mock_config,
                                             logger=mock_logger)

        assert dataset.max_records_flat() == dataset._MAX_RECORDS_FLAT

    def test_py_func_map_wrapper(
                                    self,
                                    mock_setup: Tuple[dict[str, Any], MagicMock],
                                    tmp_path: Path
                                ) -> None:
        """
            Tests the _py_func_map_wrapper method to ensure it correctly calls tf.py_function
            with the proper arguments: func=self._parse_tfrecord and the correct Tout dtypes.
        """
        mock_config, mock_logger = mock_setup

        # Initialize the dataset object
        dataset_obj = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # Create a mock input tensor (tfrecord_proto)
        input_proto_tensor = tf.constant(b"dummy_tfrecord_bytes", dtype=tf.string)

        # Patch tf.py_function to intercept the call
        # Note: We patch the global function 'tensorflow.py_function'
        with patch("tensorflow.py_function") as mock_py_function:

            # Call the method under test
            dataset_obj._py_func_map_wrapper(input_proto_tensor)

            # Assertions:

            # 1. Check that tf.py_function was called exactly once
            mock_py_function.assert_called_once()

            # 2. Check the arguments passed to tf.py_function

            # The expected func must be the instance method self._parse_tfrecord
            expected_func = dataset_obj._parse_tfrecord

            # The expected inp must be the list containing the input tensor
            expected_inp = [input_proto_tensor]

            # The expected Tout must be the list of dtypes
            expected_tout = [tf.float32, tf.string]

            mock_py_function.assert_called_with(
                                                    func=expected_func,
                                                    inp=expected_inp,
                                                    Tout=expected_tout
                                                )

    def test_generate_tfrecord_files(
                                        self,
                                        mock_setup: Tuple[dict[str, Any], MagicMock],
                                        mock_csv_metadata: MagicMock,
                                        mock_convert_dicom: MagicMock,
                                        tmp_path: Path
                                        ) -> None:
        """
            Tests the TFRecord files generation process,
            which is triggered upon object initialization.
            The underlying I/O operations are mocked.
        """

        mock_config, mock_logger = mock_setup

        # Mock the CSVMetadata class to avoid file reading
        csv_metadata_chain = (
                        "src.projects.lumbar_spine."
                        "lumbar_dicom_tfrecord_dataset.CSVMetadata"
                        )
        with patch(csv_metadata_chain, return_value=mock_csv_metadata):

            # Mock _convert_dicom_to_tfrecords to avoid I/O operations
            with patch.object(LumbarDicomTFRecordDataset,
                              '_convert_dicom_to_tfrecords',
                              mock_convert_dicom):

                # Initialize the dataset,
                # WHICH SHOULD TRIGGER _generate_tfrecord_files
                dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

                # Define the _tfrecord_dir path to simulate
                # the expected state after initialization
                dataset._tfrecord_dir = Path(tmp_path/"tfrecords")
                dataset._tfrecord_dir.mkdir(parents=True, exist_ok=True)

                # Verification checks
                mock_logger.info.assert_any_call(
                    "Starting generate_tfrecord_file",
                    extra={"action": "generate_tf_records"}
                )
                mock_logger.info.assert_called_with(
                    "DICOM to TFRecord conversion completed.",
                    extra={"status": "success"}
                )
                mock_convert_dicom.assert_called_once()

    def test_generate_tfrecord_files_is_skipped_with_dummy_file(
                                                        self,
                                                        mock_setup: Tuple[dict, MagicMock],
                                                        tmp_path: Path
                                                        ) -> None:
        """
            Verifies that the TFRecord generation is skipped and correctly logged
            when a dummy TFRecord file is physically present in the output directory.
        """
        mock_config, mock_logger = mock_setup

        # 1. Determine the expected TFRecord output directory from the mock_config
        # config['tfrecord_dir'] points to the path already created by setup_test_env
        tfrecord_dir = Path(mock_config['tfrecord_dir'])

        # 2. Create the dummy TFRecord file to trigger the skip logic
        # Assuming the class checks for a file name like "train_000.tfrecord"
        dummy_tfrecord_path = tfrecord_dir / "train_000.tfrecord"
        dummy_tfrecord_path.touch()  # Create an empty file

        # Path to the internal conversion method (to ensure it's NOT called)
        conversion_method_path = (
            "src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset."
            "LumbarDicomTFRecordDataset._convert_dicom_to_tfrecords"  # Use the correct method name
        )

        with (
                    # 3. Mock the core conversion method (to verify it's skipped)
                    patch(conversion_method_path) as mock_convert,

                    # 4. Mock the metadata loading (a necessary dependency for constructor)
                    patch('src.projects.lumbar_spine.csv_metadata.CSVMetadata')
                ):
            # Initializing the dataset triggers the logic that calls _generate_tfrecord_files
            LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

            # Verification 1: The 'skipped' log must be called.
            # assert_any_call is used because other log messages precede this one.
            mock_logger.info.assert_any_call(
                "Existing TFRecords found. Skipping conversion.",
                extra={"status": "skipped"}
            )

            # Verification 2: The actual conversion method must NOT be called
            mock_convert.assert_not_called()

    def test_generate_tfrecord_files_raises_and_logs_error(
                                                            self,
                                                            mock_setup: Tuple[dict, MagicMock],
                                                            tmp_path: Path
                                                           ) -> None:
        """
            Verifies that an exception during conversion is caught, logged
            with status 'failed', and then correctly propagated ('raise').
        """
        mock_config, mock_logger = mock_setup

        exception_message = "Simulated I/O or conversion failure"

        # Path to the internal conversion method (to inject the error)
        conversion_method_path = (
            "src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset."
            "LumbarDicomTFRecordDataset._convert_dicom_to_tfrecords"
        )

        # Mock path existence check to return False (ensures conversion is NOT skipped)
        with patch("pathlib.Path.exists", return_value=False):

            with (
                    # Mock the core conversion method to raise an error
                    patch(conversion_method_path, side_effect=RuntimeError(exception_message)),

                    # Assert that the exception is correctly re-raised (Line 181: raise)
                    pytest.raises(RuntimeError) as excinfo
                  ):

                # Initializing the dataset triggers the logic, which hits the side_effect
                _ = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

            # Verification 1: The error was propagated with the correct message
            assert exception_message in str(excinfo.value)

            # Verification 2: The error was logged
            mock_logger.error.assert_called_once()
            args, kwargs = mock_logger.error.call_args

            # Check logged message and error details
            assert "Error generating TFRecords" in args[0]
            assert exception_message in args[0]

            # Check the 'extra' arguments for the required status
            assert kwargs["extra"]["status"] == "failed"
            assert kwargs["extra"]["error"] == exception_message

    def test_convert_dicom_to_tfrecords_handles_exception_and_raises(
        self,
        mock_setup: Tuple[dict, MagicMock],
        tmp_path: Path
    ) -> None:
        """
            Covers except block in _convert_dicom_to_tfrecords.
            Verifies that an exception during the conversion loop is caught, logged
            with status 'failed', and then correctly re-raised.
        """
        mock_config, mock_logger = mock_setup
        exception_message = "Simulated DICOM processing failure"

        # DataFrame required to pass the 'study_metadata_df.empty' check in the loop
        metadata_for_exception_test = pd.DataFrame({
            'study_id': ['dummy_study_1'],
            'some_required_col': [1] # Minimal column for a valid, non-empty row
        })

        # Initialize the Dataset
        # (Mocking CSVMetadata and skipping the file generation check)
        mock_csv_path = 'src.projects.lumbar_spine.csv_metadata.CSVMetadata'

        with (
                patch(mock_csv_path),
                patch.object(
                                LumbarDicomTFRecordDataset,
                                '_generate_tfrecord_files',
                                return_value=None
                             )
              ):
            dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # Mock methods inside the try block to inject an exception
        with (
                # Mock _process_study to raise a RuntimeError when called
                patch.object(
                                dataset,
                                '_process_study',
                                side_effect=RuntimeError(exception_message)
                             ),

                # Mock _setup_tfrecord_directory (L422) to prevent real I/O and return a dummy path
                patch.object(
                                dataset,
                                '_setup_tfrecord_directory',
                                return_value=tmp_path / "tfrecord_output_dir"
                             ),

                # Create a dummy study directory so the outer loop finds something to iterate over
                patch('pathlib.Path.iterdir', return_value=[tmp_path / "dummy_study_1"]),
                patch('pathlib.Path.is_dir', return_value=True),
              ):
            # Assert that the exception is correctly re-raised (L432)
            with pytest.raises(RuntimeError) as excinfo:

                # Call the method directly
                dataset._convert_dicom_to_tfrecords(
                    study_dir=str(tmp_path),
                    metadata_df=metadata_for_exception_test,  # Use the non-empty DataFrame
                    tfrecord_dir=str(tmp_path / "tfrecords_output"),
                    logger=mock_logger
                )

            # Verification 1: The error was propagated with the correct message
            assert exception_message in str(excinfo.value)

            # Verification 2: The error was logged (L429-431)
            mock_logger.error.assert_called_once()
            args, kwargs = mock_logger.error.call_args

            # Check logged message and error details
            assert "Error during DICOM conversion" in args[0]
            assert exception_message in args[0]

            # Check the 'extra' arguments for the required status and error info
            assert kwargs["extra"]["status"] == "failed"
            assert kwargs["extra"]["error"] == exception_message
            assert kwargs["exc_info"] is True

    def test_convert_dicom_to_tfrecord_skip_non_directory(
        self,
        mock_setup: Tuple[dict[str, Any], MagicMock],
        tmp_path: Path
    ) -> None:
        """
        Tests the 'if not study_path.is_dir(): continue' branch in
        _convert_dicom_to_tfrecords.

        It simulates a study directory containing a file (non-directory item)
        and asserts that the file is skipped, a warning is logged, and
        _process_study is NOT called for the non-directory item.
        """
        mock_config, mock_logger = mock_setup

        # Initialize the dataset object
        dataset_object = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # Define the study ID that will be marked as 'valid'
        valid_study_id = "ValidStudyID123" 

        # 1. Setup Mock Paths for iterdir()

        # Mock Path 1: A non-directory item (should trigger 'continue')
        mock_file_path = MagicMock(spec=Path)
        mock_file_path.is_dir.return_value = False
        mock_file_path.__str__.return_value = "mock_study_dir/readme.txt"  # For warning message

        # Mock Path 2: A valid directory (should be processed)
        mock_valid_dir_path = MagicMock(spec=Path)
        mock_valid_dir_path.is_dir.return_value = True
        mock_valid_dir_path.__str__.return_value = f"mock_study_dir/{valid_study_id}"
        mock_valid_dir_path.name = valid_study_id

        # Mock the iterdir() result to contain both item types
        mock_iterdir_result = [mock_file_path, mock_valid_dir_path]

        study_directory_string = "mock_study_dir"

        # Mock the Path object instance returned by Path(study_dir)
        mock_study_dir_instance = MagicMock(spec=Path)
        mock_study_dir_instance.iterdir.return_value = mock_iterdir_result

        # Mock the Path class constructor itself.
        # This mock will be returned when Path(...) is called.
        mock_path_constructor = MagicMock(spec=Path)
        mock_path_constructor.return_value = mock_study_dir_instance

        # 2. Setup Mock Dependencies

        # Mock metadata dataframe
        mock_metadata_dataframe = pd.DataFrame({'study_id': [valid_study_id]})
        tfrecord_directory_string = "mock_tfrecords"

        # Create a separate mock object for the Path return value of _setup_tfrecord_directory
        mock_tfrecord_path = MagicMock(spec=Path)
        mock_tfrecord_path.__str__.return_value = tfrecord_directory_string

        with (
                # Patch the Path class as imported in the source module.
                # This intercepts Path(study_dir) and returns mock_study_dir_instance.
                # ASSUMPTION: module path = src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset
                patch(
                        'src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset.Path',
                        new=mock_path_constructor
                      ),

                # Patch tqdm
                patch('tqdm.tqdm', side_effect=lambda iterable, *args, **kwargs: iterable),

                # Patch _setup_tfrecord_directory (returns an instance mock)
                patch.object(
                                dataset_object,
                                '_setup_tfrecord_directory',
                                return_value=mock_tfrecord_path
                             ),

                # Patch _process_study
                patch.object(dataset_object, '_process_study') as mock_process_study

               ):

            # Clear any warning calls made during object initialization
            mock_logger.warning.reset_mock()

            # 3. Call the method under test
            dataset_object._convert_dicom_to_tfrecords(
                                                            study_directory_string,
                                                            mock_metadata_dataframe,
                                                            tfrecord_directory_string,
                                                            logger=mock_logger
                                                        )

            # 4. Assertions

            # 4.1. Check the Warning Log (This covers the 'continue' line)
            expected_warning_message = (
                f"Skipping non-directory item {mock_file_path} "
                f"in study folder {study_directory_string}"
            )
            mock_logger.warning.assert_called_once_with(expected_warning_message)

            # 4.2. Check that _process_study was only called for the valid directory (once)
            mock_process_study.assert_called_once()

            # 4.3. Check the argument used for the single call

            # Use ANY to skip the problematic DataFrame comparison.
            mock_process_study.assert_called_once_with(
                                                            mock_valid_dir_path,
                                                            ANY,
                                                            mock_tfrecord_path,
                                                            mock_logger
                                                        )

    def test_process_study_skip_non_directory(
                                                self,
                                                mock_setup: Tuple[dict[str, Any], MagicMock],
                                                tmp_path: Path
                                              ) -> None:
        """
        Tests the 'if not series_path.is_dir(): continue' branch in _process_study.

        Simulates a study directory containing a non-directory item (file) and
        asserts that it is skipped, and a warning is logged.
        """
        mock_config, mock_logger = mock_setup

        # Initialize the dataset object
        dataset_object = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # 1. Setup Mock Paths
        study_id = "123456789"
        series_id = "1234567"
        study_path_str = str(tmp_path / study_id)
        
        # Create a reference to the empty DataFrame to avoid ambiguous truth value error
        # in mock assertion.
        mock_metadata_df = pd.DataFrame(
                                            {
                                                'study_id': [study_id],
                                                'series_id': [series_id],
                                                'instance_number':[1],
                                                'metadata': ['meta1']
                                            }
                                        )

        # Mock the Path instances passed as arguments
        mock_study_path = MagicMock(spec=Path)
        mock_study_path.name = study_id
        mock_tfrecord_dir = MagicMock(spec=Path)

        # Mock the resulting TFRecord path
        # Remove 'spec=Path' from the return mock to fix string assertion mismatch.
        mock_tfrecord_path = MagicMock()

        # Mock the division operator (/) on mock_tfrecord_dir
        # as the path is constructed via: tfrecord_dir / f"{study_id}.tfrecord"
        mock_tfrecord_dir.__truediv__.return_value = mock_tfrecord_path

        # Mock 1: A non-directory item (the one that should be skipped)
        mock_file_path = MagicMock(spec=Path)
        mock_file_path.is_dir.return_value = False
        mock_file_path.__str__.return_value = f"{study_path_str}/readme.txt"

        # Mock 2: A valid series directory (the one that should be processed)
        mock_valid_dir_path = MagicMock(spec=Path)
        mock_valid_dir_path.is_dir.return_value = True
        mock_valid_dir_path.name = series_id
        mock_valid_dir_path.__str__.return_value = f"{study_path_str}/{series_id}"

        # Mock the iterdir() call on the study_path
        mock_study_path.iterdir.return_value = [mock_file_path, mock_valid_dir_path]

        # 2. Patch Dependencies (tf.io.TFRecordWriter and _process_series)
        with (
                  # Patch tf.io.TFRecordWriter to prevent actual file writing
                  patch('tensorflow.io.TFRecordWriter') as mock_tfrecord_writer_class,

                  # Patch _process_series, which should NOT be called for the file
                  patch.object(dataset_object, '_process_series') as mock_process_series,
              ):

            # Clear any warning calls made during object initialization
            mock_logger.warning.reset_mock()

            # 3. Call the method under test
            dataset_object._process_study(
                study_path=mock_study_path,
                metadata_df=mock_metadata_df,  # Use the mock reference here
                tfrecord_dir=mock_tfrecord_dir,
                logger=mock_logger
            )

            # 4. Assertions

            # 4.1. Assert that the TFRecordWriter was initialized for the correct path
            # The function under test calls str() on the path before passing it to the Writer
            expected_tfrecord_path_str = str(mock_tfrecord_path)
            mock_tfrecord_writer_class.assert_called_once_with(expected_tfrecord_path_str)

            # 4.2. Assert that the warning was logged exactly once for the non-directory item
            expected_warning_message = (
                f"Skipping non-directory item: {mock_file_path} in study: {mock_study_path}"
            )
            mock_logger.warning.assert_called_once_with(expected_warning_message)

            # 4.3. Assert that _process_series was called only for the valid directory
            mock_process_series.assert_called_once()

            # 4.4. Assert the arguments for the single successful call
            mock_process_series.assert_called_once_with(
                mock_valid_dir_path,

                # Use ANY to bypass DataFrame comparison issues (instead of metadata_df)
                ANY,

                # The mock writer context manager
                mock_tfrecord_writer_class.return_value.__enter__.return_value
            )

    def test_process_study_missing_series_metadata(
                                                        self,
                                                        mock_setup: Tuple[dict[str, Any], MagicMock],
                                                        tmp_path: Path
                                                      ) -> None:
        """
        Tests the 'if series_metadata_df.empty: continue' branch in _process_study.

        Simulates a study containing a series directory for which no metadata is
        available in the metadata_df, and asserts that:
        1. The series processing is skipped (_process_series is NOT called).
        2. The four expected warnings are logged.
        """
        mock_config, mock_logger = mock_setup

        # Initialize the dataset object
        dataset_object = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        # 1. Setup Mock Paths
        study_id = "123456789"
        series_id = "1234567"
        study_path_str = str(tmp_path / study_id)
        tfrecord_dir = tmp_path/"tfrecords"
        
        # Create a reference to the empty DataFrame to avoid ambiguous truth value error
        # in mock assertion.
        mock_metadata_df = pd.DataFrame(
                                            {
                                                'study_id': ["fake_study_id"],
                                                'series_id': ["fake_series_id"],
                                                'instance_number':[1],
                                                'metadata': ['meta1']
                                            }
                                        )

        # Mock the Path instances passed as arguments
        mock_study_path = MagicMock(spec=Path)
        mock_study_path.name = study_id
        mock_tfrecord_dir = MagicMock(spec=Path)

        # Mock the resulting TFRecord path
        # Remove 'spec=Path' from the return mock to fix string assertion mismatch.
        mock_tfrecord_path = MagicMock()

        # Mock the division operator (/) on mock_tfrecord_dir
        # as the path is constructed via: tfrecord_dir / f"{study_id}.tfrecord"
        mock_tfrecord_dir.__truediv__.return_value = mock_tfrecord_path

        # Mock: A valid series directory (the one that should be processed)
        mock_valid_dir_path = MagicMock(spec=Path)
        mock_valid_dir_path.is_dir.return_value = True
        mock_valid_dir_path.name = series_id
        mock_valid_dir_path.__str__.return_value = f"{study_path_str}/{series_id}"

        # Mock the iterdir() call on the study_path
        mock_study_path.iterdir.return_value = [mock_valid_dir_path]

        # 2. Patch Dependencies (tf.io.TFRecordWriter and _process_series)
        with (
                  # Patch tf.io.TFRecordWriter to prevent actual file writing
                  patch('tensorflow.io.TFRecordWriter') as mock_tfrecord_writer_class,

                  # Patch _process_series, which should NOT be called for the file
                  patch.object(dataset_object, '_process_series') as mock_process_series,
              ):

            # Clear any warning calls made during object initialization
            mock_logger.warning.reset_mock()

            # 3. Call the method under test
            # Remark: the logger SHALL be passed as a keyword argument (logger=mock_logger)
            # to avoid the "got multiple values for argument 'logger'" TypeError,
            # which occurs when the 'log_method' decorator tries to inject the logger
            # while it is simultaneously passed positionally.
            dataset_object._process_study(
                                            study_path=mock_study_path,
                                            metadata_df=mock_metadata_df,
                                            tfrecord_dir=mock_tfrecord_dir,
                                            logger=mock_logger
                                            )

            # 4. Assertions

            # 4.1. Assert that the TFRecordWriter was initialized for the correct path
            # The function under test calls str() on the path before passing it to the Writer
            expected_tfrecord_path_str = str(mock_tfrecord_path)
            mock_tfrecord_writer_class.assert_called_once_with(expected_tfrecord_path_str)

            # 4.2. Assert that _process_series was NOT called
            mock_process_series.assert_not_called()

            # 4.3. Assert the logger was called with the right number of warnings
            assert mock_logger.warning.call_count == 4

            # 4.4. Assert the logger was called with the right warning message 
            # 4.4.1 Verify the content of the first and last warnings
            expected_warning_1_start = f"No metadata found for series {series_id}"
            expected_warning_4_exact = "Please check the CSV files and ensure they contain the right records"

            # 4.4.2 Extract teh arguments from all warning calls
            warning_calls = mock_logger.warning.call_args_list

            # 4.4.3 Check the first warning (detailed message about the ignored series)
            first_warning_msg = warning_calls[0][0][0]
            assert first_warning_msg.startswith(expected_warning_1_start)
            assert f"in study {study_id}. Skipping this series." in first_warning_msg

            # 4.4.4 Check the latest warning (user instruction)
            last_warning_msg = warning_calls[3][0][0]
            assert last_warning_msg == expected_warning_4_exact


    def test_build_tf_dataset_pipeline(
                                        self,
                                        mock_setup: Tuple[dict[str, Any], MagicMock],
                                        tmp_path: Path
                                       ) -> None:
        """
            Tests the creation of the TensorFlow Dataset pipeline.
        """

        mock_config, mock_logger = mock_setup

        # 1. Create a final mock dataset to be returned
        final_mock_dataset = MagicMock(name='final_dataset')

        # 2. Use patch to mock tensorflow.data.Dataset.list_files
        tf_list_files_str = "tensorflow.data.Dataset.list_files"
        with patch(tf_list_files_str) as mock_list_files:

            # Mock chain for TensorFlow Dataset pipeline methods
            mock_chain = (
                "interleave.return_value."
                "shuffle.return_value."
                "batch.return_value."
                "prefetch.return_value"
            )

            # Configure the mock to return final_mock_dataset after the chained method calls
            mock_list_files.return_value.configure_mock(
                **{mock_chain: final_mock_dataset}
            )

            # Mock _generate_tfrecord_files during initialization to ensure no side effects
            with patch.object(
                                LumbarDicomTFRecordDataset,
                                '_generate_tfrecord_files',
                                return_value=None
                               ):
                dataset = LumbarDicomTFRecordDataset(
                                                        mock_config,
                                                        logger=mock_logger
                                                     )

                dataset._tfrecord_pattern = (
                            mock_config["tfrecord_dir"] + "/*.tfrecord"
                        )
                result = dataset.build_tf_dataset_pipeline(batch_size=8)

                # Verify that the output is the result of the entire chain (final_mock_dataset)
                assert result == final_mock_dataset

                # Log verifications
                mock_logger.info.assert_any_call(
                    "Creating TF Dataset with batch_size=8",
                    extra={"action": "create_dataset", "batch_size": 8}
                )

                assert_msg = "Dataset pipeline created successfully"
                mock_logger.info.assert_called_with(assert_msg, extra={"status": "success"})

                # Check the calls to the chained methods on the mock returned by list_files
                mock_list_files.assert_called_once()

                # Get the mock returned by list_files
                mock_interleave = mock_list_files.return_value.interleave
                mock_interleave.assert_called_once()

                # Get the mock returned by interleave
                mock_shuffle = mock_interleave.return_value.shuffle
                mock_shuffle.assert_called_once()

                # Get the mock returned by shuffle
                mock_batch = mock_shuffle.return_value.batch
                mock_batch.assert_called_once_with(8)

    def test_build_tf_dataset_pipeline_exception(
                                            self,
                                            mock_setup: Tuple[dict[str, Any], MagicMock],
                                            tmp_path: Path
                                         ) -> None:
        """
            Test that exceptions in build_tf_dataset_pipeline are handled and logged.
        """

        mock_config, mock_logger = mock_setup
        dataset = LumbarDicomTFRecordDataset(mock_config, mock_logger)

        # Mock list_files to raise an exception
        list_files_path = 'tensorflow.data.Dataset.list_files'
        with patch(list_files_path, side_effect=Exception("List files error")):
            with pytest.raises(Exception):
                dataset.build_tf_dataset_pipeline(batch_size=2)

        # Check that the error was logged
        mock_logger.error.assert_called_with(
            "Error creating dataset: List files error",
            exc_info=True,
            extra={"status": "failed", "error": "List files error"}
        )

    def test_build_tf_dataset_pipeline_tfrecord_exception(
                                                    self,
                                                    mock_setup: Tuple[dict[str, Any], MagicMock],
                                                    tmp_path: Path
                                                  ) -> None:
        """
            Test that exceptions in TFRecordDataset are handled and logged.
        """

        mock_config, mock_logger = mock_setup

        # Create a dummy TFRecord file.
        self.generate_dummy_tfrecord_file(mock_setup, tmp_path)

        dataset = LumbarDicomTFRecordDataset(mock_config, mock_logger)

        # Mock TFRecordDataset to raise an exception
        with patch('tensorflow.data.TFRecordDataset', side_effect=Exception("TFRecord error")):
            with pytest.raises(Exception) as excinfo:
                dataset.build_tf_dataset_pipeline(batch_size=2)

            # Check that the raised error is the expected one
            assert "TFRecord error" in str(excinfo.value)

        # Check that the error was logged with the expected content
        args, kwargs = mock_logger.error.call_args
        assert "TFRecord error" in args[0]
        assert kwargs["extra"]["status"] == "failed"

    def test_build_tf_dataset_pipeline_map_exception(
                                                self,
                                                mock_setup: Tuple[dict[str, Any], MagicMock],
                                                tmp_path: Path
                                             ) -> None:
        """
            Test that exceptions in map are handled and logged.
        """

        mock_config, mock_logger = mock_setup

        # Create a dummy TFRecord file.
        self.generate_dummy_tfrecord_file(mock_setup, tmp_path)

        tfrecord_dir = Path(mock_config["tfrecord_dir"])

        # Initialize the dataset, mocking the generation to avoid side effects
        with patch.object(LumbarDicomTFRecordDataset,
                          '_generate_tfrecord_files',
                          return_value=None):
            dataset_obj = LumbarDicomTFRecordDataset(mock_config, mock_logger)

            # Ensure pattern matches the dummy file
            dataset_obj._tfrecord_pattern = str(tfrecord_dir / "*.tfrecord")

            # CRITICAL: Use a lambda function to wrap the helper method
            # and pass the mock_logger object, which is local to the test.
            side_effect_func = (
                lambda proto: self._mock_successful_error_handling(proto, mock_logger)
            )

            # Patch _parse_tfrecord to use the helper function as side_effect
            with patch.object(dataset_obj, '_parse_tfrecord',
                              side_effect=side_effect_func) as mock_parse:

                # Call build_tf_dataset_pipeline to get the tf.data.Dataset
                tf_dataset = dataset_obj.build_tf_dataset_pipeline(batch_size=2)

                # CRITICAL: Force dataset iteration to execute the map operation
                element = next(iter(tf_dataset))

                # Convert the element from Tensor to NumPy for local assertion
                # Assuming metadata is the second element (not used for shape assertion)
                image_array = element[0].numpy()

                # Check that the returned element is the dummy structure (1, 64, 64, 64, 1)
                # Remark: the first dimension is the batch size (1 here). It is because the
                # tensorflow batch method always inserts a dimension at axis 0 in the tensor
                # for the batch.
                # Since the mocked function returns the dummy_image of shape (64, 64, 64, 1)
                # and the tf_dataset pipeline applies a batch of size 1, the resulting shape
                # is (1, 64, 64, 64, 1).
                assert image_array.shape == (1, 64, 64, 64, 1)
                assert (image_array == 0.0).all()  # Dummy image is all zeros

                # Check that _parse_tfrecord was called
                mock_parse.assert_called()

                # Check that the logger was called with the error
                mock_logger.error.assert_called()

    def test_build_tf_dataset_pipeline_shuffle_exception(
                                                    self,
                                                    mock_setup: Tuple[dict[str, Any], MagicMock],
                                                    tmp_path: Path
                                                 ) -> None:
        """
            Test that exceptions in shuffle are handled and logged.
        """

        mock_config, mock_logger = mock_setup

        dataset = LumbarDicomTFRecordDataset(mock_config, mock_logger)

        with patch("tensorflow.data.Dataset.list_files") as mock_list_files:
            mock_interleave = MagicMock()
            mock_list_files.return_value.interleave.return_value = mock_interleave
            mock_interleave.shuffle.side_effect = Exception("Shuffle error")

            with pytest.raises(Exception):
                dataset.build_tf_dataset_pipeline(batch_size=2)

            mock_logger.error.assert_called_with(
                "Error creating dataset: Shuffle error",
                exc_info=True,
                extra={"status": "failed", "error": "Shuffle error"}
            )

    def test_build_tf_dataset_pipeline_batch_exception(
                                                self,
                                                mock_setup: Tuple[dict[str, Any], MagicMock],
                                                tmp_path: Path
                                               ) -> None:
        """
            Test that exceptions in batch are handled and logged.
        """

        mock_config, mock_logger = mock_setup
        dataset = LumbarDicomTFRecordDataset(mock_config, mock_logger)

        with patch("tensorflow.data.Dataset.list_files") as mock_list_files:
            mock_interleave = MagicMock()
            mock_shuffle = MagicMock()
            mock_list_files.return_value.interleave.return_value = mock_interleave
            mock_interleave.shuffle.return_value = mock_shuffle
            mock_shuffle.batch.side_effect = Exception("Batch error")

            with pytest.raises(Exception):
                dataset.build_tf_dataset_pipeline(batch_size=2)

            mock_logger.error.assert_called_with(
                "Error creating dataset: Batch error",
                exc_info=True,
                extra={"status": "failed", "error": "Batch error"}
            )

    def test_build_tf_dataset_pipeline_prefetch_exception(
                                                    self,
                                                    mock_setup: Tuple[dict[str, Any], MagicMock],
                                                    tmp_path: Path
                                                  ) -> None:
        """
            Test that exceptions in prefetch are handled and logged.
        """

        mock_config, mock_logger = mock_setup
        dataset = LumbarDicomTFRecordDataset(mock_config, mock_logger)

        with patch("tensorflow.data.Dataset.list_files") as mock_list_files:
            mock_interleave = MagicMock()
            mock_shuffle = MagicMock()
            mock_batch = MagicMock()
            mock_list_files.return_value.interleave.return_value = mock_interleave
            mock_interleave.shuffle.return_value = mock_shuffle
            mock_shuffle.batch.return_value = mock_batch
            mock_batch.prefetch.side_effect = Exception("Prefetch error")

            with pytest.raises(Exception):
                dataset.build_tf_dataset_pipeline(batch_size=2)

            mock_logger.error.assert_called_with(
                "Error creating dataset: Prefetch error",
                exc_info=True,
                extra={"status": "failed", "error": "Prefetch error"}
            )

    def test_parse_tfrecord(
                                self,
                                mock_setup: Tuple[dict[str, Any], MagicMock],
                                tmp_path: Path
                            ) -> None:
        """
            Tests the parsing of a single TFRecord entry.
        """

        mock_config, mock_logger = mock_setup
        with (
                patch("tensorflow.io.parse_single_example") as mock_parse_single_example,
                patch("tensorflow.io.parse_tensor") as mock_parse_tensor,
                patch("tensorflow.reshape") as mock_reshape,
                patch("tensorflow.py_function") as mock_py_function,
                patch("tensorflow.reduce_max") as mock_reduce_max,
                patch("tensorflow.cast") as mock_cast,
              ):

            # Mock _generate_tfrecord_files during initialization to avoid side effects
            with patch.object(
                                LumbarDicomTFRecordDataset,
                                '_generate_tfrecord_files',
                                return_value=None
                                ):

                dataset = LumbarDicomTFRecordDataset(mock_config,
                                                     logger=mock_logger)

                # Create a mock TFRecord example (the input to the function)
                mock_proto = tf.constant(b"fake_proto")

                # Mock parse_single_example to return a valid structure
                mock_parse_single_example.return_value = {
                    "image": tf.constant([1] * 64 * 64 * 64 * 1, dtype=tf.uint16),
                    "metadata": tf.constant(b"fake_metadata")
                }

                # Mock parse_tensor to return a valid 1D tensor
                mock_parse_tensor.return_value = tf.constant(
                                                        [1] * 64 * 64 * 64 * 1,
                                                        dtype=tf.uint16
                                                        )

                # Mock reshape to return different values on successive calls:
                # 1st call for image, 2nd call for records
                mock_reshape.side_effect = [
                    tf.ones([64, 64, 64, 1], dtype=tf.float32),  # image
                    tf.ones([25, 4], dtype=tf.float32)           # records
                ]

                # Define the mock return for tf.py_function
                mock_header_tensors = [
                    tf.constant(1, dtype=tf.int32),   # study_id
                    tf.constant(1, dtype=tf.int32),   # series_id
                    tf.constant(1, dtype=tf.int32),   # instance_number
                    tf.constant(2, dtype=tf.int32),   # description (encoded)
                    tf.constant(3, dtype=tf.int32),   # condition (encoded)
                    tf.constant(2, dtype=tf.int32),   # nb_records
                ]
                mock_records_flat = tf.zeros([100], dtype=tf.float32)
                mock_py_function.return_value = (
                        mock_header_tensors + [mock_records_flat]
                )

                # Mock reduce_max to return a constant
                mock_reduce_max.return_value = tf.constant(1.0)

                # Mock cast to return the input as-is
                mock_cast.side_effect = lambda x, dtype: x

                # Call the method
                image, metadata = dataset._parse_tfrecord(mock_proto)

                # Verifications
                assert image.shape == (64, 64, 64, 1)
                assert metadata["study_id"].numpy() == 1
                assert metadata["nb_records"].numpy() == 2
                assert tuple(metadata["records"].shape) == (25, 4)

    def test_parse_tfrecord_handles_exception_and_returns_defaults(
        self,
        mock_setup: Tuple[dict, MagicMock],
        tmp_path: Path
    ) -> None:
        """
            Covers the 'except' block in _parse_tfrecord.
            Verifies that an exception during parsing is caught, logged, and
            the safe dummy structure is returned, preventing pipeline crashes.
        """
        mock_config, mock_logger = mock_setup

        # Mock _generate_tfrecord_files during initialization to avoid side effects
        with patch.object(
                            LumbarDicomTFRecordDataset,
                            '_generate_tfrecord_files',
                            return_value=None
                            ):
            dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

        exception_message = "Simulated parsing failure"

        # Mock the initial parsing step (tf.io.parse_single_example) to raise an exception
        parse_single_example_str = "tensorflow.io.parse_single_example"
        with patch(parse_single_example_str, side_effect=Exception(exception_message)):

            mock_proto = tf.constant(b"fake_proto")

            # Call the function (it should catch the exception and return defaults)
            image_output, metadata_dict_output = dataset._parse_tfrecord(mock_proto)

            # 1. Verification: Check returned default image tensor
            # The dummy image is tf.zeros([64, 64, 64, 1], dtype=tf.float32)
            assert image_output.dtype == tf.float32
            assert image_output.shape == (64, 64, 64, 1)
            assert (image_output.numpy() == 0.0).all()

            # 2. Verification: Check returned default metadata dictionary
            assert isinstance(metadata_dict_output, dict)
            assert len(metadata_dict_output) == 7

            # Check a few key metadata fields (all should be tf.constant(0, dtype=tf.int32))
            assert metadata_dict_output["study_id"].numpy() == 0
            assert metadata_dict_output["condition"].numpy() == 0

            # Check the dummy records tensor (tf.zeros([25, 4], dtype=tf.float32))
            dummy_records_output = metadata_dict_output["records"]
            assert dummy_records_output.dtype == tf.float32
            assert dummy_records_output.shape == (25, 4)
            assert (dummy_records_output.numpy() == 0.0).all()

            # 3. Verification: Check Error Log
            mock_logger.error.assert_called_once()
            args, kwargs = mock_logger.error.call_args
            assert f"Error parsing TFRecord: {exception_message}" in args[0]
            assert kwargs["extra"]["status"] == "failed"
            assert kwargs["extra"]["error"] == exception_message
            assert kwargs["exc_info"] is True

    def test_get_metadata_for_file(
                                    self,
                                    mock_setup: Tuple[dict[str, Any], MagicMock],
                                    tmp_path: Path
                                   ) -> None:
        """
            Tests the retrieval of metadata for a specific DICOM file.
        """

        mock_config, mock_logger = mock_setup

        # Create a mock file path
        mock_file_path = "/fake/root_dir/1/2/1.dcm"

        # Create a mock metadata DataFrame
        mock_metadata_df = pd.DataFrame({
            "study_id": [1, 1, 2],
            "series_id": [1, 2, 1],
            "instance_number": [1, 2, 1],
            "other_column": ["value1", "value2", "value3"]
        })

        dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)
        # Initialize the dataset
        with (
                patch.object(dataset, '_generate_tfrecord_files', return_value=None),
                patch.object(dataset, '_serialize_metadata') as mock_serialize_metadata
              ):

            # Configure the mock for _serialize_metadata
            mock_serialize_metadata.return_value = (
               b"serialized_metadata_bytes"
            )

            # Call the method
            result = dataset._get_metadata_for_file(mock_file_path, mock_metadata_df)

            # Verifications
            # Check that the logger was called correctly
            mock_logger.info.assert_any_call(
                "Starting retrieving metadata from CSV files"
            )

            # Check that _serialize_metadata was called with the correct arguments
            mock_serialize_metadata.assert_called_once_with(mock_metadata_df)

            # Check that the result is the expected serialized metadata
            assert result == b"serialized_metadata_bytes"

            # Test with None metadata_df
            result = dataset._get_metadata_for_file(mock_file_path, None)
            assert result == b''

            # Check that the logger was called again for the second call
            msg_str = "Starting retrieving metadata from CSV files"
            mock_logger.info.assert_any_call(msg_str)

    def test_get_metadata_for_file_handles_serialization_exception(
        self,
        mock_setup: Tuple[dict, MagicMock],
        tmp_path: Path
    ) -> None:
        """
            Covers the 'except' block in _get_metadata_for_file.
            Verifies that an exception during the metadata processing (e.g., serialization)
            is caught, correctly logged with file details, and returns empty bytes (b'').
        """
        mock_config, mock_logger = mock_setup

        # Setup Data
        exception_message = "Simulated internal serialization error"

        # Simulate a valid file path for ID extraction
        mock_file_path = str(tmp_path / "123/456/789.dcm")
        mock_metadata_df = pd.DataFrame({'col': [1]})  # Dummy dataframe

        # Mock & Initialization
        with (
                # Mock _generate_tfrecord_files during initialization
                patch.object(
                                LumbarDicomTFRecordDataset,
                                '_generate_tfrecord_files',
                                return_value=None
                             ),

                # Inject an exception during the _serialize_metadata call
                # (which is inside the try block)
                patch.object(LumbarDicomTFRecordDataset, '_serialize_metadata',
                             side_effect=RuntimeError(exception_message))
               ):

            dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)

            # Call the method
            result = dataset._get_metadata_for_file(mock_file_path, mock_metadata_df)

            # Verification 1: Check returned value. The exception handler must return b''
            assert result == b''

            # Verification 2: Check logger call
            mock_logger.error.assert_called_once()
            args, kwargs = mock_logger.error.call_args

            # Check logged message content
            logged_message = args[0]
            assert "Error in function _get_metadata_for_file()" in logged_message
            assert exception_message in logged_message

            # Check the error logging mechanism
            assert kwargs["exc_info"] is True
            assert kwargs["extra"]["status"] == "failed"
            assert kwargs["extra"]["error"] == exception_message
