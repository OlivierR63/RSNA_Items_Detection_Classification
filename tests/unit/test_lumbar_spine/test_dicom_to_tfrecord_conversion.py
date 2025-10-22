#coding: utf-8

import pytest
from unittest.mock import patch, MagicMock
import tensorflow as tf
import pandas as pd
from pathlib import Path
import logging
import SimpleITK as sitk
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset


class TestDicomToTFRecordConversion:
    """Unit tests for DICOM to TFRecord conversion in Lumbar Spine project."""


    @pytest.fixture(autouse=True)
    def setup(self, mock_config, mock_logger):
        """Fixture to initialize common attributes for all tests."""
        
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


    def test_convert_dicom_to_tfrecords(self, tmp_path):
        """Test the main function for converting DICOM to TFRecord."""

        with patch("src.core.utils.logger.get_current_logger", return_value=self.mock_logger), \
            patch("src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset.CSVMetadata") as mock_csv_metadata_class, \
            patch.object(LumbarDicomTFRecordDataset, '_encode_dataframe') as mock_encode_dataframe, \
            patch.object(LumbarDicomTFRecordDataset, '_generate_tfrecord_files') as mock_generate_tfrecord_files:

            # Configure the mock CSVMetadata class
            mock_csv_metadata_instance = MagicMock()
            mock_csv_metadata_class.return_value = mock_csv_metadata_instance
            mock_csv_metadata_instance._merged_df = self.metadata_df

            # Mock _encode_dataframe to return the same DataFrame without processing
            mock_encode_dataframe.return_value = self.metadata_df

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config, logger=self.mock_logger)
        
            # Now patch methods on the instance
            with patch.object(dataset, '_setup_output_directory') as mock_setup_output_directory, \
                patch.object(dataset, '_process_study') as mock_process_study:

                mock_setup_output_directory.return_value = self.output_dir
                mock_process_study.return_value = None

                # Call the function
                str_study_id = str(self.root_dir) + "/4003253"
                dataset._convert_dicom_to_tfrecords(str_study_id, self.metadata_df, str(tmp_path))

                # Verifications
                mock_setup_output_directory.assert_called_once_with(str(tmp_path))
                mock_process_study.assert_called()
                self.mock_logger.info.assert_any_call(
                    "Starting DICOM to TFRecord conversion",
                    extra={"action": "convert_dicom", "root_dir": str_study_id}
                )
                self.mock_logger.info.assert_called_with(
                    "DICOM to TFRecord conversion completed successfully",
                    extra={"status": "success"}
                )


    def test_setup_output_directory(self, tmp_path):
        """Test the creation of the output directory."""
        with patch("src.core.utils.logger.get_current_logger", return_value=self.mock_logger), \
            patch("src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset.CSVMetadata") as mock_csv_metadata_class, \
            patch.object(LumbarDicomTFRecordDataset, '_generate_tfrecord_files') as mock_generate_tfrecord_files:

            # Mock _generate_tfrecord_files to avoid side effects
            mock_generate_tfrecord_files.return_value = None

            # Configure the mock CSVMetadata class
            mock_csv_metadata_instance = MagicMock()
            mock_csv_metadata_class.return_value = mock_csv_metadata_instance
            mock_csv_metadata_instance._merged_df = self.metadata_df

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config, logger=self.mock_logger)
            
            # Call the function
            output_dir = dataset._setup_output_directory(str(tmp_path / "output"))

            # Verifications
            assert output_dir == tmp_path / "output"
            assert output_dir.is_dir()


    def test_process_study(self, tmp_path):
        """Test the processing of a study."""
        study_path = tmp_path / "123465879"
        study_path.mkdir()
        series_path = study_path / "1235647"
        series_path.mkdir()

        with patch("src.core.utils.logger.get_current_logger", return_value=self.mock_logger), \
             patch("src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset.CSVMetadata") as mock_csv_metadata_class, \
             patch.object(LumbarDicomTFRecordDataset, '_generate_tfrecord_files') as mock_generate_tfrecord_files:

             # Mock _generate_tfrecord_files to avoid side effects
             mock_generate_tfrecord_files.return_value = None

             # Configure the mock CSVMetadata class
             mock_csv_metadata_instance = MagicMock()
             mock_csv_metadata_class.return_value = mock_csv_metadata_instance
             mock_csv_metadata_instance._merged_df = self.metadata_df

             # Initialize the dataset with the mock logger
             dataset = LumbarDicomTFRecordDataset(self.mock_config, logger=self.mock_logger)
            
             with patch.object(dataset, '_process_series') as mock_process_series:
                 mock_process_series.return_value = None

                 # Utiliser tmp_path pour le fichier de sortie
                 output_dir = tmp_path / "tfrecords"
                 output_dir.mkdir(parents=True, exist_ok=True)

                 # Call the function
                 dataset._process_study(study_path, self.metadata_df, output_dir, self.mock_logger)

                 # Verify that the TFRecord file has been actually created
                 tfrecord_file = output_dir / "123465879.tfrecord"
                 assert tfrecord_file.is_file()

                 # Verifications
                 mock_process_series.assert_called_once()


    def test_process_series(self, tmp_path):
        """Test the processing of a series."""
        series_path = tmp_path / "123456789"
        series_path.mkdir()
        (series_path / "1.dcm").write_text("dummy")

        with patch("src.core.utils.logger.get_current_logger", return_value=self.mock_logger), \
            patch("src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset.CSVMetadata") as mock_csv_metadata_class, \
            patch.object(LumbarDicomTFRecordDataset, '_generate_tfrecord_files') as mock_generate_tfrecord_files:

            # Mock _generate_tfrecord_files to avoid side effects
            mock_generate_tfrecord_files.return_value = None

            # Configure the mock CSVMetadata class
            mock_csv_metadata_instance = MagicMock()
            mock_csv_metadata_class.return_value = mock_csv_metadata_instance
            mock_csv_metadata_instance._merged_df = self.metadata_df

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config, logger=self.mock_logger)

            with patch.object(dataset, '_process_dicom_file') as mock_process_dicom_file:
                mock_process_dicom_file.return_value = (b"img_bytes", b"metadata_bytes")
                mock_writer = MagicMock(spec=tf.io.TFRecordWriter)

                # Call the function
                dataset._process_series(series_path, self.metadata_df, mock_writer)

                # Verifications
                mock_process_dicom_file.assert_called_once()
                mock_writer.write.assert_called_once()


    def test_process_dicom_file(self, tmp_path):
        """Test the processing of a DICOM file."""
        dicom_path = tmp_path / "file1.dcm"
        dicom_path.write_text("dummy")

        with patch("src.core.utils.logger.get_current_logger", return_value=self.mock_logger), \
            patch("SimpleITK.ReadImage") as mock_read_image, \
            patch.object(LumbarDicomTFRecordDataset, '_get_metadata_for_file') as mock_get_metadata, \
            patch.object(LumbarDicomTFRecordDataset, '_generate_tfrecord_files') as mock_generate_tfrecord_files:

            # Mock _generate_tfrecord_files to avoid side effects
            mock_generate_tfrecord_files.return_value = None

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config, logger=self.mock_logger)

            # Configure the mock for SimpleITK.ReadImage
            mock_img = MagicMock()
            mock_img.GetPixelIDValue.return_value = 2  # sitkUInt16
            mock_read_image.return_value = mock_img

            # Mock sitk.GetArrayFromImage to return a numpy array
            with patch("SimpleITK.GetArrayFromImage") as mock_get_array_from_image:
                mock_img_array = [[1, 2], [3, 4]]
                mock_get_array_from_image.return_value = mock_img_array

                mock_get_metadata.return_value = b"metadata_bytes"

                # Call the function
                img_bytes, metadata_bytes = dataset._process_dicom_file(dicom_path, self.metadata_df)

                # Verifications
                mock_read_image.assert_called_once_with(str(dicom_path))
                mock_get_array_from_image.assert_called_once_with(mock_img)
                mock_get_metadata.assert_called_once_with(str(dicom_path), self.metadata_df)
                assert isinstance(img_bytes, bytes)
                assert metadata_bytes == b"metadata_bytes"


    def test_write_tfrecord_example(self):
        """Test the writing of a TFRecord example."""
        
        with patch("src.core.utils.logger.get_current_logger", return_value=self.mock_logger):
            
            mock_writer = MagicMock(spec=tf.io.TFRecordWriter)
            img_bytes = b"img_bytes"
            metadata_bytes = b"metadata_bytes"

            # Initialize the dataset with the mock logger
            dataset = LumbarDicomTFRecordDataset(self.mock_config, logger=self.mock_logger)

            # Call the function
            dataset._write_tfrecord_example(img_bytes, metadata_bytes, mock_writer)

            # Verifications
            mock_writer.write.assert_called_once()
            args, _ = mock_writer.write.call_args
            assert isinstance(args[0], bytes)
