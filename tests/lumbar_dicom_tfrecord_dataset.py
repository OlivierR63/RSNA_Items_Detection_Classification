# tests/test_dicom_dataset.py
from unittest.mock import patch, MagicMock
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
import pytest
import pandas as pd
import tensorflow as tf

@patch("src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset.CSVMetadata")
def test_generate_tfrecord_files(mock_csv_metadata, mock_config, mock_logger):
    """Test la génération des fichiers TFRecord."""
    mock_csv_instance = MagicMock()
    mock_csv_metadata.return_value = mock_csv_instance
    mock_csv_instance._merged_df = pd.DataFrame({"study_id": [1], "series_id": [1], "instance_number": [1]})

    dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)
    dataset._tfrecord_dir = Path("tests/tmp/tfrecords")
    dataset._tfrecord_dir.mkdir(parents=True, exist_ok=True)

    # Mock la méthode _convert_dicom_to_tfrecords pour éviter les E/S réelles
    dataset._convert_dicom_to_tfrecords = MagicMock()

    dataset.generate_tfrecord_files()

    mock_logger.info.assert_called_with("Starting generate_tfrecord_file", extra={"action": "generate_tf_records"})
    dataset._convert_dicom_to_tfrecords.assert_called_once()

@patch("tensorflow.data.Dataset.list_files")
@patch("tensorflow.data.Dataset.interleave")
def test_create_tf_dataset(mock_interleave, mock_list_files, mock_config, mock_logger):
    """Test la création du dataset TensorFlow."""
    mock_list_files.return_value = MagicMock()
    mock_interleave.return_value = MagicMock()

    dataset = LumbarDicomTFRecordDataset(mock_config, logger=mock_logger)
    dataset._tfrecord_pattern = "tests/tmp/tfrecords/*.tfrecord"

    result = dataset.create_tf_dataset(batch_size=8)

    assert result == mock_interleave.return_value
    mock_logger.info.assert_called_with("Creating TF Dataset with batch_size=8", extra={"action": "create_dataset", "batch_size": 8})

def test_parse_tfrecord(mock_logger):
    """Test le parsing d'un TFRecord."""
    dataset = LumbarDicomTFRecordDataset({}, logger=mock_logger)

    # Créer un exemple de TFRecord mocké
    mock_proto = tf.constant(b"fake_proto")
    with patch.object(dataset, "_deserialize_metadata") as mock_deserialize:
        mock_deserialize.return_value = {
            "study_id": 1,
            "series_id": 1,
            "instance_number": 1,
            "description": "test",
            "condition": 0,
            "nb_records": 0,
            "records": []
        }

        with patch("tensorflow.io.parse_single_example") as mock_parse:
            mock_parse.return_value = {
                "image": tf.constant(b"fake_image"),
                "metadata": tf.constant(b"fake_metadata")
            }
            with patch("tensorflow.io.parse_tensor") as mock_parse_tensor:
                mock_parse_tensor.return_value = tf.constant([[1.0]])
                with patch("tensorflow.reshape") as mock_reshape:
                    mock_reshape.return_value = tf.constant([[[[1.0]]]])

                    image, metadata = dataset._parse_tfrecord(mock_proto)

                    assert image.shape == (64, 64, 64, 1)
                    assert metadata["study_id"] == 1
                    mock_logger.info.assert_called_with("Parsing TFRecord", extra={"action": "_parse_tfrecord"})
