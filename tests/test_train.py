# tests/test_train.py
from unittest.mock import patch, MagicMock
from src.projects.lumbar_spine.train import train_model, create_tf_dataset, train_with_callbacks, main
import pytest

@patch("src.projects.lumbar_spine.train.LumbarDicomTFRecordDataset")
@patch("src.projects.lumbar_spine.train.ModelFactory")
def test_train_model(mock_model_factory, mock_dataset, mock_config, mock_logger):
    """Test la fonction train_model."""
    # Setup mocks
    mock_model = MagicMock()
    mock_model_factory.create_model.return_value = mock_model

    mock_dataset_instance = MagicMock()
    mock_dataset.return_value = mock_dataset_instance
    mock_dataset_instance.create_tf_dataset.return_value = "mock_dataset"

    # Appel de la fonction
    train_model(config=mock_config, logger=mock_logger)

    # Vérifications
    mock_model_factory.create_model.assert_called_once_with(mock_config["model_3d"])
    mock_logger.info.assert_any_call("Loading 3D model...", extra={"action": "load_model", "model_type": mock_config['model_3d']['type']})
    mock_logger.info.assert_called_with("Model saved to tests/tmp/model", extra={"status": "success", "model_path": "tests/tmp/model"})

@patch("src.projects.lumbar_spine.train.LumbarDicomTFRecordDataset")
def test_create_tf_dataset(mock_dataset, mock_config, mock_logger):
    """Test la fonction create_tf_dataset."""
    mock_dataset_instance = MagicMock()
    mock_dataset.return_value = mock_dataset_instance
    mock_dataset_instance.create_tf_dataset.return_value = "mock_dataset"

    result = create_tf_dataset(config=mock_config, logger=mock_logger)

    assert result == "mock_dataset"
    mock_logger.info.assert_called_with("Dataset created successfully.", extra={"status": "success", "batch_size": mock_config["batch_size"]})

@patch("tensorflow.keras.Model.fit")
@patch("src.projects.lumbar_spine.train.tf.keras.callbacks.ModelCheckpoint")
@patch("src.projects.lumbar_spine.train.tf.keras.callbacks.EarlyStopping")
@patch("src.projects.lumbar_spine.train.tf.keras.callbacks.TensorBoard")
def test_train_with_callbacks(mock_tensorboard, mock_earlystopping, mock_checkpoint, mock_fit, mock_config, mock_logger):
    """Test la fonction train_with_callbacks."""
    mock_model = MagicMock()
    mock_dataset = MagicMock()
    mock_history = MagicMock()
    mock_fit.return_value = mock_history
    mock_history.history = {"loss": [0.5, 0.3], "accuracy": [0.8, 0.9]}

    result = train_with_callbacks(mock_model, mock_dataset, mock_config, logger=mock_logger)

    assert result == mock_history
    mock_logger.info.assert_called_with("Model training completed successfully.",
                                        extra={"final_loss": 0.3, "final_accuracy": 0.9})

@patch("src.projects.lumbar_spine.train.setup_logger")
@patch("src.projects.lumbar_spine.train.ConfigLoader")
@patch("src.projects.lumbar_spine.train.train_model")
def test_main(mock_train_model, mock_config_loader, mock_setup_logger, mock_logger):
    """Test la fonction main."""
    mock_config = {"output_dir": "tests/tmp", "batch_size": 8, "epochs": 2, "model_3d": {"type": "mock"}}
    mock_config_loader.return_value.get.return_value = mock_config
    mock_setup_logger.return_value.__enter__.return_value = mock_logger

    with patch("src.projects.lumbar_spine.train.signal.signal"):
        main()

    mock_config_loader.assert_called_once_with("src/config/lumbar_spine_config.yaml")
    mock_train_model.assert_called_once_with(config=mock_config)
    mock_logger.info.assert_called_with("Starting training process.", extra={"status": "started", "log_dir": "tests/tmp/logs"})

