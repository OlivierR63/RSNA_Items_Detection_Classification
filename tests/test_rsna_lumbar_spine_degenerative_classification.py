# coding: utf-8

from unittest.mock import patch, MagicMock
from src.RSNA_2024_Lumbar_Spine_Degenerative_Classification import main
from tests import conftest
import pytest

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

