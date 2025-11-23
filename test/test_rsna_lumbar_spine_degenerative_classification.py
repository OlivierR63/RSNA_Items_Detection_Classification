# coding: utf-8

from unittest.mock import patch, MagicMock
from src.RSNA_2024_Lumbar_Spine_Degenerative_Classification import main
import logging


@patch("src.projects.lumbar_spine.train.setup_logger")
@patch("src.projects.lumbar_spine.train.ConfigLoader")
@patch("src.projects.lumbar_spine.train.train_model")
def test_main(mock_train_model, mock_config_loader, mock_setup_logger):
    """Test the main function."""

    # Mock logger
    mock_logger = MagicMock(spec=logging.Logger)
    mock_setup_logger.return_value.__enter__.return_value = mock_logger

    # Mock config
    mock_config = {
                    "output_dir": "tests/tmp",
                    "batch_size": 8,
                    "epochs": 2,
                    "model_3d": {"type": "mock"}
                    }
    mock_config_loader.return_value.get.return_value = mock_config

    # Call the main function here
    main()

    # Vérifications
    mock_config_loader.return_value.get.assert_called_once()
    mock_setup_logger.assert_called_once()
    mock_train_model.assert_called_once_with(config=mock_config, logger=mock_logger)
