# coding: utf-8

import unittest
from unittest.mock import patch, MagicMock
import logging
import tensorflow as tf

# Use local import to ensure coverage tracker is active
import src.RSNA_2024_Lumbar_Spine_Degenerative_Classification as src_module


class TestRSNALumbarClassification(unittest.TestCase):

    def setUp(self):
        # Setup a standard mock configuration dictionary
        self.mock_config = {
            "output_dir": "tests/logs",
            "checkpoint_path": "tests/checkpoints/model.keras",
            "max_records": 10,
            "learning_rate": 0.001,
            "clipnorm": 1.0,
            "nb_cores": 1,
            "system_stream_mirror_path": "tests/logs/system.log",
            "log_retention_days": 7
        }
        self.mock_logger = MagicMock(spec=logging.Logger)

    # --------------------------------------------------------------------------
    # TESTS FOR load_model
    # --------------------------------------------------------------------------

    @patch("src.RSNA_2024_Lumbar_Spine_Degenerative_Classification.ModelFactory")
    @patch("src.RSNA_2024_Lumbar_Spine_Degenerative_Classification.Path")
    def test_load_model_new_instance(self, mock_path, mock_factory):
        """Test load_model builds a new model when no checkpoint exists."""

        # English comment: Simulate that the checkpoint file does not exist
        mock_path.return_value.resolve.return_value.is_file.return_value = False
        
        # Mocking factory return
        mock_model = MagicMock(spec=tf.keras.Model)
        mock_factory.return_value.build_multi_series_model.return_value = mock_model

        model = src_module.load_model(depth=60, config=self.mock_config, logger=self.mock_logger)

        self.assertEqual(model, mock_model)
        mock_model.compile.assert_called_once()
        self.mock_logger.info.assert_any_call("No existing model found. Building a new model from factory.")

    def test_load_model_invalid_config(self):
        """Test load_model raises ValueError if config is incomplete."""
        incomplete_config = {"checkpoint_path": "some/path"} # Missing learning_rate, etc.
        with self.assertRaises(ValueError):
            src_module.load_model(depth=60, config=incomplete_config, logger=self.mock_logger)

    @patch("src.RSNA_2024_Lumbar_Spine_Degenerative_Classification.tf.keras.models.load_model")
    @patch("src.RSNA_2024_Lumbar_Spine_Degenerative_Classification.ModelFactory")
    @patch("src.RSNA_2024_Lumbar_Spine_Degenerative_Classification.Path")
    def test_load_model_corrupted_file(self, mock_path, mock_factory, mock_tf_load):
        """
        Test fallback to factory when the model file is corrupted.
        """

        # Simulate that file exists but loading raises an error
        mock_path.return_value.resolve.return_value.is_file.return_value = True
        mock_tf_load.side_effect = Exception("Corrupted model file")
        
        # Ensure the factory still produces a model
        mock_model = MagicMock(spec=tf.keras.Model)
        mock_factory.return_value.build_multi_series_model.return_value = mock_model

        model = src_module.load_model(depth=60, config=self.mock_config, logger=self.mock_logger)

        # Verify that we logged a warning and still got a model
        self.mock_logger.warning.assert_any_call(
            "Failed to load existing model: Corrupted model file. Falling back to factory."
        )
        self.assertIsNotNone(model)

    # --------------------------------------------------------------------------
    # TESTS FOR main flow
    # --------------------------------------------------------------------------

    @patch("src.RSNA_2024_Lumbar_Spine_Degenerative_Classification.tf") # Mock tensorflow to avoid hardware config errors
    @patch("src.RSNA_2024_Lumbar_Spine_Degenerative_Classification.ConfigLoader")
    @patch("src.RSNA_2024_Lumbar_Spine_Degenerative_Classification.setup_logger")
    @patch("src.RSNA_2024_Lumbar_Spine_Degenerative_Classification.TFRecordFilesManager")
    @patch("src.RSNA_2024_Lumbar_Spine_Degenerative_Classification.ModelTrainer")
    @patch("src.RSNA_2024_Lumbar_Spine_Degenerative_Classification.load_model")
    @patch("src.RSNA_2024_Lumbar_Spine_Degenerative_Classification.SystemStreamTee")
    @patch("src.RSNA_2024_Lumbar_Spine_Degenerative_Classification.clean_old_logs")
    def test_main_success_flow(self, mock_clean, mock_tee, mock_load, mock_trainer, 
                               mock_tf_mgr, mock_setup_log, mock_cfg_loader, mock_tf):
        """Test the full execution flow of main()."""
        
        # Setup mock returns
        mock_cfg_loader.return_value.get.return_value = self.mock_config
        mock_setup_log.return_value.__enter__.return_value = self.mock_logger
        mock_tf_mgr.return_value.get_max_series_depth.return_value = 60
        
        # Execute main - the mocked 'tf' will now prevent the RuntimeError
        src_module.main()
        
        # Verify tf config was called (optional)
        mock_tf.config.threading.set_intra_op_parallelism_threads.assert_called()

        # Verify sequences
        mock_tf_mgr.return_value.generate_tfrecord_files.assert_called_once()
        mock_load.assert_called_once()
        mock_trainer.return_value.prepare_training_and_validation_datasets.assert_called_once()
        mock_trainer.return_value.train_model.assert_called_once()
        mock_clean.assert_called_once()

    # --------------------------------------------------------------------------
    # TESTS FOR signals
    # --------------------------------------------------------------------------

    @patch("src.RSNA_2024_Lumbar_Spine_Degenerative_Classification.get_current_logger")
    @patch("sys.exit")
    def test_handle_interrupt(self, mock_exit, mock_get_logger):
        """Test that SIGINT handler logs message and exits."""
        mock_get_logger.return_value = self.mock_logger
        
        src_module.handle_interrupt(None, None)
        
        self.mock_logger.info.assert_called_with("\nInterruption detected (Ctrl+C). Exiting gracefully...")
        mock_exit.assert_called_with(0)

if __name__ == "__main__":
    unittest.main()