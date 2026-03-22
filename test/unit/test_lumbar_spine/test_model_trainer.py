# coding utf-8

import pytest
import tensorflow as tf
from pathlib import Path
import shutil
from unittest.mock import MagicMock, patch
from src.projects.lumbar_spine.model_trainer import log_memory_usage
from contextlib import ExitStack
from src.projects.lumbar_spine.model_trainer import ModelTrainer


class TestModelTrainer:
    """
    Unit tests for ModelTrainer.
    Leverages fixtures from conftest.py for configuration, logging, and data structures.
    """

    @pytest.fixture
    def mock_trainer(self, mock_config, mock_logger):
        """
        Local fixture to initialize ModelTrainer.
        Note: The model itself is mocked as it's not the unit under test.
        """
        mock_model = MagicMock(spec=tf.keras.Model)

        # Ensure the expected TFRecord directory exists for path resolution tests
        mock_config["paths"]["tfrecord"].mkdir(parents=True, exist_ok=True)

        return ModelTrainer(model=mock_model, config=mock_config, logger=mock_logger)

    def test_init_state(self, mock_trainer):
        """
        Tests if the trainer correctly initializes internal paths and variables
        using the mock_config fixture.
        """
        assert isinstance(mock_trainer._tfrecord_dir, Path)
        assert mock_trainer._tfrecord_dir.is_absolute()
        assert mock_trainer._model_depth == 1

    def test_prepare_datasets_logic(self, mock_trainer):
        """
        Tests the split and shuffle logic using mock filenames.
        Verifies that the train/validation split respects the default 0.8 ratio.
        """

        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_trainer._logger

            mock_dataset_class = stack.enter_context(
                patch("src.projects.lumbar_spine.model_trainer.LumbarDicomTFRecordDataset")
            )
            mock_glob = stack.enter_context(patch("tensorflow.io.gfile.glob"))

            # Simulate 10 TFRecord files found in the directory
            mock_glob.return_value = [f"study_{i}.tfrecord" for i in range(10)]

            # Mock the dataset generator to return a dummy TF dataset
            mock_ds_instance = mock_dataset_class.return_value
            mock_ds_instance.generate_tfrecord_dataset.return_value = (
                MagicMock(spec=tf.data.Dataset)
            )

            # Execute the split (default ratio 0.8)
            mock_trainer.prepare_training_and_validation_datasets()

            # Check proportions: 8 for training, 2 for validation
            assert mock_trainer._nb_train == 8
            assert mock_trainer._nb_val == 2

    def test_prepare_datasets_io_error(self, mock_trainer):
        """
        Tests that prepare_training_and_validation_datasets properly logs and
        re-raises exceptions if file discovery fails (e.g., IO issues).
        """

        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_trainer._logger

            mock_glob = stack.enter_context(
                patch("src.projects.lumbar_spine.model_trainer.tf.io.gfile.glob")
            )

            # Simulate a filesystem access error
            mock_glob.side_effect = OSError("Filesystem not reachable")

            with pytest.raises(OSError, match="Filesystem not reachable"):
                mock_trainer.prepare_training_and_validation_datasets()

    def test_prepare_datasets_logic_failure(self, mock_trainer):
        """
        Tests that an error during LumbarDicomTFRecordDataset instantiation
        is caught and logged, while verifying preliminary monitoring steps.
        """

        with ExitStack() as stack:
            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(
                patch('src.core.utils.logger.get_current_logger')
            )
            mock_get_log.return_value = mock_trainer._logger

            # Simulate a failure during dataset object creation
            mock_ds_class = stack.enter_context(
                patch("src.projects.lumbar_spine.model_trainer.LumbarDicomTFRecordDataset")
            )
            mock_ds_class.side_effect = Exception("Dataset mapping configuration error")

            # Mock memory logging to verify it's called before the crash
            mock_mem = stack.enter_context(
                patch("src.projects.lumbar_spine.model_trainer.log_memory_usage")
            )

            with pytest.raises(Exception, match="Dataset mapping configuration error"):
                mock_trainer.prepare_training_and_validation_datasets()

            # VERIFICATION:
            # 1. Ensure the first log_memory_usage call was made before the exception
            mock_mem.assert_called_once()

            # 2. Optionally verify the stage name passed to the monitor
            args, kwargs = mock_mem.call_args
            assert kwargs['stage_name'] == "Before Dataset creation"

    def test_train_with_callbacks_execution(self, mock_trainer):
        """
        Ensures that model.fit() is called with the correct calculated steps
        and the expected number of callbacks.
        """

        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_trainer._logger

            # Setup pre-requisites
            mock_trainer._nb_train = 20  # 20 samples / batch_size 2 = 10 steps
            mock_trainer._nb_val = 6     # 6 samples / batch_size 2 = 3 steps
            mock_trainer._train_dataset = MagicMock(spec=tf.data.Dataset)
            mock_trainer._validation_dataset = MagicMock(spec=tf.data.Dataset)

            # Mock the fit history return
            mock_history = MagicMock()
            mock_history.history = {'loss': [0.5], 'val_loss': [0.4], 'accuracy': [0.8]}
            mock_trainer._model.fit.return_value = mock_history

            # Execute training logic
            history = mock_trainer._train_with_callbacks()

            # Verify fit parameters
            mock_trainer._model.fit.assert_called_once()
            _, kwargs = mock_trainer._model.fit.call_args
            assert kwargs['steps_per_epoch'] == 10
            assert kwargs['validation_steps'] == 3

            # Check that multiple callbacks are present (Checkpoint, EarlyStopping, etc.)
            assert len(kwargs['callbacks']) > 0

            # Ensure the method returns the history object from the model fit
            assert history == mock_history

            # Validate that the returned history object contains expected training metrics
            assert 'loss' in history.history

    def test_train_model_success_flow(self, mock_trainer, mock_logger):
        """
        Tests the high-level train_model method to ensure it orchestrates
        the training and logs success.
        """

        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_trainer._logger

            # Mock the internal training call to return success
            mock_trainer._train_with_callbacks = MagicMock(return_value=MagicMock())

            mock_trainer.train_model()

            # Check if the success message was logged via mock_logger (using caplog indirectly)
            # We check the internal model.fit was called through the mocked internal method
            mock_trainer._train_with_callbacks.assert_called_once()

    def test_train_model_failure_logging(self, mock_trainer):
        """
        Tests that an exception during training is properly captured and re-raised.
        """

        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_trainer._logger

            # Simulate a critical failure (e.g., out of memory or file error)
            mock_trainer._train_with_callbacks = (
                MagicMock(side_effect=RuntimeError("Training failed"))
            )

            with pytest.raises(RuntimeError, match="Training failed"):
                mock_trainer.train_model()

    def test_train_with_callbacks_fit_exception(self, mock_trainer):
        """
        Tests that if model.fit fails (e.g., GPU Error), the exception is
        propagated to allow higher-level handling.
        """

        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_trainer._logger

            # Setup minimal state
            mock_trainer._nb_train = 4
            mock_trainer._nb_val = 2
            mock_trainer._train_dataset = MagicMock()
            mock_trainer._validation_dataset = MagicMock()

            # Simulate a Keras training failure (e.g., NaN loss or hardware failure)
            mock_trainer._model.fit.side_effect = RuntimeError("Loss is NaN, stopping training.")

            with pytest.raises(RuntimeError, match="Loss is NaN"):
                mock_trainer._train_with_callbacks()

    def test_invalid_config_raises_error(self, mock_trainer):
        """
        Checks that the trainer raises a ValueError if the batch_size
        is misconfigured as zero or negative.
        """

        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_trainer._logger

            mock_trainer._config["training"]["batch_size"] = 0
            mock_trainer._nb_train = 10

            error_msg = "the setting variable 'training -> batch_size' is missing or invalid."
            with pytest.raises(ValueError, match=error_msg):
                mock_trainer._train_with_callbacks()

    def test_prepare_datasets_empty_directory(self, mock_trainer):
        """
        Tests the behavior when no TFRecord files are found in the directory.
        The trainer should handle empty lists without crashing but might result
        in zero samples.
        """

        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_trainer._logger

            mock_glob = stack.enter_context(patch("tensorflow.io.gfile.glob"))

            # Simulate an empty folder
            mock_glob.return_value = []

            # In this case, split_idx will be 0
            mock_trainer.prepare_training_and_validation_datasets()

            assert mock_trainer._nb_train == 0
            assert mock_trainer._nb_val == 0

    def test_train_with_callbacks_zero_samples_error(self, mock_trainer):
        """
        Ensures that if the dataset is empty, the steps calculation logic
        still works (max(1, ...)) or handles it gracefully before fitting.
        """

        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_trainer._logger

            # Setup pre-requisites with zero samples
            mock_trainer._nb_train = 0
            mock_trainer._nb_val = 0
            mock_trainer._train_dataset = MagicMock(spec=tf.data.Dataset)
            mock_trainer._validation_dataset = MagicMock(spec=tf.data.Dataset)

            mock_history = MagicMock()
            mock_history.history = {'loss': [0.0], 'val_loss': [0.0]}
            mock_trainer._model.fit.return_value = mock_history

            # Execution
            mock_trainer._train_with_callbacks()

            # Verify that steps_per_epoch is at least 1 to avoid TensorFlow errors
            _, kwargs = mock_trainer._model.fit.call_args
            assert kwargs['steps_per_epoch'] == 1
            assert kwargs['validation_steps'] == 1

    def test_log_memory_usage_exception_safety(self, mock_trainer):
        """
        Tests that log_memory_usage does not crash the main process
        even if the monitoring tools fail.
        """

        with ExitStack() as stack:

            # Patch the global logger getter used by @log_method
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_trainer._logger

            # Simulate a failure in psutil (e.g., permission denied)
            mock_vmem = stack.enter_context(
                patch("src.projects.lumbar_spine.model_trainer.psutil.virtual_memory")
            )
            mock_vmem.side_effect = Exception("OS Permission Denied")

            # This should not raise an exception because of the internal try/except block
            # It should simply print the error message and continue.
            try:
                log_memory_usage(process=mock_trainer._process, stage_name="Crash Test")
            except Exception as e:
                pytest.fail(f"log_memory_usage raised {e} instead of capturing it.")

    def test_output_directories_creation(self, mock_trainer):
        """
        Tests that the trainer ensures output and checkpoint directories
        exist before starting the training process.
        """

        with ExitStack() as stack:
            # Patch the logger injection
            mock_get_log = stack.enter_context(patch('src.core.utils.logger.get_current_logger'))
            mock_get_log.return_value = mock_trainer._logger

            # Mock fit to avoid actual training
            mock_trainer._model.fit = MagicMock()

            # Setup paths that don't exist yet
            new_output_dir = mock_trainer._config["root_dir"] / "new_output_folder"
            mock_trainer._config["paths"]["output"] = new_output_dir

            # Ensure the directory does not exist before call
            if new_output_dir.exists():
                shutil.rmtree(new_output_dir)

            # Pre-requisites for _train_with_callbacks
            mock_trainer._nb_train = 2
            mock_trainer._nb_val = 1
            mock_trainer._train_dataset = MagicMock()
            mock_trainer._validation_dataset = MagicMock()

            # Execute
            mock_trainer._train_with_callbacks()

            # Verification: The directory should have been created
            assert new_output_dir.exists()
            assert new_output_dir.is_dir()
