# coding: utf-8

from unittest.mock import patch, MagicMock, PropertyMock
import pytest
import tensorflow as tf
import sys
import os
from contextlib import ExitStack

# Use local import to ensure coverage tracker is active
import src.RSNA_2024_Lumbar_Spine_Degenerative_Classification as src_module


class TestRSNALumbarClassification:

    # --------------------------------------------------------------------------
    # TESTS FOR get_or_build_model
    # --------------------------------------------------------------------------

    def test_get_or_build_model_new_instance(self, mock_config):
        """
        Test get_or_build_model builds a new model when no checkpoint exists.
        """

        with ExitStack() as stack:
            stack.enter_context(patch("tensorflow.keras.metrics.MeanAbsoluteError", MagicMock()))
            mock_metric_class = stack.enter_context(patch.object(src_module, "RSNAKaggleMetric"))
            mock_metric_instance = MagicMock()
            mock_metric_class.return_value = mock_metric_instance

            mock_intra_op_parallelism_threads = (
                "tensorflow.python.eager.context.Context."
                "intra_op_parallelism_threads"
            )
            stack.enter_context(
                patch(
                    mock_intra_op_parallelism_threads,
                    new_callable=PropertyMock,
                    return_value=1
                )
            )

            mock_inter_op_parallelism_threads = (
                "tensorflow.python.eager.context.Context."
                "inter_op_parallelism_threads"
            )
            stack.enter_context(
                patch(
                    mock_inter_op_parallelism_threads,
                    new_callable=PropertyMock,
                    return_value=1
                )
            )

            mock_path = stack.enter_context(patch.object(src_module, "Path"))

            # Simulate that the checkpoint file does not exist
            mock_path.return_value.resolve.return_value.is_file.return_value = False

            local_mock_logger = MagicMock()
            mock_model = MagicMock(spec=tf.keras.Model)

            mock_factory = stack.enter_context(patch.object(src_module, "ModelFactory"))

            # Mocking factory return
            mock_factory.return_value.build_multi_series_model.return_value = mock_model

            model = src_module.get_or_build_model(
                depth=60,
                config=mock_config,
                logger=local_mock_logger
            )

            assert model == mock_model
            mock_model.compile.assert_called_once()

            error_msg = "No existing model found. Building a new model from factory."
            local_mock_logger.info.assert_any_call(error_msg)

    def test_get_or_build_model_invalid_config(self, mock_logger):
        """
        Test get_or_build_model raises ValueError if config is incomplete.
        """

        incomplete_config = {"checkpoint_path": "some/path"}  # Missing learning_rate, etc.

        with pytest.raises(ValueError):
            src_module.get_or_build_model(
                depth=60,
                config=incomplete_config,
                logger=mock_logger
            )

    def test_get_or_build_model_corrupted_file(self, mock_config):
        """
        Test fallback to factory when the model file is corrupted.
        """

        with ExitStack() as stack:
            # Simulate that file exists but loading raises an error
            mock_path = stack.enter_context(patch.object(src_module, "Path"))
            mock_path.return_value.resolve.return_value.is_file.return_value = True

            mock_tf_load = stack.enter_context(patch.object(src_module, "load_model"))
            mock_tf_load.side_effect = Exception("Corrupted model file")

            local_mock_logger = MagicMock()

            # Ensure the factory still produces a model
            mock_model = MagicMock(spec=tf.keras.Model)
            mock_factory = stack.enter_context(patch.object(src_module, "ModelFactory"))
            mock_factory.return_value.build_multi_series_model.return_value = mock_model

            model = src_module.get_or_build_model(
                depth=60,
                config=mock_config,
                logger=local_mock_logger
            )

            # Verify that we logged a warning and still got a model
            local_mock_logger.warning.assert_any_call(
                "Failed to load existing model: Corrupted model file. Trying to restore weights."
            )
            assert model is not None

    # --------------------------------------------------------------------------
    # TESTS FOR main flow
    # --------------------------------------------------------------------------

    # Mock tensorflow to avoid hardware config errors
    def test_main_success_flow(self, mock_config, mock_logger):
        """
        Test the full execution flow of main().
        """
        with ExitStack() as stack:
            mock_clean = stack.enter_context(patch.object(src_module, "clean_old_logs"))
            mock_build = stack.enter_context(patch.object(src_module, "get_or_build_model"))
            mock_trainer = stack.enter_context(patch.object(src_module, "ModelTrainer"))

            mock_tee = MagicMock()
            mock_tee_class = stack.enter_context(patch.object(src_module, "SystemStreamTee"))
            mock_tee_class.return_value = mock_tee

            # Setup mock returns
            mock_cfg_loader = stack.enter_context(
                patch.object(
                    src_module,
                    "ConfigLoader"
                )
            )
            mock_loader_instance = mock_cfg_loader.return_value
            mock_loader_instance.get.return_value = mock_config
            mock_loader_instance.get_value.side_effect = (
                lambda key, default=None: mock_config.get(key, default)
            )
            mock_loader_instance.set_value.return_value = None

            mock_setup_log = stack.enter_context(patch.object(src_module, "setup_logger"))
            mock_setup_log.return_value.__enter__.return_value = mock_logger

            tfrecord_file_mgr_path = (
                "src.projects.lumbar_spine.tfrecord_files_manager.TFRecordFilesManager"
            )
            mock_tf_mgr = stack.enter_context(patch(tfrecord_file_mgr_path))
            mock_tf_mgr.return_value.get_max_series_depth.return_value = 60

            mock_tf = stack.enter_context(patch.object(src_module, "tf"))

            # Execute main - the mocked 'tf' will now prevent the RuntimeError
            src_module.main()

            # Verify tf config was called (optional)
            system_cfg = mock_config.get("system")
            if system_cfg is None:
                mock_logger.error("mock_config -> system is not defined")

            nb_cores = mock_config["system"]["nb_cores"]
            mock_logger.info(f"nb_cores = {nb_cores}")

            set_threads_mock = mock_tf.config.threading.set_intra_op_parallelism_threads

            if nb_cores > 1:
                set_threads_mock.assert_called_once_with(nb_cores)
            else:
                set_threads_mock.assert_not_called()

            # Verify sequences
            expected_log_path = mock_config["paths"]["log_mirror"]
            mock_tee_class.assert_called_once_with(expected_log_path)
            assert sys.stdout == mock_tee
            assert sys.stderr == mock_tee
            mock_build.assert_called_once()
            mock_trainer.return_value.prepare_training_and_validation_datasets.assert_called_once()
            mock_trainer.return_value.train_model.assert_called_once()
            mock_clean.assert_called_once()

    # --------------------------------------------------------------------------
    # TESTS FOR signals
    # --------------------------------------------------------------------------

    def test_handle_interrupt(self, mock_logger, caplog):
        """
        Test that SIGINT handler logs message and exits.
        """
        with ExitStack() as stack:
            mock_exit = stack.enter_context(patch.object(src_module.sys, "exit"))
            mock_get_logger = stack.enter_context(patch.object(src_module, "get_current_logger"))
            mock_get_logger.return_value = mock_logger

            # Invoke the handler
            src_module.handle_interrupt(None, None)

            # Verify that sys.exit is called
            mock_exit.assert_called_once_with(0)

            # Verify message capture in caplog.
            error_msg = "Interruption detected (Ctrl+C). Exiting gracefully..."
            assert error_msg in caplog.text

    @pytest.mark.parametrize(
        "is_kaggle, expected_target, file_exists",
        [
            # Scenario: Kaggle environment, file exists
            (True, "lumbar_spine_config_kaggle.yaml", True),

            # Scenario: Windows environment, file exists
            (False, "lumbar_spine_config_windows.yaml", True),

            # Scenario: File missing (should skip)
            (False, "lumbar_spine_config_windows.yaml", False)
        ]
    )
    def test_setup_config_symlink(self, is_kaggle, expected_target, file_exists, tmp_path):
        """
        Test the automated symlink creation for different environments and file availability.
        """
        mock_dir_path = tmp_path / "mock/absolute/path"

        # Define the simulated environment variables
        env_vars = {'KAGGLE_KERNEL_RUN_TYPE': 'interactive'} if is_kaggle else {}

        with ExitStack() as stack:
            # Mock environment variables to simulate Kaggle vs Local
            stack.enter_context(patch.dict(os.environ, env_vars, clear=True))

            # Mock the Path class within the source module to avoid disk I/O
            mock_path_class = stack.enter_context(patch(f"{src_module.__name__}.Path"))

            # Setup the mock directory and its resolved path
            mock_dir = MagicMock()
            mock_path_class.return_value.resolve.return_value = mock_dir

            # Setup mocks for the main config link and the target source file
            mock_main_config = MagicMock()
            mock_target_file = MagicMock()

            # Mock the division operator (/) to return specific mocks in order
            # First call: config_dir / "lumbar_spine_config.yaml"
            # Second call: config_dir / target_name
            mock_dir.__truediv__.side_effect = [mock_main_config, mock_target_file]

            # Simulate whether the target YAML file exists on 'disk'
            mock_target_file.exists.return_value = file_exists

            # Execute the function under test
            src_module.setup_config_symlink(mock_dir_path)

            # --- Assertions ---

            if not file_exists:
                # If source file is missing, symlink creation must be skipped
                mock_main_config.symlink_to.assert_not_called()
            else:
                # 1. Verify that the correct target filename was requested based on environment
                mock_dir.__truediv__.assert_any_call(expected_target)

                # 2. Verify that any existing file or link was removed before creation
                # (Covers both .is_symlink() and .exists() checks in the source)
                assert mock_main_config.unlink.called

                # 3. Verify the symlink was created pointing to the correct target mock
                mock_main_config.symlink_to.assert_called_once_with(mock_target_file)

# if __name__ == "__main__":
#    unittest.main()
