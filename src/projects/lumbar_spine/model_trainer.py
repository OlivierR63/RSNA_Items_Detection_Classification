# coding: utf-8

import os
import gc
import math
import psutil
import logging
import sys
import random
from pathlib import Path
from typing import Any
from tf_keras.callbacks import ReduceLROnPlateau

import tensorflow as tf
import tf_keras

from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
from src.core.utils.logger import get_current_logger, log_method
from src.core.callbacks.log_training_callback import LogTrainingCallback
from src.core.callbacks.system_resource_monitor_callback import SystemResourceMonitorCallback
from src.core.callbacks.dynamic_loss_balancer_callback import DynamicLossBalancerCallback
from src.core.callbacks.epoch_sync_callback import EpochSyncCallback
from src.core.callbacks.print_epoch_callback import PrintEpochCallback
from src.core.callbacks.memory_cleanup_callback import MemoryCleanupCallback
from src.core.callbacks.kaggle_dataset_checkpoint_sync_callback import (
    KaggleDatasetCheckpointSyncCallback as KaggleSync
)
from src.core.callbacks.robust_model_checkpoint_callback import RobustModelCheckpointCallback
from src.core.utils.monitoring_utils import log_memory_usage
from src.config.config_loader import ConfigLoader


class ModelTrainer:
    """
    Orchestrates the training process for the RSNA Lumbar Spine model.

    This class handles the end-to-end training pipeline, including dataset preparation
    (splitting TFRecord files), callback configuration (checkpointing, TensorBoard),
    and the execution of the training loop. It integrates structured logging and
    proactive memory monitoring to ensure stability when processing large
    3D medical imaging volumes.
    """

    # Single source of truth for filenames
    CHECKPOINT_FILENAME = "model_checkpoint.keras"
    BEST_MODEL_FILENAME = "best_model.keras"

    def __init__(
        self,
        model: tf_keras.Model,
        logger: logging.Logger | None = None,
        model_depth: int = 1,
        initial_epoch: int = 0,
        loss_weight_var: tf.Variable = None
    ) -> None:

        """
        Initializes the ModelTrainer with the model, configuration setup, and tracking tools.

        Args:
            - model (tf_keras.Model): The compiled Keras model to be trained.
            - logger (logging.Logger | None): Logger instance for process tracking.
                Defaults to the current system logger if not provided.
            - model_depth (int): The depth of the 3D input volume, used for
                dataset dimension configuration.
            - loss_weight_var (tf.Variable | None): A shared TensorFlow variable
                used during model compilation to track and dynamically adjust the
                coordinate regression loss weight. Defaults to None, which triggers
                a local fallback initialization.
        """

        # Force a clean break between lines
        self._model = model
        self._config = ConfigLoader().get()
        self._logger = logger or get_current_logger()
        self._process = psutil.Process(os.getpid())

        tfrecord_paths = self._config["paths"]["tfrecord"]
        if isinstance(tfrecord_paths, dict):
            self._tfrecord_read_dir = Path(tfrecord_paths["read_only_dir"]).resolve()
            self._tfrecord_write_dir = Path(tfrecord_paths["read_write_dir"]).resolve()
        else:
            self._tfrecord_read_dir = None
            self._tfrecord_write_dir = Path(tfrecord_paths).resolve()

        # Test line: if the error is here, it's a scope issue
        self._model_depth = model_depth

        # Use the shared variable if provided, else create a local one (fallback)
        if loss_weight_var is not None:
            self._loss_weight_var = loss_weight_var
        else:
            self._loss_weight_var = tf.Variable(
                self._config['compilation']['loss_weights']['location_output'],
                dtype=tf.float32,
                trainable=False
            )

        self._nb_train = None
        self._nb_val = None
        self._train_dataset = None
        self._validation_dataset = None
        self._dataset_manager = None
        self._current_initial_epoch = self._calculate_initial_epoch()

    @log_method()
    def train_model(
        self,
        *,
        logger: logging.Logger | None
    ) -> None:

        """
        High-level entry point to start the model training process.

        This method injects the logger, handles the training execution flow,
        and ensures that any critical failure during the training loop is
        properly logged with a full stack trace before being raised.

        Args:
            logger (logging.Logger | None): Overriding logger instance.

        Raises:
            Exception: Re-raises any exception encountered during the
                training lifecycle for external handling.
        """

        logger = logger or self._logger

        logger.info(
            "Loading 3D model...",
            extra={
                "action": "load_model",
                "model_type": self._config['models']['head_3d']['type']
            }
        )

        try:
            # Train the model
            _ = self._train_with_callbacks()

            logger.info(
                f"Model saved to {self._config['paths']['checkpoint']}/model_checkpoint.keras",
                extra={
                    "status": "success",
                    "model_path": str(
                        Path(self._config["paths"]["checkpoint"])/"model_checkpoint.keras"
                    )
                }
            )

        except Exception as e:
            log_memory_usage(
                stage_name="Pre-fit memory state",
                process=self._process,
                logger=self._logger
            )
            logger.critical(
                f"Error in train_model: {str(e)}",
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise

    @log_method()
    def prepare_training_and_validation_datasets(
        self,
        logger: logging.Logger | None = None
    ) -> None:

        """
        Builds and splits the TensorFlow dataset pipelines for training and validation.

        This method identifies all available TFRecord files, performs a randomized shuffle
        at the file level to ensure data independence, and splits them according to the
        configured 'train_split_ratio'. It then initializes the specialized
        `LumbarDicomTFRecordDataset` to create optimized input pipelines.

        The resulting datasets and their respective sample counts are stored as internal
        attributes of the class instance.

        Args:
            logger (logging.Logger | None): Logger instance for tracking dataset
                creation and memory usage. Defaults to self._logger.

        Returns:
            None: The results are assigned to self._train_dataset, self._nb_train,
                  self._validation_dataset, and self._nb_val.

        Raises:
            Exception: Propagates any error encountered during file discovery,
                shuffling, or pipeline generation.
        """

        logger = logger or self._logger

        training_cfg = self._config['training']

        logger.info("Setting up TensorFlow dataset pipeline...",
                    extra={"action": "create_dataset"})

        try:
            batch_size = training_cfg["batch_size"]

            log_memory_usage(
                stage_name="Before Dataset creation",
                process=self._process,
                logger=self._logger
            )

            self._dataset_manager = LumbarDicomTFRecordDataset(
                self._logger,
                self._model_depth
            )

            # 1. List files and shuffle them from both directories
            read_files = []
            if self._tfrecord_read_dir is not None:
                read_files = tf.io.gfile.glob(str(self._tfrecord_read_dir / "*.tfrecord"))

            write_files = []
            if self._tfrecord_write_dir is not None:
                write_files = tf.io.gfile.glob(str(self._tfrecord_write_dir / "*.tfrecord"))

            # Deduplicate by filename (to prevent double counting if the same patient is in both)
            file_map = {}
            for f in read_files:
                file_map[Path(f).name] = f
            for f in write_files:
                file_map[Path(f).name] = f
            all_tfrecord_files = list(file_map.values())

            # 2. Shuffle the files
            random.seed(42)
            random.shuffle(all_tfrecord_files)

            # 3. Calculate total count and define split index for train/validation
            nb_tfrecord_files = len(all_tfrecord_files)
            train_ratio = training_cfg['train_split_ratio']
            split_idx = int(nb_tfrecord_files * train_ratio)

            # 4. Split the train and validation datasets
            train_list = all_tfrecord_files[:split_idx]
            val_list = all_tfrecord_files[split_idx:]

            # 5. The TensorFlow world starts here
            self._nb_train = len(train_list)
            self._train_dataset = self._dataset_manager.generate_tfrecord_dataset(
                train_list,
                batch_size=batch_size,
                is_training=True
            )
            self._nb_val = len(val_list)
            self._validation_dataset = self._dataset_manager.generate_tfrecord_dataset(
                val_list,
                batch_size=batch_size,
                is_training=False
            )

            logger.info("Training and validation dataset created successfully.",
                        extra={"status": "success"})

            return

        except tf.errors.OpError as e:
            # Catch any error occurring during TensorFlow graph execution (OOM, File IO, etc.)
            critical_msg = f"Tensorflow Runtime error: {str(e)}"
            self._logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failed", "error": str(e)}
            )
            sys.exit(1)

        except Exception as e:
            # Create a "dynamic" tag for precise diagnostic:
            error_type = type(e).__name__
            memory_info_msg = f"Post-crash resource (CPU / RAM) snapshot: {error_type}"

            # Call the log function with that tag
            log_memory_usage(
                stage_name=memory_info_msg,
                process=self._process,
                logger=self._logger
            )

            # Normally log the error
            logger.critical(
                f"Error creating dataset: {str(e)}",
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise

    @log_method()
    def _train_with_callbacks(
        self,
        *,
        logger: logging.Logger | None = None
    ) -> Any:

        """
        High-level entry point to start the model training process.

        This method injects the logger, handles the training execution flow,
        and ensures that any critical failure during the training loop is
        properly logged with a full stack trace before being raised.

        Args:
            logger (logging.Logger | None): Overriding logger instance.

        Raises:
            Exception: Re-raises any exception encountered during the
                training lifecycle for external handling.
        """

        logger = logger or self._logger

        paths_cfg = self._config["paths"]
        output_dir = paths_cfg['output']

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Paths for checkpointing
        # 'best_model.keras' stores the weights with the lowest validation loss
        # 'last_model.keras' is updated every epoch to allow training resumption
        checkpoint_dir_path = Path(paths_cfg["checkpoint"]).resolve()

        # Ensure the directory tree exists before attempting to write the file
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

        # Optional permission check (very useful on Kaggle)
        if not os.access(checkpoint_dir_path, os.W_OK):
            error_msg = (
                f"The directory {checkpoint_dir_path} is not writable. Please check permissions."
            )
            raise PermissionError(error_msg)

        last_path = checkpoint_dir_path / self.CHECKPOINT_FILENAME
        best_path = checkpoint_dir_path / self.BEST_MODEL_FILENAME

        # Initialize your custom hardware and metrics monitor
        system_cfg = self._config["system"]
        memory_threshold_percent_str = system_cfg["memory_threshold_percent"]

        memory_threshold_percent = float(memory_threshold_percent_str)

        monitor_callback = SystemResourceMonitorCallback(
            memory_threshold_percent=float(memory_threshold_percent)
        )

        # Simplified print callback for batch monitoring
        print_callback = PrintEpochCallback(logger=logger, batch_log_frequency=1)

        # Create a callback to sync the dataset epoch with the trainer epoch
        sync_epoch_callback = EpochSyncCallback(
            trainer=self,
            initial_offset=self._current_initial_epoch
        )

        # Calculate steps_per_epoch and validation_step, since the dataset
        # uses .repeat()  (infinite stream)
        training_cfg = self._config['training']
        batch_size = training_cfg['batch_size']

        steps_per_epoch = max(1, math.ceil(self._nb_train / batch_size))
        validation_steps = max(1, math.ceil(self._nb_val / batch_size))

        logger.info(
            f"Starting model training: {steps_per_epoch} steps/epoch, "
            f"{validation_steps} validation steps."
        )

        training_progress_callback = LogTrainingCallback(logger, validation_steps)

        momentum = self._config['training']['loss_balancer']['momentum']
        min_weight = self._config['training']['loss_balancer']['min_weight']
        max_weight = self._config['training']['loss_balancer']['max_weight']

        dynamic_loss_balancer_callback = DynamicLossBalancerCallback(
            weight_variable=self._loss_weight_var,
            momentum=momentum,
            min_weight=min_weight,
            max_weight=max_weight,
            logger=logger
        )

        kaggle_sync_callback = KaggleSync(
            checkpoint_dir=str(checkpoint_dir_path),
            checkpoint_filename=self.CHECKPOINT_FILENAME
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_severity_output_rsna_main_score',
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )

        ram_cleaner_callback = MemoryCleanupCallback(
            run_gc=True,
            clear_session=False,
            logger=self._logger
        )

        best_checkpoint_callback = RobustModelCheckpointCallback(
            logger=logger,
            filepath=str(best_path),
            save_weights_only=True,
            save_best_only=False,
            monitor="val_severity_output_rsna_main_score",
            monitor_mode="min",
            save_freq='epoch',
            verbose=0,  # We handle logging manually in RobustModelCheckpointCallback
        )

        last_checkpoint_callback = RobustModelCheckpointCallback(
            logger=logger,
            filepath=str(last_path),
            save_best_only=False,
            save_weights_only=True,
            save_freq='epoch',
            verbose=0
        )

        callbacks_cfg = self._config['callbacks']
        patience_cfg = callbacks_cfg['patience']

        callbacks = [
            sync_epoch_callback,
            monitor_callback,
            # cpu_temperature_callback,
            training_progress_callback,
            print_callback,
            lr_scheduler,
            best_checkpoint_callback,
            last_checkpoint_callback,
            ram_cleaner_callback,
            kaggle_sync_callback,
            dynamic_loss_balancer_callback,

            # Stop training if validation loss plateaus.
            tf_keras.callbacks.EarlyStopping(
                monitor="val_severity_output_rsna_main_score",
                patience=patience_cfg,
                restore_best_weights=True,
                verbose=1
            ),

            # Export log for visualization in tensorboard
            tf_keras.callbacks.TensorBoard(
                log_dir=str((Path(output_dir) / "tensorboard_logs").resolve()),
                histogram_freq=1,
                profile_batch=(0)
            )
        ]

        logger.info(
            "Training callbacks configured.",
            extra={"callbacks": [callback.__class__.__name__ for callback in callbacks]}
        )

        epochs_cfg = training_cfg["epochs"]

        # Clear session and collect garbage to maximize available VRAM before fitting
        del training_cfg
        del callbacks_cfg
        gc.collect()

        log_memory_usage(
            stage_name="Pre-fit memory state",
            process=self._process,
            logger=self._logger
        )

        try:
            # Execute the training loop
            self._logger.info(f"Resuming training from epoch {self._current_initial_epoch}")

            history = self._model.fit(
                self._train_dataset,
                epochs=epochs_cfg,
                steps_per_epoch=steps_per_epoch,
                validation_data=self._validation_dataset,
                validation_steps=validation_steps,
                initial_epoch=self._current_initial_epoch,
                callbacks=callbacks,
                verbose=0
            )

            # Calculation of the average accuracy for the logger
            acc_keys = [k for k in history.history.keys() if k.endswith('_accuracy')]
            if acc_keys:
                final_acc = sum([history.history[k][-1] for k in acc_keys]) / len(acc_keys)
            else:
                final_acc = 0

            logger.info(
                "Model training completed successfully.",
                extra={
                        "final_loss": history.history['loss'][-1],
                        "final_accuracy": final_acc
                })
            return history

        except Exception as e:
            critical_msg = f"Error during training: {str(e)}"
            logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise

    def _set_epoch(self, epoch: int) -> tf.Tensor:
        return self._dataset_manager._current_epoch_var.assign(epoch)

    def _calculate_initial_epoch(self) -> int:
        """
        Determines the initial epoch index to resume training.

        Locates the most recent training log file in the output directory
        and parses it to identify the last completed epoch.

        Returns:
            int: The next epoch index to be trained (last completed + 1).
                 Defaults to 0 if no log file is found or if parsing fails.
        """
        self._logger.info("Calculating initial epoch for training resumption...")
        log_dir = Path(self._config['paths']['output']) / "logs"
        use_json = self._config['logging'].get('use_json', False)
        pattern = "train_*.json" if use_json else "train_*.log"

        latest_log = self._get_latest_file(log_dir, pattern)
        self._logger.info(f"Latest log file identified: {latest_log}")
        if not latest_log:
            self._logger.info("No existing log file found.")
            return 0

        try:
            return self._extract_epoch_from_file(latest_log, use_json)
        except Exception as e:
            self._logger.warning(f"Failed to read latest log {latest_log}: {e}", exc_info=True)
            return 0

    def _get_latest_file(self, directory: Path, pattern: str) -> Path | None:
        """
        Identifies the most recently modified file matching a specific pattern.

        Args:
            directory (Path): The directory path to search in.
            pattern (str): The glob pattern (e.g., "*.log").

        Returns:
            Path | None: The Path object of the most recent file, or None if no
                         files match the pattern or the directory does not exist.
        """
        files = list(directory.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)

    def _extract_epoch_from_file(self, file_path: Path, use_json: bool) -> int:
        """
        Parses a log file to extract the epoch for the next training session.

        Reads the log file in either JSON or plain text format, identifies the
        last completed epoch, and returns the incremented value to define the
        starting epoch for resumed training.

        Args:
            file_path (Path): Path to the log file to parse.
            use_json (bool): If True, parses the file as JSON; otherwise,
                parses as plain text line-by-line.

        Returns:
            int: The epoch number to resume training from, or 0 if no valid
                 epoch data is found.
        """
        if use_json:
            import json
            with open(file_path, 'r') as f:
                logs = json.load(f)
                return logs[-1].get('epoch', 0) + 1 if logs else 0

        # Text parsing
        next_epoch = 0
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("Epoch"):
                    next_epoch = max(next_epoch, int(line.split()[1]) + 1)
        return next_epoch
