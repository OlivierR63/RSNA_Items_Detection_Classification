# coding: utf-8

import os
import gc
import psutil
import logging
import tensorflow as tf

from pathlib import Path
from typing import Optional, Any

from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
from src.core.utils.logger import get_current_logger, log_method
from src.core.utils.log_training_callbacks import LogTrainingCallbacks
from src.core.utils.system_resource_monitor_callbacks import SystemResourceMonitorCallbacks
from keras.callbacks import LambdaCallback, ProgbarLogger


def log_memory_usage(*, process, stage_name=""):
    """
    Logs the current RAM usage of the Python process and the overall system.
    Useful for monitoring memory leaks during 3D medical imaging pipelines.
    """
    try:
        # Get the current process ID and its memory information
        mem_info = process.memory_info()

        # Convert Resident Set Size (RSS) to Megabytes (MB)
        # RSS represents the portion of memory occupied by a process that is held in RAM
        mem_mb = mem_info.rss / (1024 * 1024)

        # Retrieve the total percentage of system RAM currently in use
        total_ram_percent = psutil.virtual_memory().percent

        # Output the memory snapshot to the console
        print(f">>> [RAM] {stage_name} | Process: {mem_mb:.2f} MB | System: {total_ram_percent}%")

    except Exception as e:
        print(f"Memory monitoring failed: {e}")


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
        model: tf.keras.Model,
        config: dict,
        logger: Optional[logging.Logger] = None,
        model_depth: int = 1
    ) -> None:

        """
        Initializes the ModelTrainer with the model, configuration, and tracking tools.

        Args:
            - model (tf.keras.Model): The compiled Keras model to be trained.
            - config (dict): Configuration dictionary containing hyperparameters,
                directory paths, and training settings.
            - logger (Optional[logging.Logger]): Logger instance for process tracking.
                Defaults to the current system logger if not provided.
            - model_depth (int): The depth of the 3D input volume, used for
                dataset dimension configuration.
        """

        # Force a clean break between lines
        self._model = model
        self._config = config
        self._logger = logger or get_current_logger()
        self._process = psutil.Process(os.getpid())

        # Define the directory where the TFRecord files shall be stored.
        self._tfrecord_dir = Path(config["paths"]["tfrecord"]).resolve()

        # Define the pattern to match all TFRecord files in the directory.
        self._tfrecord_pattern = str(self._tfrecord_dir / "*.tfrecord")

        # Test line: if the error is here, it's a scope issue
        self._model_depth = model_depth

        self._nb_train = None
        self._nb_val = None
        self._train_dataset = None
        self._validation_dataset = None
        self._dataset_manager = None

    @log_method()
    def train_model(
        self,
        *,
        logger: Optional[logging.Logger] = None
    ) -> None:

        """
        High-level entry point to start the model training process.

        This method injects the logger, handles the training execution flow,
        and ensures that any critical failure during the training loop is
        properly logged with a full stack trace before being raised.

        Args:
            logger (Optional[logging.Logger]): Overriding logger instance.

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
            logger.error(
                f"Error in train_model: {str(e)}",
                exc_info=True,
                extra={"status": "failed", "error": str(e)}
            )
            raise

    @log_method()
    def prepare_training_and_validation_datasets(
        self,
        logger: Optional[logging.Logger] = None
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
            logger (Optional[logging.Logger]): Logger instance for tracking dataset
                creation and memory usage. Defaults to self._logger.

        Returns:
            None: The results are assigned to self._train_dataset, self._nb_train,
                  self._validation_dataset, and self._nb_val.

        Raises:
            Exception: Propagates any error encountered during file discovery,
                shuffling, or pipeline generation.
        """

        logger = logger or self._logger

        training_cfg = self._config.get('training', None)
        if training_cfg is None:
            error_msg = (
                "Fatal error in prepare_training_and_validation_datasets: "
                "the setting variable 'training' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(error_msg)

        logger.info("Setting up TensorFlow dataset pipeline...",
                    extra={"action": "create_dataset"})

        try:
            batch_size = training_cfg.get("batch_size", None)

            if batch_size is None:
                error_msg = (
                    "Fatal error in prepare_training_and_validation_datasets: "
                    "the setting variable 'training -> batch_size' is required "
                    "but was not found. Please check your YAML file structure."
                )
                raise ValueError(error_msg)

            if batch_size <= 0:
                error_msg = (
                    "Fatal error in prepare_training_and_validation_datasets: "
                    "'batch_size' shall be strictly positive"
                )
                raise ValueError(error_msg)

            log_memory_usage(stage_name="Before Dataset creation", process=self._process)

            self._dataset_manager = LumbarDicomTFRecordDataset(
                self._config,
                self._logger,
                self._model_depth
            )

            # 1. List files and shuffle them
            all_tfrecord_files = tf.io.gfile.glob(self._tfrecord_pattern)

            # 2. Shuffle the files
            shuffled_tfrecord_list = tf.random.shuffle(all_tfrecord_files, seed=42)

            # 3. Calculate the total number of files
            nb_tfrecord_files = tf.shape(shuffled_tfrecord_list)[0]

            # 3. Define the train / validation ratio
            train_ratio = training_cfg.get('train_split_ratio', None)

            if train_ratio is None:
                error_msg = (
                    "Fatal error in prepare_training_and_validation_datasets: "
                    "the setting variable 'training -> train_split_ratio' is required "
                    "but was not found. Please check your YAML file structure."
                )
                raise ValueError(error_msg)

            # 4. Split the train and validation datasets
            split_idx = tf.cast(tf.cast(nb_tfrecord_files, tf.float32) * train_ratio, tf.int32)
            train_list = shuffled_tfrecord_list[:split_idx]
            val_list = shuffled_tfrecord_list[split_idx:]

            # 5. The TensorFlow world starts here
            self._nb_train = len(train_list.numpy())
            self._train_dataset = self._dataset_manager.generate_tfrecord_dataset(
                train_list,
                batch_size=batch_size,
                is_training=True
            )
            self._nb_val = len(val_list.numpy())
            self._validation_dataset = self._dataset_manager.generate_tfrecord_dataset(
                val_list,
                batch_size=batch_size,
                is_training=False
            )

            log_memory_usage(stage_name="After Dataset creation", process=self._process)

            logger.info("Training and validation dataset created successfully.",
                        extra={"status": "success"})

            return

        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            raise

    @log_method()
    def _train_with_callbacks(
        self,
        *,
        logger: Optional[logging.Logger] = None
    ) -> Any:

        """
        High-level entry point to start the model training process.

        This method injects the logger, handles the training execution flow,
        and ensures that any critical failure during the training loop is
        properly logged with a full stack trace before being raised.

        Args:
            logger (Optional[logging.Logger]): Overriding logger instance.

        Raises:
            Exception: Re-raises any exception encountered during the
                training lifecycle for external handling.
        """

        logger = logger or self._logger

        paths_cfg = self._config["paths"]
        if paths_cfg is None:
            error_msg = (
                "Fatal error in _train_with_callbacks: "
                "the setting variable 'paths' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(error_msg)

        output_dir = paths_cfg.get('output', None)
        if output_dir is None:
            error_msg = (
                "Fatal error in _train_with_callbacks: "
                "the setting variable 'paths -> output' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(error_msg)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Paths for checkpointing
        # 'best_model.keras' stores the weights with the lowest validation loss
        # 'last_model.keras' is updated every epoch to allow training resumption
        checkpoint_dir = Path(paths_cfg.get("checkpoint", None))
        if checkpoint_dir is None:
            error_msg = (
                "Fatal error in _train_with_callbacks: "
                "the setting variable 'paths -> checkpoint' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(error_msg)

        checkpoint_dir = checkpoint_dir.resolve()
        last_path = checkpoint_dir / self.CHECKPOINT_FILENAME
        best_path = checkpoint_dir / self.BEST_MODEL_FILENAME

        # Initialize your custom hardware and metrics monitor
        training_progress_callback = LogTrainingCallbacks(logger)
        monitor_callback = SystemResourceMonitorCallbacks(memory_threshold_percent=85.0)

        # Simplified print callback for batch monitoring
        # Since LogTrainingCallbacks already prints RAM/Time,
        # we only print metrics here every 5 steps.
        print_callback = LambdaCallback(
            on_batch_end=lambda batch, logs: print(
                f" >>> Step {int(batch)+1:04d} | Loss: {logs['loss']:.4f}"
            )  # if (int(batch)+1) % 5 == 0 else None
        )

        # Create a callback to sync the dataset epoch with the trainer epoch
        sync_epoch_callback = LambdaCallback(
            on_epoch_begin=lambda epoch,
            logs: self._dataset_manager._current_epoch_var.assign(epoch)
        )

        callbacks_cfg = self._config.get('callbacks', None)
        if callbacks_cfg is None:
            error_msg = (
                "Fatal error in _train_with_callbacks: "
                "the setting variable 'training -> callbacks' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(error_msg)

        patience_cfg = callbacks_cfg.get('patience', None)
        if patience_cfg is None:
            error_msg = (
                "Fatal error in _train_with_callbacks: "
                "the setting variable 'training -> callbacks -> patience' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(error_msg)

        callbacks = [
            sync_epoch_callback,
            monitor_callback,
            training_progress_callback,
            print_callback,
            ProgbarLogger(),

            # Save the best version of the model for final inference
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(best_path),
                save_best_only=True,
                monitor="val_loss",
                mode="min"
            ),

            # Save the state of the last epoch to enable 'load_model' resume logic
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(last_path),
                save_best_only=False,
                verbose=1
            ),

            # Stop training if validation loss plateaus.
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience_cfg,
                restore_best_weights=True,
                verbose=1
            ),

            # Export log for visualization in tensorboard
            tf.keras.callbacks.TensorBoard(
                log_dir=str((Path(output_dir) / "tensorboard_logs").resolve()),
                histogram_freq=1,
                profile_batch=(0)
            )
        ]

        logger.info("Training callbacks configured.",
                    extra={"callbacks": [callback.__class__.__name__ for callback in callbacks]})

        # Calculate steps_per_epoch and validation_step, since the dataset
        # uses .repeat()  (infinite stream)
        training_cfg = self._config.get('training', None)
        if training_cfg is None:
            error_msg = (
                "Fatal error in _train_with_callbacks: "
                "the setting variable 'training' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(error_msg)

        batch_size = training_cfg.get('batch_size', None)

        if (batch_size is None) or (batch_size <= 0):
            msg_error = (
                "Error in function _train_with_callbacks: "
                "the setting variable 'training -> batch_size' is missing or invalid. "
                "It must be a strictly positive integer. Please check your YAML configuration."
            )
            logger.error(msg_error)
            raise ValueError(msg_error)

        steps_per_epoch = max(1, self._nb_train // batch_size)
        validation_steps = max(1, self._nb_val // batch_size)

        logger.info(
            f"Starting model training: {steps_per_epoch} steps/epoch, "
            f"{validation_steps} validation steps."
        )

        # Clear session and collect garbage to maximize available VRAM before fitting
        gc.collect()
        log_memory_usage(stage_name="Pre-fit memory state", process=self._process)

        try:
            epochs_cfg = training_cfg.get("epochs", None)

            if epochs_cfg is None:
                msg_error = (
                    "Error in function _train_with_callbacks: "
                    "the setting variable 'training -> epochs' is required "
                    "but was not found. Please check your YAML file structure."
                )
                logger.error(msg_error)
                raise ValueError(msg_error)

            # Execute the training loop
            history = self._model.fit(
                self._train_dataset,
                epochs=epochs_cfg,
                steps_per_epoch=steps_per_epoch,
                validation_data=self._validation_dataset,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=0
            )

            # Calculation of the average accuracy for the logger
            acc_keys = [k for k in history.history.keys() if k.endswith('_accuracy')]
            if acc_keys:
                final_acc = sum([history.history[k][-1] for k in acc_keys]) / len(acc_keys)
            else:
                final_acc = 0

            logger.info("Model training completed successfully.",
                        extra={
                            "final_loss": history.history['loss'][-1],
                            "final_accuracy": final_acc
                        })
            return history

        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            raise
