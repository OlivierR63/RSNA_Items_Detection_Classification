# coding: utf-8

import os
import gc
import psutil
import logging
from pathlib import Path
from typing import Optional, Any, Tuple

from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
from src.core.utils.logger import get_current_logger, log_method
from src.core.utils.log_training_progress import LogTrainingProgress

import tensorflow as tf

# In recent TensorFlow versions, Keras 3 is a separate package installed
# as a requirement of the tensorflow meta-package. 
from keras.callbacks import LambdaCallback, ProgbarLogger


def log_memory_usage(stage_name=""):
    """
    Logs the current RAM usage of the Python process and the overall system.
    Useful for monitoring memory leaks during 3D medical imaging pipelines.
    """
    # Get the current process ID and its memory information
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # Convert Resident Set Size (RSS) to Megabytes (MB)
    # RSS represents the portion of memory occupied by a process that is held in RAM
    mem_mb = mem_info.rss / (1024 * 1024)
    
    # Retrieve the total percentage of system RAM currently in use
    total_ram_percent = psutil.virtual_memory().percent
    
    # Output the memory snapshot to the console
    print(f">>> [RAM] {stage_name} | Process: {mem_mb:.2f} MB | System: {total_ram_percent}%")


class ModelTrainer:
    """
    Orchestrates the training process for the RSNA Lumbar Spine model.
    
    This class handles dataset preparation, callback configuration, and 
    the execution of the training loop. It integrates structured logging 
    and memory monitoring to ensure stability when processing large 
    3D medical imaging volumes.
    """
    def __init__(
        self, 
        model: tf.keras.Model, 
        config: dict, 
        logger: Optional[logging.Logger] = None, 
        model_depth: int = 1
    ) -> None:

        """Initializes the ModelTrainer."""
        # Force a clean break between lines
        self._model = model
        self._config = config
        self._logger = logger or get_current_logger()

        # Test line: if the error is here, it's a scope issue
        self._model_depth = model_depth

        # Unpack dataset results
        results = self._prepare_dataset_for_training()
        self._nb_train, self._train_dataset, self._nb_val, self._validation_dataset = results

    @log_method()
    def train_model(
        self,
        *,
        logger: Optional[logging.Logger]=None,
    ) -> None:

        """
            Training function with automatic logger injection and structured logging.
        """

        logger = logger or self._logger

        logger.info(
            "Loading 3D model...",
            extra={
                   "action": "load_model",
                   "model_type": self._config['model_3d']['type']
                   }
        )

        try:
            # Train the model
            _ = self._train_with_callbacks()

            logger.info(
                f"Model saved to {self._config['checkpoint_path']}",
                extra={
                    "status": "success",
                    "model_path": str(Path(self._config["checkpoint_path"]))
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
    def _prepare_dataset_for_training(self, logger: Optional[logging.Logger] = None) -> Tuple[int, tf.data.Dataset, int, tf.data.Dataset]:
        """
        Creates TensorFlow training and validation datasets with logging.
        Stores and returns dataset metadata and pipeline objects.

        Args:
            logger: Logger instance for process tracking. If None, retrieves 
                    the current active logger.

        Returns:
            Tuple containing:
                - nb_train (int): Number of training samples.
                - train_dataset (tf.data.Dataset): The pre-processed training pipeline.
                - nb_val (int): Number of validation samples.
                - val_dataset (tf.data.Dataset): The pre-processed validation pipeline.

        Raises:
            Exception: If the pipeline construction via LumbarDicomTFRecordDataset fails.
        """

        logger = logger or self._logger

        logger.info("Setting up TensorFlow dataset pipeline...",
                    extra={"action": "create_dataset", "batch_size": self._config["batch_size"]})

        try:
            batch_size = self._config["batch_size"]

            log_memory_usage("Before Dataset creation")

            dataset_object = LumbarDicomTFRecordDataset(self._config, self._logger, self._model_depth)

            # batch_size details the number of studies processed simultaneously
            nb_train, train_dataset, nb_val, val_dataset = dataset_object.build_tf_dataset_pipeline(batch_size=batch_size)

            log_memory_usage("After Dataset creation")

            logger.info("Training and validation dataset created successfully.",
                        extra={"status": "success", "batch_size": self._config["batch_size"]})

            return nb_train, train_dataset, nb_val, val_dataset
    
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            raise


    @log_method()
    def _train_with_callbacks(self, *,logger: Optional[logging.Logger] = None) -> Any:

        """Trains model with callbacks and logging."""
        logger = logger or self._logger

        logger.info("Setting up training callbacks...", extra={"action": "setup_callbacks"})

        # Paths for checkpointing
        # 'best_model.keras' stores the weights with the lowest validation loss
        # 'last_model.keras' is updated every epoch to allow training resumption
        checkpoint_dir = Path(self._config.get("checkpoint_dir", "checkpoints"))
        best_path = checkpoint_dir / "best_model.keras"
        last_path = checkpoint_dir / "last_model.keras"

        # In ModelTrainer._train_with_callbacks
        print_callback = LambdaCallback(
            on_batch_end=lambda batch, logs: print(
                f" >>> Step {batch:04d} | Loss: {logs['loss']:.4f} | Avg Acc: {logs.get('accuracy', 0):.4f}"
            ) if batch % 5 == 0 else None # Log every 5 steps for readability
        )

        callbacks = [
            print_callback,
            ProgbarLogger(),
            LogTrainingProgress(logger),

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
                verbose=0
            ),

            # Stop training if validation loss plateaus.
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self._config['patience'],
                restore_best_weights=True,
                verbose=1
            ),

            # Export log for visualization in tensorboard
            tf.keras.callbacks.TensorBoard(
                log_dir=str(Path(self._config["output_dir"]) / "tensorboard_logs"),
                histogram_freq=1
            )
        ]

        logger.info("Training callbacks configured.",
                    extra={"callbacks": [c.__class__.__name__ for c in callbacks]})

        # Calculate steps_per_epoch and validation_step, since the dataset
        # uses .repeat()  (infinite stream)
        batch_size = self._config.get('batch_size', 1)

        if batch_size <= 0:
            msg_error = (
                f"Error in function ModelTrainer.train_with_callback(): "
                "the configuration parameter 'batch_size' shall be strictly positive"
            )
            logger.error(msg_error)
            raise ValueError(msg_error)

        steps_per_epoch = max(1, self._nb_train // batch_size)
        validation_steps = max(1, self._nb_val // batch_size)

        logger.info(
            f"Starting model training: {steps_per_epoch} steps/epoch, {validation_steps} validation steps.",
            extra={
                "epochs": self._config.get("epochs", 1),
                "batch_size": self._config.get("batch_size")
            }
        )

        # Clear session and collect garbage to maximize available VRAM before fitting
        gc.collect()
        tf.keras.backend.clear_session()
        log_memory_usage("Pre-fit memory state")

        for img, label in self._train_dataset.take(1):
            log_memory_usage("After loading 1st Batch")
            break

        try:
            # Execute the training loop
            history = self._model.fit(
                self._train_dataset,
                epochs=self._config["epochs"],
                steps_per_epoch=steps_per_epoch,
                validation_data=self._validation_dataset,
                validation_steps=validation_steps,
                callbacks=callbacks, 
                verbose=1
            )

            # Calculation of the average accuracy for the logger
            acc_keys = [k for k in history.history.keys() if k.endswith('_accuracy')]
            final_acc = sum([history.history[k][-1] for k in acc_keys]) / len(acc_keys) if acc_keys else 0

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
