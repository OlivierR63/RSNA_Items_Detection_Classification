# coding: utf-8

import tf_keras
import psutil
import os
from typing import Any
from time import time, perf_counter, strftime, gmtime


class LogTrainingCallback(tf_keras.callbacks.Callback):
    """
    Custom callback to monitor training performance, timing, and progress.

    This callback provides real-time feedback during the training process,
    including a visual progress bar, high-resolution step timing using
    perf_counter(), and a global Estimated Time of Arrival (ETA).
    """
    def __init__(self, logger, validation_steps=0):
        """
        Initializes the callback with a specific application logger.

        Args:
            logger (logging.Logger): The logger instance where summaries
                will be redirected.
        """
        super().__init__()
        self.logger = logger
        self.step_times = []
        self._process = psutil.Process(os.getpid())
        self._val_steps_fixed = validation_steps

        # Initialize attributes with default values to avoid AttributeErrors
        self.nb_steps = 0
        self.nb_epochs = 0
        self.total_batches = 0
        self.current_epoch = 0

    def on_train_begin(self, logs=None) -> None:
        """
        Sets up the global session parameters and initializes the session timer.

        Args:
            logs (dict, optional): Currently unused by this method but required by
            the Keras API.
        """
        logs = logs or {}

        super().on_train_begin(logs)

        # Initialize record for average step time
        self.step_times = []

        # Safely retrieve parameters.
        # As a reminder: the keys 'epochs' and 'steps' derive from the arguments
        # 'epochs' and 'steps_per_epoch' passed through the model.fit() method
        try:
            self.nb_epochs = int(self.params.get('epochs', 0))
            self.nb_steps = int(self.params.get('steps', 0))
        except (ValueError, TypeError):
            raise ValueError("Parameters 'epochs' or 'steps' must be integers.")

        self.total_batches = self.nb_epochs * self.nb_steps

        if self.total_batches <= 0:
            raise ValueError(
                "Invalid configuration for 'epochs' or 'steps'. "
                "Please revise the YAML configuration file"
            )

        # Ensure parameters are integers for calculation
        self.logger.info("\n" + "="*50)
        self.logger.info("    Training session started!")
        self.logger.info(f"    Targeting {self.nb_epochs} epochs with {self.nb_steps} steps each.")
        self.logger.info("="*50 + "\n")

    def on_epoch_begin(self, epoch, logs: dict[str, Any] | None = None):
        """
        Tracks the current epoch index and resets step-specific metrics.

        Args:
            epoch (int): Index of the epoch starting.
            logs (dict, optional): Currently unused by this method but
            required by the Keras API.
        """
        logs = logs or {}
        super().on_epoch_begin(epoch, logs)
        self.current_epoch = epoch
        self.step_times = []

    def on_train_batch_begin(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        """
        Captures the exact start time of a training batch.

        Args:
            batch (int): Index of the batch within the current epoch.
            logs (dict, optional): Currently unused by this method but required
            by the Keras API.
        """
        logs = logs or {}

        # Correctly call the parent method to maintain Keras callback chain
        super().on_train_batch_begin(batch, logs)

        # Record the start time of the current batch
        self.batch_start_time = perf_counter()

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None):
        """
        Calculates batch duration, updates the progress bar, and computes the ETA.

        Args:
            batch (int): Index of the batch that just finished.
            logs (dict, optional): Metrics for the current batch (e.g., loss).
        """
        logs = logs or {}

        # Force conversion to python int to avoid EagerTensor conflicts
        batch_int = int(batch)

        super().on_train_batch_end(batch, logs)

        # Calculate time spent on this step
        step_time = perf_counter() - self.batch_start_time
        self.step_times.append(step_time)

        # Calculate ETA
        # Total batches processed so far across all epochs
        current_epoch = getattr(self, 'current_epoch', 0)
        batches_done = (current_epoch * self.nb_steps) + batch_int
        batches_left = self.total_batches - batches_done

        avg_time = sum(self.step_times) / len(self.step_times)
        eta_seconds = batches_left * avg_time
        eta_timestamp = time() + eta_seconds
        eta_str = strftime("%H:%M:%S", gmtime(eta_timestamp))

        # Log the performance metrics
        info_msg = f" Batch {batch_int + 1:04d} | Time: {step_time:.2f}s | ETA: {eta_str}"
        self.logger.info(info_msg)

    def on_test_begin(self, logs: dict[str, Any] | None = None):
        """
        Signals the start of the validation (testing) phase.

        Args:
            logs (dict, optional): Currently unused by this method.
        """
        logs = logs or {}
        super().on_test_begin(logs)

        # Retrieve validation steps from Keras params
        # Note: 'validation_steps' is often stored as 'steps' in the test context
        val_steps = self._val_steps_fixed

        self.logger.info("\n" + "-"*30)
        self.logger.info(f" >>> Starting Validation Phase ({val_steps} steps)")
        self.logger.info("-"*30)
        self.val_start_time = perf_counter()

    def on_test_end(self, logs: dict[str, Any] | None = None):
        """
        Signals the completion of the validation phase and logs its duration.

        Args:
            logs (dict, optional): Metrics from the validation phase. Required
            by the Keras API.
        """
        logs = logs or {}
        super().on_test_end(logs)

        val_duration = perf_counter() - self.val_start_time
        self.logger.info(f" >>> Validation Phase finished in {val_duration:.2f}s")
        self.logger.info("-"*30 + "\n")

    def on_epoch_end(self, epoch: int, logs=None):
        """
        Summarizes final epoch metrics and redirects them to the logger.

        Args:
            epoch (int): Index of the epoch that just finished.
            logs (dict, optional): Aggregated metrics for the epoch.
        """
        logs = logs or {}

        super().on_epoch_end(epoch, logs)

        # Format metrics into a readable string for the console/log file
        # Example: loss: 0.4521 - accuracy: 0.8912
        metrics_str = " - ".join([f"{k}: {v:.4f}\n\t" for k, v in logs.items()])

        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
        else:
            avg_step_time = 0

        summary_msg = (
            f"\n\nEpoch {epoch + 1} finished: \n\t - {metrics_str} | "
            f"Avg Step Time: {avg_step_time:.2f}s"
        )
        self.logger.info(f"\n >>> {summary_msg}")

        self.logger.info(
            summary_msg,
            extra={
                "epoch": epoch + 1,
                "metrics": logs,
                "avg_step_time": round(avg_step_time, 2),
                "status": "in_progress"
            }
        )

        # Reset step times for next epoch
        self.step_times = []

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the callback for serialization.
        """
        config = super().get_config()
        config.update({
            "validation_steps": self._val_steps_fixed
        })

        # Important: Remove the logger from the config, as it cannot be serialized
        # This prevents issues when saving/loading the model with this callback.
        config.pop("logger", None)

        return config
