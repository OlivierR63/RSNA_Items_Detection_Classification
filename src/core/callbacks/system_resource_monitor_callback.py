# coding utf-8

import psutil
import os
import tf_keras
import gc
import logging


class SystemResourceMonitorCallback(tf_keras.callbacks.Callback):
    """
    Callback for monitoring system resources and managing memory during training.

    This callback performs two primary functions to prevent Out-of-Memory (OOM)
    crashes and system instability, which are common when processing large
    3D medical imaging volumes on CPU:

    1. Batch-level Monitoring: At the end of every training batch, it checks
       the total system RAM usage. If the usage exceeds a defined percentage
       threshold, it triggers an emergency stop of the training process.

    2. Epoch-level Cleanup: At the end of every epoch, it clears the Keras
       global state and triggers the Python garbage collector. This prevents
       memory leaks and the accumulation of temporary tensors (gradients,
       validation buffers) between training cycles.

    Attributes:
        memory_threshold (float): The maximum allowed percentage of system RAM
            usage (0.0 to 100.0) before stopping training.
        process (psutil.Process): The current OS process instance used to
            track process-specific memory consumption.
    """
    def __init__(
        self,
        memory_threshold_percent: float = 90.0,
        frequency=10,
        logger: logging.Logger | None = None
    ):
        """
        Initializes the monitor with custom thresholds and logging settings.

        Args:
            memory_threshold_percent (float): Percentage of system RAM usage
                triggering an automatic training stop (default: 90.0).
            frequency (int): Interval in batches at which the current process
                memory usage is logged (default: 10).
            logger (logging.Logger, optional): Dedicated logger for monitor
                events. If None, the module's default logger is used.
        """
        super().__init__()
        self.memory_threshold = memory_threshold_percent
        self.frequency = frequency
        self.logger = logger or logging.getLogger(__name__)
        self.process = psutil.Process(os.getpid())

    def on_train_batch_end(self, batch, logs=None) -> None:
        """
        Check system and process memory usage after each training batch.

        Args:
            batch (int): Index of the batch within the current epoch.
            logs (dict, optional): Dictionary containing metric results for the
                current batch (e.g., loss, accuracy). Provided by the Keras
                training loop, kept for API compatibility even if unused here.

        If the total system RAM usage exceeds the defined threshold,
        it sets the model's 'stop_training' flag to True to prevent a crash.
        """

        logs = logs or {}
        super().on_train_batch_end(batch, logs)

        # Force conversion to python int to avoid EagerTensor conflicts
        batch_int = int(batch) + 1

        # 1. Get system-wide memory usage
        mem = psutil.virtual_memory()

        # 2. Emergency stop (Check before anything else)
        if float(mem.percent) > self.memory_threshold:
            critical_msg = (
                f"\n[CRITICAL] Memory usage ({mem.percent}%) exceeded threshold! "
                f"Stopping training to prevent system crash."
            )
            self.logger.critical(critical_msg)
            self.model.stop_training = True

        # 3. Get current process memory usage (RSS)
        if batch_int % self.frequency == 0:
            process_mem_gb = self.process.memory_info().rss / (1024 ** 3)

            # 3. Log the status
            # if batch_int % 5 == 0: # Log every 5 batches to avoid cluttering
            info_msg = (
                f"\n\n[System Monitor] Batch {batch_int:03d}: "
                f"System RAM: {mem.percent}% | "
                f"Process RAM: {process_mem_gb:.2f} GB"
            )

            self.logger.info(info_msg)

    def on_epoch_end(self, epoch, logs=None):
        """
        Perform a deep memory cleanup at the end of every epoch.

        Args:
            epoch (int): Index of the epoch.
            logs (dict, optional): Dictionary containing metric results for the
                epoch. Provided by Keras, included to maintain signature
                consistency with the Callback API.

        Forces Python's garbage collector to release unreferenced tensors,
        gradients, and validation buffers before starting the next epoch.
        """

        logs = logs or {}
        super().on_epoch_end(epoch, logs)

        # 1. Trigger the Python Garbage Collector
        # This forces the immediate release of unreferenced objects in RAM
        nb_objects = gc.collect()

        # 2. Optional: Print a confirmation message to the console
        info_msg = (
            f"\n[System] Memory cleanup completed after Epoch {epoch + 1}: "
            f"Released {nb_objects} unreferenced objects"
        )
        self.logger.info(info_msg)

    def get_config(self):
        """
        Returns the configuration of the callback for serialization.

        Returns:
            dict: Configuration parameters used for model saving/loading.
        """
        config = super().get_config()
        config.update({
            "memory_threshold_percent": self.memory_threshold
        })
        return config
