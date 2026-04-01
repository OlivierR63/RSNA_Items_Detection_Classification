# coding utf-8

import psutil
import os
import tensorflow as tf
import gc


class SystemResourceMonitorCallbacks(tf.keras.callbacks.Callback):
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
    def __init__(self, memory_threshold_percent=90.0):
        super().__init__()
        self.memory_threshold = memory_threshold_percent
        self.process = psutil.Process(os.getpid())

    def on_train_batch_end(self, batch, logs=None):
        """
        Check system and process memory usage after each training batch.

        If the total system RAM usage exceeds the defined threshold,
        it sets the model's 'stop_training' flag to True to prevent a crash.
        """

        # Force conversion to python int to avoid EagerTensor conflicts
        batch_int = int(batch) + 1

        # 1. Get system-wide memory usage
        mem = psutil.virtual_memory()

        # 2. Get current process memory usage (RSS)
        process_mem_gb = self.process.memory_info().rss / (1024 ** 3)

        # 3. Log the status
        # if batch_int % 5 == 0: # Log every 5 batches to avoid cluttering
        print(f"\n\n[System Monitor] Batch {batch_int:03d}: "
              f"System RAM: {mem.percent}% | "
              f"Process RAM: {process_mem_gb:.2f} GB")

        # 4. Emergency stop
        if float(mem.percent) > self.memory_threshold:
            print(f"\n[CRITICAL] Memory usage ({mem.percent}%) exceeded threshold! "
                  "Stopping training to prevent system crash.")
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        """
        Perform a deep memory cleanup at the end of every epoch.

        Resets the Keras global state and forces Python's garbage collector
        to release unreferenced tensors, gradients, and validation buffers
        before starting the next epoch.
        """

        # 1. Clear the Keras global state and free the computational graph
        # This prevents the accumulation of temporary tensors from the previous epoch
        tf.keras.backend.clear_session()

        # 2. Trigger the Python Garbage Collector
        # This forces the immediate release of unreferenced objects in RAM
        gc.collect()

        # 3. Optional: Print a confirmation message to the console
        print(f"\n[System] Memory cleanup completed after Epoch {epoch + 1}")
