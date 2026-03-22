# coding: utf-8

import tensorflow as tf
import psutil
import os
from time import time


class LogTrainingCallbacks(tf.keras.callbacks.Callback):
    """
        Custom callback to redirect Keras training metrics to the application logger.
    """
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.step_times = []
        self._process = psutil.Process(os.getpid())

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.
        """
        self.epoch_start_time = time()

        # Initialize record for average step time
        self.step_times = []

        # Safely retrieve parameters
        epochs_param = self.params.get('epochs', 0)
        steps_param = self.params.get('steps', 0)

        # Ensure we have integers for comparison
        try:
            nb_epochs = int(epochs_param) if epochs_param is not None else 0
            nb_steps = int(steps_param) if steps_param is not None else 0

        except (ValueError, TypeError):
            nb_epochs = epochs_param  # fallback to '?' or 0
            nb_steps = steps_param

        print("\n" + "="*50)
        print("    Training session started!")

        # Proper pluralization logic
        epoch_str = "epoch" if nb_epochs == 1 else "epochs"
        step_str = "step" if nb_steps == 1 else "steps"

        print(f"    Targeting {nb_epochs} {epoch_str} with {nb_steps} {step_str} each.")
        print("="*50 + "\n")

    def on_train_batch_begin(self, batch, logs=None):
        # Force conversion to python int to avoid EagerTensor conflicts
        # batch_int = int(batch)

        # Record the start time of the current batch
        self.batch_start_time = time()

        # We use datetime for the human-readable display
        # current_time = datetime.now().strftime("%H:%M:%S")

    def on_train_batch_end(self, batch, logs=None):

        # Force conversion to python int to avoid EagerTensor conflicts
        batch_int = int(batch)

        # Calculate time spent on this step
        step_time = time() - self.batch_start_time
        self.step_times.append(step_time)

        # Get current RAM usage
        ram_usage = self._process.memory_info().rss / (1024 ** 3)  # Convert to GB

        # Log the performance metrics
        print(f"\n >>> Step {batch_int + 1:03d} | Time: {step_time:.2f}s | RAM: {ram_usage:.2f} GB")

    def on_epoch_end(self, epoch, logs=None):
        """
        Runs at the end of each epoch to log training and validation metrics.

        Args:
            epoch (int): The index of the epoch that just finished.
            logs (dict): Dictionary containing the training metrics (e.g., loss, accuracy).
                         Validation metrics are included if validation_data is provided.
        """
        logs = logs or {}
        # Format metrics into a readable string for the console/log file
        # Example: loss: 0.4521 - accuracy: 0.8912
        metrics_str = " - ".join([f"{k}: {v:.4f}\n\t" for k, v in logs.items()])

        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
        else:
            avg_step_time = 0

        summary_msg = (
            f"\n\nEpoch {epoch + 1} finished: \n\t - {metrics_str} | "
            "Avg Step Time: {avg_step_time:.2f}s"
        )
        print(f"\n >>> {summary_msg}")

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
