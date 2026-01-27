# coding: utf-8
import tensorflow as tf

class LogTrainingProgress(tf.keras.callbacks.Callback):
    """
        Custom callback to redirect Keras training metrics to the application logger.
    """
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

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
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        
        self.logger.info(
            f"Epoch {epoch + 1} finished: {metrics_str}",
            extra={
                "epoch": epoch + 1,
                "metrics": logs,
                "status": "in_progress"
            }
        )