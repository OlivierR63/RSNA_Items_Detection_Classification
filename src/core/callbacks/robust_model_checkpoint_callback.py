# coding: utf-8

import tf_keras
import logging


class RobustModelCheckpointCallback(tf_keras.callbacks.ModelCheckpoint):
    """
    Custom ModelCheckpoint that logs the specific success/failure result
    of the save operation directly via the application logger.
    """
    def __init__(self, logger: logging.Logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        try:
            # Attempt to save the model using the parent class method
            super().on_epoch_end(epoch, logs)

            # If we reached this line, the save was successful (or at least no exception raised)
            self.logger.info(
                f"Successfully saved checkpoint for epoch {epoch + 1} at: {self.filepath}"
            )
        except Exception as e:
            # If an exception is raised, log the failure clearly
            self.logger.error(
                f"FAILED to save checkpoint for epoch {epoch + 1} to {self.filepath}. Error: {e}"
            )
            # Re-raise the exception to let the training process know something went wrong
            raise e

    def get_config(self):
        """
        Returns the configuration of the callback for serialization.
        """
        config = super().get_config()
        config.pop("logger", None)  # Remove logger from config, as it cannot be serialized
        return config
