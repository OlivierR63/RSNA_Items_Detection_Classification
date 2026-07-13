# coding: utf-8

import tf_keras
from typing import Dict, Any
import logging


class PrintEpochCallback(tf_keras.callbacks.Callback):
    """
    Callback that logs the start and end of epochs in a clean format.
    """
    def __init__(
        self,
        logger: logging.Logger | None = None,
        batch_log_frequency: int = 100
    ) -> None:
        """
        Initialize the callback with a logger and a batch frequency.

        Args:
            logger: The logging.Logger instance to use.
            batch_log_frequency: How often to log batch progress (must be > 0).

        Raises:
            ValueError: If batch_log_frequency is less than or equal to 0.
        """
        super().__init__()

        # Verify that the frequency is an integer
        if not isinstance(batch_log_frequency, int):
            raise TypeError(
                f"batch_log_frequency must be an integer, got {type(batch_log_frequency).__name__}"
            )

        if batch_log_frequency <= 0:
            raise ValueError(
                f"batch_log_frequency must be a positive integer, got {batch_log_frequency}"
            )

        self.logger = logger
        self._batch_log_frequency = batch_log_frequency

    def on_epoch_begin(
        self,
        epoch: int,
        logs: Dict[str, Any] = None
    ) -> None:
        """
        Log the start of an epoch.

        Args:
            epoch: The current epoch index.
            logs: Dictionary of training metrics. Required by the Keras API
        """
        logs = logs or {}
        super().on_epoch_begin(epoch, logs)
        if self.logger:
            self.logger.info(f"--- Starting Epoch {epoch + 1} ---")

    def on_epoch_end(
        self,
        epoch: int,
        logs: Dict[str, Any] = None
    ) -> None:
        """
        Log the end of an epoch.

        Args:
            epoch: The current epoch index.
            logs: Dictionary of training metrics. Required by the Keras API.
        """
        logs = logs or {}
        super().on_epoch_end(epoch, logs)
        if self.logger:
            self.logger.info(f"--- Finished Epoch {epoch + 1} ---")

    def on_batch_end(
        self,
        batch: int,
        logs: Dict[str, Any] = None
    ) -> None:
        """
        Log batch-level loss at specified frequency.

        Args:
            batch: The current batch index.
            logs: Dictionary of training metrics. Required by the Keras API
        """
        logs = logs or {}
        super().on_batch_end(batch, logs)

        if self.logger and (batch + 1) % self._batch_log_frequency == 0:
            loss = logs.get("loss", "N/A") if logs else "N/A"

            try:
                # Attempt to convert loss to float for consistent formatting
                loss_val = float(loss)
                self.logger.info(f"Batch {batch + 1:04d} finished (Loss: {loss_val:.4f})")
            except (TypeError, ValueError):
                self.logger.info(f"Batch {batch + 1:04d} finished (Loss: N/A)")

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration for serialization.

        This method is essential for model persistence. When saving a model
        (e.g., via model.save()), Keras calls this method to store the
        callback's parameters.
        """
        # We don't save the logger as it is not picklable.
        # It will be injected upon reconstruction.
        config = super().get_config()
        config.update(
            {
                "batch_log_frequency": self._batch_log_frequency
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PrintEpochCallback':
        """
        Recreates the callback from config.

        This method is used when loading a saved model. It allows Keras to
        reinstantiate the callback with the parameters saved in get_config().
        Note: The logger must be re-injected manually after reconstruction.
        """
        return cls(logger=None, **config)
