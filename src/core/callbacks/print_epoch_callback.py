# coding: utf-8

import tf_keras
from typing import Dict, Any
import logging


class PrintEpochCallback(tf_keras.callbacks.Callback):
    """
    Callback that logs the start and end of epochs in a clean format.
    Replaces simple LambdaCallback for better control and serialization.
    """
    def __init__(
        self,
        logger: logging.Logger | None = None,
        batch_log_frequency: int = 100
    ) -> None:
        super().__init__()
        self.logger = logger
        self._batch_log_frequency = batch_log_frequency

    def on_epoch_begin(
        self,
        epoch: int,
        logs: Dict[str, Any] = None
    ) -> None:
        if self.logger:
            self.logger.info(f"--- Starting Epoch {epoch + 1} ---")

    def on_epoch_end(
        self,
        epoch: int,
        logs: Dict[str, Any] = None
    ) -> None:
        if self.logger:
            self.logger.info(f"--- Finished Epoch {epoch + 1} ---")

    def on_batch_end(
        self,
        batch: int,
        logs: Dict[str, Any] = None
    ) -> None:
        if self.logger and (batch + 1) % self._batch_log_frequency == 0:
            loss = logs.get("loss", "N/A") if logs else "N/A"
            self.logger.info(f"Batch {batch + 1:04d} finished (Loss: {loss:.4f})")

    def get_config(self) -> Dict[str, Any]:
        # We don't save the logger as it is not picklable.
        # It will be injected upon reconstruction.
        return super().get_config()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PrintEpochCallback':
        return cls(logger=None)
