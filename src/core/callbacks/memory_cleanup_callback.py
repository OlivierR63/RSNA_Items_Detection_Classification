# coding: utf-8

import gc
import tf_keras
import logging
from typing import Dict, Any


class MemoryCleanupCallback(tf_keras.callbacks.Callback):
    """
    A Keras callback to manage memory usage during training.

    This callback provides utilities to trigger manual garbage collection
    at the end of every epoch.

    Attributes:
        run_gc (bool): If True, triggers garbage collection at epoch end.
        _logger (logging.Logger | None): Optional logger for tracking memory events.
    """

    def __init__(
        self,
        run_gc: bool = True,
        logger: logging.Logger | None = None
    ) -> None:
        """
        Initializes the MemoryCleanupCallback.

        Args:
            run_gc (bool): Enable/disable garbage collection. Defaults to True.
            logger (logging.Logger | None): A logger instance for monitoring.
        """
        super().__init__()
        self.run_gc = run_gc
        self._logger = logger

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        """
        Executes memory cleanup routines at the end of an epoch.

        Calls super().on_epoch_end() defensively, followed by optional
        garbage collection and session clearing.

        Args:
            epoch (int): The index of the epoch that just finished.
            logs (Dict[str, Any] | None): Dictionary of metrics from the epoch.
        """
        # Defensive programming: call super() to ensure full chain of execution
        # is respected, even if currently implemented as empty in the base class.
        logs = logs or {}

        super().on_epoch_end(epoch, logs=logs)
        if self.run_gc:
            nb_objects = gc.collect()
            if self._logger is not None:
                info_msg = f"RAM management: {nb_objects} objects have been cleansed from memory"
                self._logger.info(info_msg)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the callback for serialization.

        The logger is removed from the configuration to prevent pickling errors.

        Returns:
            Dict[str, Any]: A dictionary containing the callback configuration.
        """
        config = super().get_config()
        config.update({
            "run_gc": self.run_gc,
        })

        # Prevent serialization errors of the non-serializable logger object
        config.pop("_logger", None)
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MemoryCleanupCallback':
        """
        Creates an instance of MemoryCleanupCallback from its configuration.

        Args:
            config (Dict[str, Any]): Dictionary of configuration values.

        Returns:
            MemoryCleanupCallback: A new instance of the callback.
        """
        return cls(**config)
