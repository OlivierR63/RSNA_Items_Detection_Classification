# coding: utf-8

import gc
import tf_keras
import logging
from typing import Dict, Any


class MemoryCleanupCallback(tf_keras.callbacks.Callback):
    """
    Callback dedicated to memory management at the end of each epoch.

    Args:
        run_gc (bool): Whether to run garbage collector. Default: True.
        clear_session (bool): Whether to clear Keras session. Default: False.
                              WARNING: Setting this to True during fit() will likely
                              break the model graph. Only use for specific workflows.
    """

    def __init__(
        self,
        run_gc: bool = True,
        clear_session: bool = False,
        logger: logging.Logger | None = None
    ) -> None:
        super().__init__()
        self.run_gc = run_gc
        self.clear_session = clear_session
        self._logger = logger

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        if self.run_gc:
            nb_objects = gc.collect()
            if self._logger is not None:
                info_msg = f"RAM management: {nb_objects} objects have been cleansed from memory"
                self._logger.info(info_msg)

        if self.clear_session:
            tf_keras.backend.clear_session()

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "run_gc": self.run_gc,
            "clear_session": self.clear_session
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MemoryCleanupCallback':
        return cls(**config)
