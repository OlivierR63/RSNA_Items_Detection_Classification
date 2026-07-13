# coding: utf-8

from __future__ import annotations  # Enables deferred evaluation of type annotations
import tf_keras
from typing import Any, TYPE_CHECKING

# If TYPE_CHECKING is True, we import ModelTrainer for type hints without causing circular imports
if TYPE_CHECKING:
    from src.projects.lumbar_spine.model_trainer import ModelTrainer


class EpochSyncCallback(tf_keras.callbacks.Callback):
    """
    Synchronizes the dataset epoch with the Keras training loop.
    """
    def __init__(
        self,
        trainer: 'ModelTrainer',  # Model trainer used to synchronize the global epoch
        initial_offset: int
    ) -> None:
        super().__init__()
        self._trainer = trainer
        self._initial_offset = initial_offset

    def on_epoch_begin(
        self,
        epoch: int,
        logs: dict[str, Any] | None = None
    ) -> None:
        # Ensure logs is initialized as an empty dictionary if None
        logs = logs or {}

        # Best practice: Always call the method of the parent class
        super().on_epoch_begin(epoch, logs)

        # Sync the dataset manager with the absolute (global) epoch
        self._trainer._set_epoch(epoch + self._initial_offset)

    def get_config(self) -> dict[str, Any]:
        # Allow Keras to save this callback's state
        config = super().get_config()
        config.update({"initial_offset": self._initial_offset})
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'EpochSyncCallback':
        # Logic to rebuild this callback from a saved model.
        # trainer is set to None as it will be reinjected by the ModelTrainer at runtime.
        return cls(trainer=None, initial_offset=config["initial_offset"])
