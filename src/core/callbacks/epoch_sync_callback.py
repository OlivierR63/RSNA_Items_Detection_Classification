# coding: utf-8

import tf_keras
from typing import Any, TYPE_CHECKING

# If TYPE_CHECKING is True, we import ModelTrainer for type hints without causing circular imports
if TYPE_CHECKING:
    from src.projects.lumbar_spine.model_trainer import ModelTrainer


class EpochSyncCallback(tf_keras.callbacks.Callback):
    """
    Synchronizes the dataset epoch with the Keras training loop.
    Replaces the LambdaCallback for better serialization support.
    """
    def __init__(
        self,
        trainer: ModelTrainer,
        initial_offset: int
    ) -> None:
        super().__init__()
        self._trainer = trainer
        self._initial_offset = initial_offset

    def on_epoch_begin(
        self,
        epoch: int,
        logs: dict[str, float] | None=None) -> None:
        # Sync the dataset manager with the absolute (global) epoch
        self._trainer._set_epoch(epoch + self._initial_offset)

    def get_config(self) -> dict[str, Any]:
        # Allow Keras to save this callback's state
        config = super().get_config()
        config.update({"initial_offset": self._initial_offset})
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'EpochSyncCallback':
        # Logic to rebuild this callback from a saved model
        # trainer=None because it will be reinjected by the ModelTrainer at the next runtime.
        return cls(trainer=None, initial_offset=config["initial_offset"])
