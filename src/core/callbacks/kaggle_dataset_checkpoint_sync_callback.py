# coding: utf-8

import os
import subprocess
import tf_keras
import shutil
from pathlib import Path
from src.core.utils.logger import get_current_logger


class KaggleDatasetCheckpointSyncCallback(tf_keras.callbacks.Callback):
    """
    Callback to automatically push saved checkpoints to a private Kaggle Dataset
    at the end of each epoch to prevent data loss from Timeouts.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_filename: str = "model_checkpoint.keras",
        dataset_id: str = "olivierrochat/rsna-lumbar-spine-checkpoints"
    ):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_filename = checkpoint_filename
        self.dataset_id = dataset_id
        self.logger = get_current_logger()
        self._is_uploading: bool = False

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        # Ensure the target checkpoint file was successfully written by ModelCheckpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            self.checkpoint_filename
        )

        if not os.path.exists(checkpoint_path):
            self.logger.warning(
                f"[KaggleSync] Checkpoint file not found at {checkpoint_path}. Skipping sync."
            )
            return

        self.logger.info(
            f"[KaggleSync] Initiating asynchronous upload of Epoch {epoch+1} checkpoint..."
        )

        # Asynchronously trigger Kaggle CLI based on the environment
        is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

        if is_kaggle:
            kaggle_exe = shutil.which("kaggle")

        else:
            kaggle_path = (
                Path.home() / "anaconda3" / "envs" / "airflow_env" / "Scripts" / "kaggle.exe"
            )

            if not kaggle_path.exists():
                self.logger.error(f"Kaggle executable not found at: {kaggle_path}")
                raise FileNotFoundError(f"Kaggle CLI not found at {kaggle_path}")

            kaggle_exe = str(kaggle_path)

        cmd = (
            f'"{kaggle_exe}" datasets version -p "{self.checkpoint_dir}" '
            f"-m 'Auto-update Epoch {epoch + 1}' -r zip"
        )

        try:
            # Non-blocking process execution using DEVNULL to prevent OS buffer deadlocks
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self.logger.info(
                f"[KaggleSync] Upload process started successfully (PID: {process.pid})."
            )
        except Exception as e:
            self.logger.error(
                f"[KaggleSync] Failed to trigger Kaggle CLI dataset update: {str(e)}"
            )

    def get_config(self):
        """
        Returns the configuration of the callback for serialization.
        """
        config = super().get_config()
        config.update({
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_filename": self.checkpoint_filename,
            "dataset_id": self.dataset_id
        })
        return config
