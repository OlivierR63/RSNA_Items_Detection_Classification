# coding: utf-8

import os
import time
import subprocess
import tf_keras
import shutil
import json
import logging
from pathlib import Path
from src.core.utils.logger import get_current_logger
from typing import Any


class KaggleDatasetCheckpointSyncCallback(tf_keras.callbacks.Callback):
    """
    Callback to automatically push saved checkpoints to a private Kaggle Dataset
    at the end of each epoch to prevent data loss from Timeouts.
    """

    def __init__(
        self,
        config: dict[str, Any],
        logs: logging.Logger,
        dataset_id: str = "olivierrochat/rsna-lumbar-spine-logs_and_checkpoints"
    ):
        super().__init__()
        self._dataset_id = dataset_id
        self.logger = get_current_logger()
        self._config = config
        self._username = dataset_id.split("/")[0]
        self._upload_process: subprocess.Popen | None = None
        self._log_path = None

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """
        Triggered at the end of every training epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict | None): A dictionary containing metric results for this epoch
                (e.g., 'loss', 'accuracy'). This argument is a mandatory requirement
                of the Keras Callback API, allowing the callback to access
                and potentially log or track these performance metrics during
                the training process. Defaults to None.
        """
        logs = logs or {}

        # Best practice : always call the method of the parent class
        super().on_epoch_end(epoch, logs)

        # Sync to Kaggle dataset at the end of each epoch
        self._update_logs_and_checkpoints_dataset(epoch)

    def on_train_end(self, logs=None) -> None:
        """
        Keras hook triggered automatically at the very end of training.
        Ensures all pending Kaggle uploads are finished before exiting.
        """
        self.logger.info("[KaggleSync] Training finished. Finalizing uploads...")
        self._finalize_uploads()

        self.logger.info("[KaggleSync] All background tasks are clean. Callback exiting.")

    def _update_logs_and_checkpoints_dataset(self, epoch: int) -> None:
        """
        Triggers the Kaggle CLI asynchronously. Checks if the previous
        upload process has finished before starting a new one.
        """
        # Only proceed if running within a Kaggle Kernel
        if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is None:
            return

        # Redirects the logs to the mirror log file:
        if self._log_path is None:
            self._log_path = Path(self._config['paths']['output']['read_write_dir'])

        # Prevent concurrent uploads if the previous one is still running
        if self._upload_process is not None:
            if self._upload_process.poll() is None:
                self.logger.warning(
                    "[KaggleSync] Process active. Waiting for completion..."
                )

                # Polling loop: check every second
                while self._upload_process.poll() is None:
                    time.sleep(1)

        self.logger.info("[KaggleSync] Initiating upload.")

        root_dir_path = Path(self._config['paths']['output']['read_write_dir']).resolve()
        json_file = root_dir_path / "dataset-metadata.json"

        # Ensure metadata file is present
        if not json_file.exists():
            title = " ".join([
                word.upper() if word.lower() == "rsna" else word.capitalize()
                for word in self._dataset_id.split('/')[0].replace("-", " ").split()
            ])
            metadata = {
                "title": title,
                "id": self._dataset_id,
                "licenses": [{"name": "CC0-1.0"}]
            }
            with open(json_file, "w") as f:
                json.dump(metadata, f, indent=4)

        # Build command: use -r zip to optimize upload speed
        cmd = [
            shutil.which("kaggle"), "datasets", "version",
            "-p", str(root_dir_path),
            "-m", f"Auto-update Epoch {epoch + 1}",
            "-r", "zip"
        ]

        try:
            # Use Popen to run asynchronously
            with open(self._log_path, "a") as log_file:
                self._upload_process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    text=True
                )

            # You might want to track process completion in a separate thread/hook
            # or accept that it runs in the background.
            self.logger.info(
                f"[KaggleSync] Async upload started (PID: {self._upload_process.pid})."
            )

        except Exception as e:
            self.logger.error(f"[KaggleSync] Failed to trigger Kaggle CLI: {str(e)}")

    def _finalize_uploads(self, timeout: int = 300) -> None:
        """
        Ensures all background uploads are completed by polling the process state.
        """
        if self._upload_process is not None:
            start_time = time.time()

            # Check if the process is still running
            if self._upload_process.poll() is None:
                self.logger.info("[KaggleSync] Waiting for pending upload to finish...")

                # Polling loop: check every second
                while self._upload_process.poll() is None:
                    if time.time() - start_time > timeout:
                        self.logger.error("[KaggleSync] Timeout reached. Killing upload process.")
                        self._upload_process.kill()
                        break
                    time.sleep(1)

                self.logger.info("[KaggleSync] Pending upload finished.")

            # Check return code for errors
            exit_code = self._upload_process.returncode
            if exit_code != 0:
                self.logger.error(
                    f"[KaggleSync] Upload failed with return code {exit_code}"
                    "Data may not be synced."
                )
            else:
                self.logger.info("[KaggleSync] Upload confirmed successful (Return Code: 0).")

            self._upload_process = None

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the callback for serialization.
        """
        config = super().get_config()
        config.update({
            "dataset_id": self._dataset_id
        })
        return config
