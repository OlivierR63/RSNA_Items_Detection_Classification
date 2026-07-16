# coding: utf-8

import os
import time
import zipfile
import tempfile
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
        logger: logging.Logger = None,
        dataset_id: str = "olivierrochat/rsna-lumbar-spine-logs-and-checkpoints"
    ):
        super().__init__()
        self._dataset_id = dataset_id
        self.logger = logger or get_current_logger()
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
        Keras callback hook triggered automatically upon training completion.

        Acts as a final synchronization barrier for the Kaggle dataset. This
        method ensures that any remaining asynchronous upload tasks are
        fully completed before the training process terminates, preventing
        potential data loss or truncated checkpoint uploads.

        Args:
            logs (dict, optional): Dictionary containing training metrics.
                Defaults to None.
        """
        self.logger.info("[KaggleSync] Training finished. Finalizing uploads...")
        self._finalize_uploads()

        self.logger.info("[KaggleSync] All background tasks are clean. Callback exiting.")

    def _update_logs_and_checkpoints_dataset(self, epoch: int) -> None:
        """
        Triggers an asynchronous Kaggle dataset update using a temporary staging area.

        This method prepares the dataset for synchronization by creating a flat
        staging directory. It handles metadata generation (if missing), compresses
        training artifacts (logs, checkpoints) into a single archive to avoid
        directory structure conflicts with the Kaggle CLI, and initiates the
        background upload process.

        Args:
            epoch (int): The current training epoch number, used to label
                the dataset version commit message.

        Returns:
            None: This method operates asynchronously and does not return
                a value.
        """
        if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is None:
            return

        try:
            self._wait_for_previous_upload()
            self.logger.info("[KaggleSync] Initiating upload via staging.")

            source_dir = Path(self._config['paths']['output']['read_write_dir']).resolve()

            with tempfile.TemporaryDirectory() as staging_dir:
                staging_path = Path(staging_dir)

                # Separation of concerns
                self._prepare_metadata(source_dir, staging_path)
                self._zip_source_content(source_dir, staging_path / "data.zip")

                # Running
                self._trigger_kaggle_cli(staging_path, epoch)

        except Exception as e:
            self.logger.error(f"[Kaggle Sync] Failed to update dataset: {str(e)}")

    def _wait_for_previous_upload(self) -> None:
        """
        Blocks execution until the active Kaggle CLI background process completes.

        Checks the status of the current '_upload_process'. If a process is
        currently running (poll returns None), the method enters a blocking
        loop, sleeping for 1 second intervals until the process finishes
        to ensure data consistency and prevent concurrent upload conflicts.
        """
        try:
            if self._upload_process and self._upload_process.poll() is None:
                self.logger.warning("[KaggleSync] Process active. Waiting...")
                while self._upload_process.poll() is None:
                    time.sleep(1)

        except Exception as e:
            self.logger.error(f"[KaggleSync] Error while waiting for process: {str(e)}")

    def _prepare_metadata(
        self,
        source_dir: Path,
        staging_path: Path
    ) -> None:
        """
        Prepares the 'dataset-metadata.json' file in the staging directory.

        This method attempts to copy the existing metadata file from the source
        directory. If the file is missing, it automatically generates a compliant
        default metadata file using the instance's dataset ID, ensuring that the
        Kaggle CLI has the necessary configuration to identify the target dataset.

        Args:
            source_dir (Path): The original directory containing the artifacts
                and potentially an existing metadata file.
            staging_path (Path): The temporary staging directory where the
                metadata must be placed for the upload process.

        Returns:
            None
        """
        try:
            json_src = source_dir / "dataset-metadata.json"
            json_dst = staging_path / "dataset-metadata.json"

            if json_src.exists():
                shutil.copy(json_src, json_dst)
            else:
                # Generate metadata file
                metadata = {
                    "title": "RSNA Lumbar Spine Logs And Checkpoints",
                    "id": self._dataset_id,
                    "licenses": [{"name": "CC0-1.0"}]
                }
                with open(json_dst, "w") as f:
                    json.dump(metadata, f, indent=4)

        except (OSError, IOError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to prepare metadata: {e}")

    def _zip_source_content(
        self,
        source_dir: Path,
        zip_path: Path
    ) -> None:
        """
        Compresses source directory contents into a single archive for upload.

        Iterates through the source directory and recursively zips all files
        and subdirectories, excluding existing metadata and the archive itself.
        This approach ensures a flat and clean archive structure, which is
        required to prevent directory nesting conflicts when interacting
        with the Kaggle CLI.

        Args:
            source_dir (Path): The root directory containing training
                artifacts (logs, checkpoints, etc.) to be compressed.
            zip_path (Path): The destination path for the created ZIP archive.

        Returns:
            None
        """
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for item in source_dir.iterdir():
                    if item.name not in ["dataset-metadata.json", "data.zip"]:
                        if item.is_dir():
                            for file in item.rglob('*'):
                                zipf.write(file, file.relative_to(source_dir))
                        else:
                            zipf.write(item, item.name)

        except (OSError, zipfile.BadZipFile) as e:
            raise RuntimeError(f"Compression failed: {e}")

    def _trigger_kaggle_cli(self, staging_path: Path, epoch: int):
        """
        Executes the Kaggle CLI asynchronously to update the remote dataset.

        Constructs and triggers a subprocess command using the Kaggle CLI.
        The method uses '--dir-mode tar' to ensure the staging directory,
        containing the 'dataset-metadata.json' and the 'data.zip' archive,
        is correctly interpreted by Kaggle as a single dataset version update.

        Args:
            staging_path (Path): The path to the staging directory containing
                the prepared metadata and zipped content.
            epoch (int): The current training epoch, used to generate a
                descriptive version message for the dataset.

        Returns:
            None: The process is managed via 'self._upload_process' for
                asynchronous monitoring.
        """
        cmd = [
            shutil.which("kaggle"), "datasets", "version",
            "-p", str(staging_path),
            "-m", f"Auto-update Epoch {epoch + 1}",
            "--dir-mode", "tar"
        ]
        self.logger.info(f"[KaggleSync] Triggering Kaggle update. Command: {' '.join(cmd)}")

        try:
            # We use Popen, but we will inspect the result via communicate()
            self._upload_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # This is non-blocking for the whole script because it's called
            # within the async flow, but allows us to capture the exit status.
            stdout, stderr = self._upload_process.communicate()

            if self._upload_process.returncode != 0:
                self.logger.error(
                    f"[KaggleSync] Kaggle CLI failed (Code {self._upload_process.returncode})"
                )
                self.logger.error(f"[KaggleSync] STDERR: {stderr.strip()}")
                self.logger.error(f"[KaggleSync] STDOUT: {stdout.strip()}")
            else:
                self.logger.info("[KaggleSync] Upload request sent successfully.")

        except Exception as e:
            self.logger.error(f"[KaggleSync] Exception during Kaggle CLI trigger: {str(e)}")

    def _finalize_uploads(self, timeout: int = 300) -> None:
        """
        Synchronizes and closes any pending asynchronous Kaggle upload process.

        This method monitors the active background process, waiting for it to
        complete within a defined timeframe. If the process exceeds the timeout,
        it is forcefully terminated to prevent the training job from hanging
        indefinitely. Finally, it validates the exit status to report success
        or failure of the synchronization.

        Args:
            timeout (int): Maximum time in seconds to wait for the upload
                process to complete before termination. Defaults to 300.

        Returns:
            None
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
