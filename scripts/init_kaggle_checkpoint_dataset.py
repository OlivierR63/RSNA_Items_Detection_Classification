# coding: utf-8

import json
import subprocess
from pathlib import Path
from src.core.utils.logger import get_current_logger


def create_kaggle_checkpoint_dataset() -> None:
    """
    Initializes and creates the private Kaggle dataset for model checkpoints.
    """
    logger = get_current_logger()

    init_dir = Path("metadata_init")
    metadata_file = init_dir / "dataset-metadata.json"
    dummy_file = init_dir / "placeholder.txt"

    # Absolute path to the Kaggle executable in your conda environment
    kaggle_exe = r"C:\Users\Olivier\anaconda3\envs\airflow_env\Scripts\kaggle.exe"

    # 1. Create a clean temporary directory for initialization
    logger.info(f"Creating initialization directory at: {init_dir}")
    init_dir.mkdir(parents=True, exist_ok=True)

    # 2. Generate a placeholder file (Kaggle requires at least one file to create a dataset)
    with open(dummy_file, "w", encoding="utf-8") as f:
        f.write(
            "This placeholder prevents Kaggle CLI from failing due to an empty folder."
        )

    # 3. Define the metadata structure matching your Kaggle account
    metadata = {
        "title": "RSNA Lumbar Spine Checkpoints",
        "id": "olivierrochat/rsna-lumbar-spine-checkpoints",
        "licenses": [{"name": "CC0-1.0"}],
    }

    # 4. Write the dataset-metadata.json file
    logger.info(f"Writing metadata configuration to {metadata_file}")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 5. Execute Kaggle CLI command to create the dataset as private
    logger.info("Triggering Kaggle CLI to create the private dataset...")
    cmd = f"{kaggle_exe} datasets create -p {init_dir} -r zip"

    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        logger.info(f"Kaggle CLI Output:\n{result.stdout}")
        print(
            f"[SUCCESS] Private dataset successfully registered on Kaggle.\n"
            f"ID: {metadata['id']}"
        )

    except subprocess.CalledProcessError as e:
        logger.error(f"Kaggle CLI execution failed: {e.stderr}")
        print(
            f"[ERROR] Failed to create dataset. Ensure your Kaggle API credentials "
            f"are loaded in your environment variables.\nDetails: {e.stderr}"
        )

    finally:
        # 6. Clean up temporary initialization files locally
        if dummy_file.exists():
            dummy_file.unlink()
        if metadata_file.exists():
            metadata_file.unlink()
        if init_dir.exists():
            init_dir.rmdir()
        logger.info("Temporary initialization directory cleaned up.")


if __name__ == "__main__":
    create_kaggle_checkpoint_dataset()
