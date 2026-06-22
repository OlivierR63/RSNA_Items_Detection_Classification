# coding: utf-8

import subprocess
from pathlib import Path
import shutil
import os

# --- Local Path Configurations (Windows) ---
# Root directory of your local Git project
base_project_path = Path(
    r"C:\Users\Olivier\Desktop\Projet_Kaggle\RSNA_Items_Detection_Classification"
)

# Directory containing your 'run_rsna.ipynb' notebook and 'kernel-metadata.json' config file
kaggle_run_dir = base_project_path / "Kaggle_run"


def prepare_and_push_to_kaggle():
    """
    Configures the local application for Kaggle execution and
    triggers the push via the Kaggle CLI.
    """
    print("-" * 60)
    print("1. Preparing Kaggle configuration file...")
    print("-" * 60)

    # Define YAML configuration file paths
    kaggle_config_file = base_project_path / "src/config/lumbar_spine_config_kaggle.yaml"
    generic_config_file = base_project_path / "src/config/lumbar_spine_config.yaml"

    # Safety check: Ensure the source Kaggle configuration file exists
    if not kaggle_config_file.exists():
        print(f"Error: The file {kaggle_config_file} could not be found.")
        return

    # To prevent administrator privilege issues with symbolic links on Windows,
    # the generic configuration file is directly replaced with a copy of the Kaggle version.
    if generic_config_file.exists() or generic_config_file.is_symlink():
        generic_config_file.unlink()  # Remove the existing file or symlink

    shutil.copy(kaggle_config_file, generic_config_file)
    print(
        f"Configuration synchronized: {generic_config_file.name} "
        f"is now a copy of {kaggle_config_file.name}"
    )

    print("\n" + "-" * 60)
    print("2. Executing kernel push to Kaggle servers...")
    print("-" * 60)

    # Set up the CLI command arguments as a single clean string
    command_str = f'python -m kaggle kernels push -p "{kaggle_run_dir}"'

    print(f"command = {command_str}")

    env = os.environ.copy()

    # Paste your raw key value from kaggle.json here
    env["KAGGLE_API_TOKEN"] = "KGAT_a525e5127e52f44d4d8d1cfabe46a69e"

    # Using shell=True is the standard way when passing arguments as a string
    try:
        _ = subprocess.run(
            command_str,
            shell=True,
            check=True,
            env=env  # Forces the subprocess to use this token
            # capture_output=True and text=True are removed to unblock console streaming
        )
        print("\nApplication successfully pushed! Remote training session has started.")

    except subprocess.CalledProcessError as e:
        print("\n Error encountered during Kaggle kernel push:")
        print(f"The Kaggle CLI failed with exit status {e.returncode}.")
        print("Please check the lines right above to see Kaggle's explicit error message.")


if __name__ == "__main__":
    prepare_and_push_to_kaggle()
