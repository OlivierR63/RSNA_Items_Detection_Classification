# coding: utf-8
import os
import sys
import subprocess
import zipfile
import json
from pathlib import Path
import shutil
import logging

# Configure a simple logger for the local deployment process
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deploy_manager")

# --- PROJECT SETTING ---

# Kaggle dataset slug
SRC_DATASET_SLUG = "rsna-lumbar-spine-src-code"

# Must match the id in kernel-metadata.json
KERNEL_SLUG = "rsna-lumbar-spine-training"

# Kaggle logs and checkpoints dataset slug
LOGS_CHECKPOINTS_DATASET_SLUG = "rsna-lumbar-spine-logs-and-checkpoints"
USER_HOME = Path.home()

# Configure the Kaggle CLI executable path based on the environment or local installation
KAGGLE_EXE = (
    shutil.which("kaggle") or
    str(USER_HOME / "anaconda3" / "envs" / "airflow_env" / "Scripts" / "kaggle.exe")
)

# Configure environment variables to force UTF-8 locally
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"


def create_src_zip(root_dir: Path, zip_name: str = "my_src.zip"):
    """
    Creates a ZIP archive of the src folder, ignoring unnecessary files
    and directories, to be used as a Kaggle dataset.
    """
    root_dir.mkdir(parents=True, exist_ok=True)

    src_dir = root_dir.resolve() / "src"

    if not src_dir.exists():
        logger.error(f"Error: The directory {src_dir} does not exist.")
        sys.exit(1)

    tmp_dir = root_dir / "src_data" / "tmp_dataset"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    zip_path = tmp_dir / zip_name

    # No need for manual unlink() as 'w' mode overwrites existing files
    logger.info(f"Compressing the 'src' directory into {zip_name}")
    ignored_extensions = {'.pyc', '.pyo', '.git', '.ipynb_checkpoints'}
    ignored_dirs = {'__pycache__', '.git', '.vscode', 'logs', 'outputs'}

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

        for root, dirs, files in os.walk(src_dir):
            # Filter directories to ignore
            dirs[:] = [d for d in dirs if d not in ignored_dirs]

            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in ignored_extensions:
                    continue

                # Keep the relative path with respect to the project root
                arcname = file_path.relative_to(root_dir)
                zipf.write(file_path, arcname)

    logger.info("ZIP archive created successfully.")
    return zip_path


def generate_bootstrap_main(root_dir: Path):
    """
    Generates the minimal main.py file that will serve as the Kaggle trigger.
    """
    main_path = root_dir / "main.py"
    logger.info(f"Generating the dynamic startup script in {main_path}")

    # Generate bootstrap code block
    bootstrap_code = """
# coding: utf-8

import os
import sys
import zipfile
import shutil
import importlib
import logging
from pathlib import Path

# 1. Setup a fallback/bootstrap logger to log during the initial workspace setup.
BOOTSTRAP_LOGGER_NAME = "bootstrap_driver"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger(BOOTSTRAP_LOGGER_NAME)

# Identify environment
is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None


def scan_kaggle_inputs(input_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    \"\"\"
    Scans the Kaggle input directory to dynamically locate the source code
    and checkpoints of previous runs, while ignoring massive image datasets.
    \"\"\"
    zip_path, unzipped_src, previous_run_dir = None, None, None
    ignored_folders = {
        'rsna-2024-lumbar-spine-degenerative-classification',
        'rsna-lumbar-spine-tfrecords'
    }

    for root, dirs, files in os.walk(input_dir):
        # Prune giant folders in-place to prevent scanning millions of image files
        dirs[:] = [d for d in dirs if d not in ignored_folders]
        root_path = Path(root)

        # 1. Identify source code
        if "rsna-lumbar-spine-src" in root:
            if "my_src.zip" in files and not zip_path:
                zip_path = root_path / "my_src.zip"
                logger.info(f"Found source ZIP dynamically at: {zip_path}")

            if "src" in dirs and not unzipped_src:
                candidate_path = root_path / "src"
                if (candidate_path / "projects").exists() or (candidate_path / "core").exists():
                    unzipped_src = candidate_path
                    logger.info(f"Found source directory dynamically at: {unzipped_src}")

        # 2. Identify previous outputs (for Warm-Start continuous learning)
        output_dirs = ("input_data_inspection", "cache_metadata", "checkpoints", "logs")

        is_output_dir = any(folder in dirs for folder in output_dirs)
        is_target_root = "rsna-lumbar-spine-logs-and-checkpoints" in root

        if is_output_dir and is_target_root:
            logger.info(f"Found previous run outputs dynamically at: {root_path}")

    return zip_path, unzipped_src, root_path  # previous_run_dir


def prepare_source_code(
    zip_path: Path | None,
    unzipped_src: Path | None,
    dest_src: Path,
    working_dir: Path
):
    \"\"\"
    Extracts or copies the application code into the writable workspace.
    \"\"\"
    if zip_path and zip_path.exists():
        logger.info(f"Extracting source code from {zip_path} to {working_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(working_dir)

    elif unzipped_src and unzipped_src.exists():
        logger.info(f"Copying source code from {unzipped_src} to {dest_src}")
        if dest_src.exists():
            shutil.rmtree(dest_src) if dest_src.is_dir() else dest_src.unlink()
        shutil.copytree(unzipped_src, dest_src)

    else:
        logger.error("Execution Error: Neither source code ZIP nor folder was found.")


def restore_warm_start_data(previous_run_dir: Path | None, target_dir: Path):
    \"\"\"
    Copies historical model checkpoints and logs from a previous run to
    the current writable run workspace to enable continuous training.
    \"\"\"

    if not previous_run_dir or not previous_run_dir.exists():
        logger.info("No previous session history found. Starting training/logs from scratch.")
        return

    logger.info(f"Restoring session history from {previous_run_dir} to {target_dir}")
    try:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(previous_run_dir, target_dir)
        logger.info("Successfully restored previous model checkpoints and logs for Warm-Start!")
    except Exception as e:
        logger.warning(f"Failed to copy previous run outputs: {e}")


if is_kaggle:
    logger.info("Initializing Kaggle Driver Environment")

    # Define paths using pathlib
    working_path = Path("/kaggle/working")
    kaggle_input_path = Path("/kaggle/input/datasets/olivierrochat").resolve()

    default_zip = (
        kaggle_input_path / "rsna-lumbar-spine-src-code" /
        "src_data" / "tmp_dataset" / "my_src.zip"
    )
    default_src = kaggle_input_path / "rsna-lumbar-spine-src-code" / "src"

    target_src = working_path / "src"
    target_lumbar_spine = working_path / "output"

    # Step 1: Scan Kaggle inputs for essential directories
    zip_found, src_found, previous_run = scan_kaggle_inputs(kaggle_input_path)

    # Fallback to defaults if dynamic search did not find anything
    final_zip = zip_found or (default_zip if default_zip.exists() else None)
    final_src = src_found or (default_src if default_src.exists() else None)

    # Step 2: Deploy source code and warm-start data
    prepare_source_code(final_zip, final_src, target_src, working_path)
    restore_warm_start_data(previous_run, target_lumbar_spine)

    # Step 3: Register working directory in Python search path
    if str(working_path) not in sys.path:
        sys.path.insert(0, str(working_path))

    # Step 4: Promote to standard application logger if available
    try:
        # We clear the module cache and load dynamically to bypass static linter warnings,
        # pointing directly to the absolute package path.
        sys.modules.pop('src.core.utils.logger', None)
        logger_module = importlib.import_module("src.core.utils.logger")
        logger = logger_module.get_current_logger()
        logger.info("Successfully promoted to final application workspace logger.")

    except Exception as e:
        logger.warning(
            f"Could not load custom logger.py, staying on bootstrap logger. Reason: {e}"
        )

if __name__ == "__main__":
    module_path = "src.RSNA_2024_Lumbar_Spine_Degenerative_Classification"
    logger.info(f"Transitioning execution control to main module: {module_path}")

    try:
        app = importlib.import_module(module_path)
        app.main()
    except Exception as e:
        logger.error(f"Application crash detected: {e}", exc_info=True)
        raise e
"""
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(bootstrap_code)
    logger.info(f"File {main_path} successfully generated.")


def push_to_kaggle(root_dir: Path, zip_path: Path):
    """
    Manage the transfer of the ZIP Dataset and Kernel Script via Kaggle CLI.
    """
    # 1. Retrieve Kaggle username
    with open("kernel-metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    username = meta["id"].split("/")[0]
    dataset_id = f"{username}/{SRC_DATASET_SLUG}"

    # 2. Initialize or update the Kaggle Dataset which contains the source code
    tmp_dir = root_dir / "src_data" / "tmp_dataset"
    tmp_dir.mkdir(exist_ok=True, parents=True)
    dataset_id = f"{username}/{SRC_DATASET_SLUG}"
    ensure_dataset_exists(
        tmp_dir,
        SRC_DATASET_SLUG,
        dataset_id,
        update_dataset=True
    )

    # 3. Initialize the Kaggle history dataset (logs and checkpoints).
    # The dataset is left as-is if it already exists
    src_dir = root_dir / "src_data" / "dataset_init"
    src_dir.mkdir(exist_ok=True, parents=True)

    # The directory must store at least one file more than dataset-metadata.json
    # to prevent the kaggle from failing.
    # therefore, generates a dummy file into the folder
    Path(src_dir / "dummy_file.keep").touch()

    dataset_id = f"{username}/{LOGS_CHECKPOINTS_DATASET_SLUG}"
    ensure_dataset_exists(src_dir, LOGS_CHECKPOINTS_DATASET_SLUG, dataset_id, update_dataset=False)

    # 4. Push execution Kernel (main.py)
    logger.info("Pushing entry point and triggering remote execution on Kaggle")
    subprocess.run([KAGGLE_EXE, "kernels", "push", "-p", "."], check=True)
    logger.info("Everything transferred successfully. Training has been initiated.")


def ensure_dataset_exists(
    src_dir: Path,
    dataset_slug: str,
    dataset_id: str,
    update_dataset: bool = True
) -> None:
    """
    Ensure that the dataset-metadata.json file exists in the src directory.

    If the file is missing, it creates a new one with a default structure
    required for Kaggle dataset creation.

    Args:
        src_dir (Path): The path to the source directory where metadata should reside.
        dataset_slug (str): The slug identifier for the Kaggle dataset.
        dataset_id (str): The unique ID or owner/slug string for the dataset.
        update_dataset (bool) : If True, triggers a dataset version update on Kaggle
                                if the dataset already exists
    """
    metadata_path = src_dir / "dataset-metadata.json"

    try:
        if not metadata_path.exists():
            info_msg = (
                f"Dataset metadata not found in {src_dir}. Creating it now for slug: {dataset_slug}"
            )
            logger.info(info_msg)

            # Convert slug to readable title
            title = " ".join(
                [
                    word.upper() if word.lower() == "rsna" else word.capitalize()
                    for word in dataset_slug.replace("-", " ").split()
                ]
            )

            default_metadata = {
                "title": title,
                "id": dataset_id,
                "licenses": [{"name": "CC0-1.0"}]
            }

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(default_metadata, f, indent=4)

            logger.info(f"Successfully generated default metadata at {metadata_path}")

        else:
            # Print the metadata being used for the current deployment
            logger.info(f"Updating existing Dataset {dataset_id} on Kaggle")

        # Validate metadata content
        validate_metadata(metadata_path)
        exist = check_dataset_exists(dataset_id)

        if not exist:
            logger.info(
                f"Dataset {dataset_id} not found. Creating new dataset "
                f"under directory {src_dir}."
            )

            # Run the 'create' command logic
            # The '-u' flag (unzip) is used to ensure
            # that compressed files within the dataset are correctly extracted
            # by the Kaggle platform after upload.
            cmd = [KAGGLE_EXE, "datasets", "create", "-p", str(src_dir), "-u"]

        elif update_dataset is True:
            logger.info(f"Dataset {dataset_id} found. Updating version.")

            # Run the 'version' command logic
            cmd = [
                    KAGGLE_EXE,
                    "datasets",
                    "version",
                    "-p",
                    str(src_dir),
                    "-m",
                    "Auto-update source code",
                    "-r",
                    "zip"
                ]
        else:
            cmd = None
            logger.info(f"Dataset {dataset_id} found and kept as is.")

        if cmd is not None:
            logger.info(f"Pushing to kaggle with command {cmd}")
            subprocess.run(cmd, check=True)
            logger.info("Push completed successfully")

    except subprocess.CalledProcessError as e:
        logger.error(f"Kaggle CLI execution failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during deployment: {e}")
        sys.exit(1)


def validate_metadata(metadata_path: Path):
    """
    Strictly validate the dataset-metadata.json file structure.
    """
    required_keys = {"title", "id", "licenses"}

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check required top-level keys
        if not required_keys.issubset(data.keys()):
            raise ValueError(f"Metadata missing required keys: {required_keys - data.keys()}")

        # Check license format
        if not isinstance(data["licenses"], list) or len(data["licenses"]) == 0:
            raise ValueError("Licenses must be a non-empty list.")

        logger.info("Metadata validation successful.")
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Metadata validation failed: {e}")
        raise


def check_dataset_exists(dataset_id: str) -> bool:
    """
    Check if a Kaggle dataset exists using the CLI.
    dataset_id format: 'username/dataset-slug'
    """
    try:
        # We attempt to retrieve the metadata for the dataset
        # This will return a non-zero exit code if the dataset is not found
        subprocess.run(
            [KAGGLE_EXE, "datasets", "metadata", dataset_id, "-p", "."],
            check=True,
            capture_output=True,
            text=True
        )
        return True

    except subprocess.CalledProcessError:
        return False


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Pipeline execution
    zip_file = create_src_zip(PROJECT_ROOT)
    logger.info(f"ZIP Archive created: {zip_file}")

    generate_bootstrap_main(PROJECT_ROOT)
    logger.info("Function generate_bootstrap_main completed successfully")

    push_to_kaggle(PROJECT_ROOT, zip_file)
    logger.info("Function push_to_kaggle completed successfully")
