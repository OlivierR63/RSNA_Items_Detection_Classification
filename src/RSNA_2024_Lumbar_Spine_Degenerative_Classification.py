# coding: utf-8

# ----------------- Standard imports --------------------------------------------------------
import signal
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import gc
import json
import os
import tf_keras

from src.projects.lumbar_spine.csv_metadata_handler import CSVMetadataHandler
from src.projects.lumbar_spine.model_trainer import ModelTrainer
from src.core.models.model_factory import ModelFactory
from src.projects.lumbar_spine.RSNA_lumbar_losses_and_metric import RSNALossAndMetricProvider


# ----------------- TensorFlow and projects imports -----------------------------------------
def setup_config_symlink(config_dir_path: str) -> None:
    """
    Automates the creation of a symbolic link for the configuration file
    based on the execution environment (Kaggle vs. Windows).

    Args:
        config_dir_path (str): Absolute path to the directory containing YAML files.
    """
    config_dir = Path(config_dir_path).resolve()
    main_config = config_dir / "lumbar_spine_config.yaml"

    # Identify the environment
    is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

    if is_kaggle:
        target_name = "lumbar_spine_config_kaggle.yaml"
    else:
        target_name = "lumbar_spine_config_windows.yaml"

    target_file = config_dir / target_name

    # Safety check: does the target source file exist?
    if not target_file.exists():
        print(f"Warning: Source file {target_file} not found. Symlink skipped.")
        return

    # Create or update the symlink
    try:
        if main_config.is_symlink() or main_config.exists():
            main_config.unlink()  # Remove existing file or broken link

        # Create the link (pointing from main_config to target_file)
        main_config.symlink_to(target_file)

    except Exception as e:
        print(f"Failed to create symlink: {e}")


def handle_interrupt(signum, frame):
    """
    Handles interrupt signals (Ctrl+C) to ensure proper log closure.
    """
    from src.core.utils.logger import get_current_logger

    try:
        logger = get_current_logger()
        logger.info("\nInterruption detected (Ctrl+C). Exiting gracefully...")

    except RuntimeError:
        print("\nInterruption detected (Ctrl+C). Exiting gracefully...")

    sys.exit(0)


def _validate_input_params(
    depth: int,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Validates dynamic parameters and ensures correct data types for model building.
    """
    # 1. Depth validation (depends on runtime input, not static YAML)
    if depth is None or depth <= 0:
        msg = f"Invalid series depth: {depth}. Must be a positive integer."
        logger.critical(msg, exc_info=True, extra={"status": "failure"})
        raise ValueError(msg)

    # 2. Type validation (Preventive casting)
    # Even if keys exist, ensure content is numerically exploitable
    try:
        _ = float(config['optimizer']['learning_rate'])
        _ = int(config['data_specs']['max_records_per_frame'])
    except (ValueError, TypeError) as e:
        msg = f"Configuration type validation failed (expected numeric values): {e}"
        logger.critical(msg)
        raise ValueError(msg)


def _get_target_checkpoint(config: dict[str, str | dict]) -> Tuple[Path | None, str]:
    """
    Determines the checkpoint file to load based on the resume mode policy.

    The function checks for the existence of model files in the following order:
    1. If mode is 'best', it looks for the best performing model file.
    2. Fallback (or if mode is 'last'): it looks for the most recent checkpoint.
    3. If no files exist, it returns None for the path.

    Args:
        config (dict): Global configuration containing path and callback settings.

    Returns:
        Tuple[Optional[Path], str]: A tuple containing the resolved Path to the
            checkpoint (or None) and the selected resume mode string.
    """
    # Select the checkpoint loading policy: 'best' for the lowest validation loss
    # or 'last' to continue from the most recent epoch.
    checkpoint_dir = Path(config["paths"]["checkpoint"]).resolve()
    resume_mode = config["callbacks"]["resume_mode"]

    # Define filenames based on your ModelTrainer saving logic
    best_path = checkpoint_dir / ModelTrainer.BEST_MODEL_FILENAME
    last_path = checkpoint_dir / ModelTrainer.CHECKPOINT_FILENAME

    # Select the target file
    if resume_mode == "best" and best_path.is_file():
        checkpoint_path = best_path
    elif last_path.is_file():
        checkpoint_path = last_path
    else:
        checkpoint_path = None

    return checkpoint_path, resume_mode


def _load_existing_model(
    checkpoint_path: Path | None,
    mode: str,
    logger: logging.Logger
) -> tf_keras.Model | None:
    """
    Attempts to load a complete Keras model from a given checkpoint path.

    This method clears the current Keras session to prevent memory leaks and
    attempts to restore the full model state (architecture, weights, and
    optimizer). If the file is missing or serialization fails (common with
    custom Lambda layers), it fails gracefully and returns None.

    Args:
        checkpoint_path (Path | None): The resolved path to the .keras file.
        mode (str): The resume mode ('best' or 'last') for logging purposes.
        logger (logging.Logger): Logger for tracking the restoration process.

    Returns:
        tf_keras.Model | None: The loaded model if successful, else None.
    """

    if not checkpoint_path:
        return None

    try:
        info_msg = (
            f"Existing model found ({mode} mode) at {checkpoint_path}. "
            "Loading ..."
        )
        logger.info(info_msg)
        tf_keras.backend.clear_session()
        model = tf_keras.models.load_model(
            checkpoint_path,
            custom_objects=ModelFactory.CUSTOM_OBJECTS,
            safe_mode=False,
        )
        info_msg = (f"Model successfully restored from {checkpoint_path}.")
        logger.info(info_msg)
        return model

    except Exception as e:
        warning_msg = (
            f"Full restore failed: {e}. "
            "Switching to weight salvage logic."
        )
        logger.warning(warning_msg, exc_info=True)
        return None


def _build_fresh_or_salvage(
    depth: int,
    config: dict,
    checkpoint_path: Path | None,
    logger: logging.Logger
) -> 'tf_keras.Model':

    """
    Builds a new model architecture and attempts to restore weights if possible.

    This function acts as a fail-safe. It first builds a fresh model from the
    ModelFactory. If a checkpoint path is provided, it attempts to 'salvage'
    the weights using skip_mismatch=True, which allows loading weights even
    if the architecture has slightly changed.

    Args:
        depth (int): The series depth for 3D input layers.
        config (dict): Global configuration dictionary.
        checkpoint_path (Path | None): Path to the existing weights file.
        logger (logging.Logger): Logger for tracking the build process.

    Returns:
        tf_keras.Model: A model instance, either fresh or partially restored.

    Raises:
        RuntimeError: If the model cannot be built from the factory.
    """

    try:
        max_records = config['data_specs']['max_records_per_frame']
        checkpoint_dir = Path(config["paths"]["checkpoint"]).resolve()

        factory_model = ModelFactory(
            series_depth=depth,
            logger=logger,
            nb_output_records=max_records
        )
        model = factory_model.build_multi_series_model()

        # The model is now built, so factory_model can be released now.
        del factory_model
        gc.collect()

    except Exception as e:
        msg_critical = f"Fatal error. Failed to build new model: {e}"

        logger.critical(
            msg_critical,
            extra={"status": "failure", "error": str(e)},
            exc_info=True
        )

        raise RuntimeError(msg_critical)

    try:
        if checkpoint_path:
            model.load_weights(checkpoint_path, skip_mismatch=True)
            logger.info("Weights successfully salvaged with skip_mismatch=True")

            new_file = checkpoint_dir / "model_restored_fixed.keras"
            model.save(new_file)
            logger.info(
                "Model saved as 'model_restored_fixed.keras'",
                extra={"status": "recovered"}
            )

    except Exception as e_weights:
        msg_warning = (
            f"Warning:  Unable to restore the weights. Falling back to factory. {e_weights}"
        )
        logger.warning(
            msg_warning,
            extra={"status": "restart", "warning": str(e_weights)},
            exc_info=True
        )

    return model


def _finalize_and_compile_model(
    model: 'tf_keras.Model',
    config: dict,
    logger: logging.Logger
) -> 'tf_keras.Model':

    """
    Configures multi-task losses, weights, and metrics, then compiles the model.

    This function sets up a specialized compilation strategy:
    - Categorical Cross-Entropy (via provider) for 'severity_output'.
    - Mean Squared Error (MSE) for 'location_output' to penalize outliers.
    - Custom loss weighting to balance classification and coordinate regression.

    Args:
        model (tf_keras.Model): The built Keras model instance.
        config (dict): Global configuration dictionary.
        logger (logging.Logger): Logger for compilation status tracking.

    Returns:
        tf_keras.Model: The compiled Keras model.

    Raises:
        Exception: Re-raises any exception encountered during compilation.
    """

    try:
        provider = RSNALossAndMetricProvider(logger=logger)

        # --- Define Losses ---
        # Using MSE for location to penalize large spatial errors more heavily

        losses = {
            "severity_output": provider.get_loss(),
            "location_output": "mse"
        }

        # --- Define Loss Weights ---
        # We give more weight to location (5.0) because the coordinate regression
        # is currently struggling to converge compared to classification.

        loss_weights_settings = config["compilation"]['loss_weights']

        loss_weights = {
            "severity_output": loss_weights_settings["severity_output"],
            "location_output": loss_weights_settings["location_output"]
        }

        # --- Define Metrics ---
        metrics = {
            # Added accuracy for a quick baseline
            "severity_output": [provider.get_metrics(), "accuracy"],
            "location_output": [tf_keras.metrics.MeanAbsoluteError(name="mae")],
            "study_id_output": None
        }

        # --- Model Compilation ---
        # Adam optimizer with gradient clipping to prevent exploding gradients
        # during multi-task learning.

        optim_type = config["optimizer"]['type'].lower()
        run_eagerly_val = config['compilation']['run_eagerly']

        model.compile(
            optimizer=optim_type,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            run_eagerly=run_eagerly_val
        )

        model.summary()
        logger.info(
            "New model compiled with weighted losses.",
            extra={
                "optimizer": optim_type,
                "learning_rate": config["optimizer"]["learning_rate"],
                "clip_norm": config["optimizer"]["clipnorm"]
            }
        )
        return model

    except Exception as e:
        critical_msg = f"Fatal error : {e}"

        logger.critical(
            critical_msg,
            exc_info=True,
            extra={"status": "failure"}
        )

        raise e


def _get_or_build_model(
    depth: int,
    config: dict,
    logger: logging.Logger
) -> 'tf_keras.Model':

    """
    Retrieves a pre-existing model from a checkpoint or initializes a new one,
    with an automated weight restoration fallback.

    This function implements a robust "Resume Training" logic designed to handle
    structural breaking changes (e.g., Lambda layer serialization issues).
    It operates in three priority stages:
    1. Full Restore: Attempts to load the complete Keras model
       (architecture + weights + optimizer state).
    2. Weight Salvage: If a full load fails due to serialization errors, it builds a fresh
       architecture from the ModelFactory and injects existing weights by name, then saves
       a "fixed" version of the model.
    3. Fresh Start: If no checkpoint exists, it initializes and compiles a new model from scratch.

    Args:
        depth (int): The maximum number of slices per imaging series,
            used for 3D convolution input shapes.
        config (dict): Global configuration dictionary containing paths,
            hyperparameters, and model settings.
        logger (logging.Logger): Logger instance for status
            tracking and error reporting.

    Returns:
        tf_keras.Model: A compiled TensorFlow Keras model ready for training,
            potentially restored from a previous state.

    Raises:
        ValueError: If the series depth is invalid (None or <= 0) or if critical
            configuration variables are missing.
        Exception: If a fatal error occurs during model factory building or compilation.
    """

    # Initial validation
    _validate_input_params(depth, config, logger)
    checkpoint_path, mode = _get_target_checkpoint(config)
    model = _load_existing_model(checkpoint_path, mode, logger)

    if model is None:
        model = _build_fresh_or_salvage(depth, config, checkpoint_path, logger)

    return _finalize_and_compile_model(model, config, logger)


def main():
    """
        Main function to load configuration, set up the TensorFlow dataset pipeline,
        load and compile the 3D model, and start the training process.
    """

    # Define the policy: float16 for computations, float32 for critical variables.

    # Select the proper Yaml config file, depending on the current platform : WINDOWS or Kaggle
    setup_config_symlink("src/config")

    # 1. System initialization
    from src.config.config_loader import ConfigLoader
    config_loader = ConfigLoader("src/config/lumbar_spine_config.yaml")
    config: dict = config_loader.get()

    # Threads settings
    system_cfg = config["system"]
    nb_cores_config = system_cfg['nb_cores']
    is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
    cpu_threads = nb_cores_config if is_kaggle else 7

    os.environ["TF_USE_LEGACY_KERAS"] = '1'
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 2. Allow to parallelize operations on the requested number of cores
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(cpu_threads)
    except RuntimeError:
        # Unable to change if TF already initialized, but carry on the process
        print("TF Context already initialized, impossible to update threading.")

    # Project imports
    from src.core.utils.logger import setup_logger
    from src.core.utils.clean_logs import clean_old_logs
    from src.projects.lumbar_spine.model_trainer import ModelTrainer
    from src.projects.lumbar_spine.tfrecord_files_manager import TFRecordFilesManager

    # Apply global TF settings from the config
    run_eagerly_val = config['compilation']['run_eagerly']

    tf.config.run_functions_eagerly(run_eagerly_val)
    tf.data.experimental.enable_debug_mode()  # Optional but helpful for tf.data

    # Mute Python-level TensorFlow/Keras warnings
    # This handles warnings such as deprecated 'tf.placeholder' calls within internal Keras modules
    tf.get_logger().setLevel(logging.ERROR)

    # Setup signal handler for graceful interruption
    signal.signal(signal.SIGINT, handle_interrupt)

    # ---------------- Environmental configuration (before any TF import) -----------------------
    # Suppress TensorFlow C++ environmental logs
    #   '0' = all logs (default),
    #   '1' = filter INFO,
    #   '2' = filter INFO & WARNING,
    #   '3' = filter all including ERROR

    # Disable oneDNN custom operations messages
    # This prevents warnings regarding slight numerical differences due to floating-point round-off
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    paths_cfg = config['paths']

    # 2. Initialize logger with process-specific context
    log_dir = Path(paths_cfg["output"]) / "logs"

    # Redirect terminal stdout and stderr to a log file
    from src.core.utils.system_stream_tee import SystemStreamTee
    terminal_log_path = str(log_dir / "terminal_output.log")

    SystemStreamTee(terminal_log_path)

    with setup_logger(
        process_name="train",
        log_dir=log_dir
    ) as logger:

        # The logger is now set up available globally via get_current_logger().
        # It will automatically close at the end of this block.
        logger.info(f"Configuration loaded successfully. Loaded_values: {config}")

        try:
            if config["series_depth"] is None:
                data_specs_cfg = config["data_specs"]
                percentile_str = data_specs_cfg["series_depth_percentile"]

                if "tfrecord" in paths_cfg and "dicom_studies" in paths_cfg:
                    series_depth = config_loader.get_series_depth(
                        tfrecord_cache_dir=paths_cfg["tfrecord_metadata_cache"],
                        dicom_studies_dir=paths_cfg["dicom_studies"],
                        percentile=float(percentile_str),
                        logger=logger
                    )
                    config_loader.set_value("series_depth", series_depth)

            else:
                series_depth = config_loader.get_value("series_depth")

            info_msg = (
                "Max TFRecord files depth (number of DICOM files per series): "
                f"{series_depth}"
            )

            logger.info(info_msg)

            # Clear any background memory from the model building phase
            # to free up RAM before the data pipeline starts prefetching.
            tf_keras.backend.clear_session()

            # Instantiate the singleton CSVMetadataHandler for further use:
            _ = CSVMetadataHandler(
                logger=logger,
                dicom_studies_dir=paths_cfg["dicom_studies"],
                **paths_cfg["csv"]
            )

            # Load the compiled model if it already exists. In the other case, generate
            # a new model and compile it.
            model: tf_keras.Model = _get_or_build_model(series_depth, config, logger)

            logger.info(
                "Starting training process.",
                extra={"status": "started", "log_dir": log_dir}
            )

            # Extract DICOM images and metadata from source directories
            # and serialize them into dedicated TFRecord files (one per patient)
            tfrecord_files_manager = TFRecordFilesManager(logger)
            actual_nb_tfrecord_files = tfrecord_files_manager.generate_tfrecord_files()

            # Store the number of tfrecord_files in the cache file
            tfrecord_cache_dir = Path(paths_cfg["tfrecord_metadata_cache"])
            cache_path = tfrecord_cache_dir / "cache.json"

            with open(cache_path, 'r', encoding='utf-8') as f:
                try:
                    cache_data = json.load(f)

                except json.JSONDecodeError:
                    # In case the file is empty or corrupted
                    cache_data = {}

            # Add the new key / value pair:
            cache_data['actual_nb_tfrecord_files'] = actual_nb_tfrecord_files

            # Rewrite the cache file:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=4)

            logger.info(f"Cache successfully updated in {cache_path}")

            trainer = ModelTrainer(
                config=config,
                logger=logger,
                model=model,
                model_depth=series_depth
            )

            trainer.prepare_training_and_validation_datasets()
            trainer.train_model()

            # Force Keras to close properly before the end of the script
            tf_keras.backend.clear_session()
            gc.collect()

        except Exception as e:
            critical_msg = f"Critical error during training: {str(e)}"
            logger.critical(
                critical_msg,
                exc_info=True,
                extra={"status": "failure", "error": str(e)}
            )
            raise e

        finally:
            logger.info("Training process completed. Log file will be closed automatically.",
                        extra={"status": "completed"})

        # Remove log files older than 30 days
        log_retention_days = system_cfg['log_retention_days']

        clean_old_logs(days=int(log_retention_days))


if __name__ == "__main__":
    main()
