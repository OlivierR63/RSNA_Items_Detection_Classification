# coding: utf-8

# ----------------- Standard imports --------------------------------------------------------
from typing import TYPE_CHECKING
import signal
import sys
import logging
from pathlib import Path
import gc
import json
import os

if TYPE_CHECKING:
    import tensorflow as tf


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
    from . import get_current_logger

    try:
        logger = get_current_logger()
        logger.info("\nInterruption detected (Ctrl+C). Exiting gracefully...")

    except RuntimeError:
        print("\nInterruption detected (Ctrl+C). Exiting gracefully...")

    sys.exit(0)


def get_or_build_model(
    depth: int,
    config: dict,
    logger: logging.Logger
) -> 'tf.keras.Model':

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
        tf.keras.Model: A compiled TensorFlow Keras model ready for training,
            potentially restored from a previous state.

    Raises:
        ValueError: If the series depth is invalid (None or <= 0) or if critical
            configuration variables are missing.
        Exception: If a fatal error occurs during model factory building or compilation.
    """

    import tensorflow as tf
    from src.projects.lumbar_spine.model_trainer import ModelTrainer
    from src.core.models.model_factory import ModelFactory
    from src.projects.lumbar_spine.RSNA_lumbar_losses_and_metric import (
        rsna_weighted_log_loss,
        RSNAKaggleMetric
    )

    if depth is None or depth <= 0:
        critical_msg = (
            "Function get_or_build_model failed. Null depth parameter"
        )
        logger.critical(
            critical_msg,
            exc_info=True,
            extra={"status": "failure"}
        )
        raise ValueError(critical_msg)

    # Extract configs locally within the function to ensure independence
    try:
        optimizer_cfg = config['optimizer']
        compilation_cfg = config['compilation']
        callbacks_cfg = config['callbacks']

        data_specs_cfg = config['data_specs']
        max_records = int(data_specs_cfg['max_records_per_frame'])

        learning_rate = float(optimizer_cfg['learning_rate'])
        clip_norm = float(optimizer_cfg['clipnorm'])

    except (ValueError, TypeError) as e:
        critical_msg = f"Configuration type error: {e}"
        logger.critical(
            critical_msg,
            exc_info=True,
            extra={"status": "failure"}
        )
        raise ValueError(critical_msg)

    # Select the checkpoint loading policy: 'best' for the lowest validation loss
    # or 'last' to continue from the most recent epoch.
    checkpoint_dir = Path(config["paths"]["checkpoint"]).resolve()
    resume_mode = callbacks_cfg["resume_mode"]

    # Define filenames based on your ModelTrainer saving logic
    best_path = checkpoint_dir / ModelTrainer.BEST_MODEL_FILENAME
    last_path = checkpoint_dir / ModelTrainer.CHECKPOINT_FILENAME

    # Select the target file
    if resume_mode == "best" and best_path.is_file():
        checkpoint_full_path = best_path
    elif last_path.is_file():
        checkpoint_full_path = last_path
    else:
        checkpoint_full_path = None

    if checkpoint_full_path:
        info_msg = (
            f"Existing model found ({resume_mode} mode) at {checkpoint_full_path}. "
            "Loading for resume..."
        )
        logger.info(info_msg)

        try:
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(
                checkpoint_full_path,
                custom_objects=ModelFactory.CUSTOM_OBJECTS,
                safe_mode=False,
            )
            info_msg = ("Model loading completed.")
            logger.info(info_msg)
            return model

        except Exception as e_1:
            logger.warning(f"Failed to load existing model: {e_1}. Trying to restore weights.")

    try:
        info_msg = "No existing model found. Building a new model from factory."
        logger.info(info_msg)
        factory_model = ModelFactory(
            series_depth=depth,
            config=config,
            logger=logger,
            nb_output_records=max_records
        )
        model = factory_model.build_multi_series_model()

        # The model is now built, so factory_model can be released now.
        del factory_model
        gc.collect()

    except Exception as e_2:
        msg_critical = f"Fatal error. Failed to build new model: {e_2}"

        logger.critical(
            msg_critical,
            extra={"status": "failure", "error": str({e_2})},
            exc_info=True
        )

        raise RuntimeError(msg_critical)

    try:
        if checkpoint_full_path:
            model.load_weights(checkpoint_full_path, skip_mismatch=True)
            logger.info("Weights loaded directly with skip_mismatch=True")

            new_file = checkpoint_dir / "model_restored_fixed.keras"
            model.save(new_file)
            logger.info(
                "Model saved as 'model_restored_fixed.keras'",
                extra={"status": "recovered"}
            )

    except Exception as e_3:
        msg_warning = f"Warning:  Unable to restore the weights. Falling back to factory. {e_3}"
        logger.warning(
            msg_warning,
            extra={"status": "restart", "warning": str(e_3)},
            exc_info=True)

    try:
        # --- Define Losses ---
        # Using MSE for location to penalize large spatial errors more heavily
        losses = {
            "severity_output": rsna_weighted_log_loss,
            "location_output": "mse"
        }

        # --- Define Loss Weights ---
        # We give more weight to location (5.0) because the coordinate regression
        # is currently struggling to converge compared to classification.

        loss_weights_settings = compilation_cfg['loss_weights']

        loss_weights = {
            "severity_output": loss_weights_settings["severity_output"],
            "location_output": loss_weights_settings["location_output"]
        }

        # --- Define Metrics ---
        rsna_score = RSNAKaggleMetric(logger=logger)
        metrics = {
            "severity_output": [rsna_score, "accuracy"],  # Added accuracy for a quick baseline
            "location_output": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "study_id_output": None
        }

        # --- Model Compilation ---
        # Adam optimizer with gradient clipping to prevent exploding gradients
        # during multi-task learning.

        optim_type = optimizer_cfg['type'].lower()
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
                "learning_rate": learning_rate,
                "clip_norm": clip_norm
            }
        )
        return model

    except Exception as e_4:
        critical_msg = f"Fatal error : {e_4}"

        logger.critical(
            critical_msg,
            exc_info=True,
            extra={"status": "failure"}
        )

        raise e_4


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

    os.environ["TF_USE_LEGACY_KERAS"] = "1"
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

    config = config_loader.get()

    with setup_logger(
        process_name="train",
        log_dir=log_dir,
        config=config
    ) as logger:

        # The logger is now set up available globally via get_current_logger().
        # It will automatically close at the end of this block.
        logger.info(f"Configuration loaded successfully. Loaded_values: {config}")

        try:
            if config["series_depth"] is None:
                data_specs_cfg = config["data_specs"]
                percentile_str = data_specs_cfg["series_depth_percentile"]

                if "tfrecord" in paths_cfg and "dicom_studies" in paths_cfg:
                    series_depth = config_loader.calculate_series_depth(
                        tfrecord_dir=paths_cfg["tfrecord"],
                        dicom_studies_dir=paths_cfg["dicom_studies"],
                        percentile=int(percentile_str),
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
            tf.keras.backend.clear_session()

            # Load the compiled model if it already exists. In the other case, generate
            # a new model and compile it.
            model: tf.keras.Model = get_or_build_model(series_depth, config, logger)

            logger.info(
                "Starting training process.",
                extra={"status": "started", "log_dir": log_dir}
            )

            # Extract DICOM images and metadata from source directories
            # and serialize them into dedicated TFRecord files (one per patient)
            tfrecord_files_manager = TFRecordFilesManager(config, logger)
            actual_nb_tfrecord_files = tfrecord_files_manager.generate_tfrecord_files()

            # Store the number of tfrecord_files in the cache file
            tfrecord_dir = Path(paths_cfg['tfrecord'])
            cache_path = tfrecord_dir / "depth_metadata_cache.json"

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

            logger.info("Cache successfully updated in {cache_path}")

            trainer = ModelTrainer(
                config=config,
                logger=logger,
                model=model,
                model_depth=series_depth
            )

            trainer.prepare_training_and_validation_datasets()
            trainer.train_model()

            # Force Keras to close properly before the end of the script
            tf.keras.backend.clear_session()
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
