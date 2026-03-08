# coding: utf-8

import os

# Create a temp folder on your F: drive if it doesn't exist
os.environ['TMPDIR'] = 'F:\\temp_tf'
os.environ['TEMP'] = 'F:\\temp_tf'
os.environ['TMP'] = 'F:\\temp_tf'

os.environ['TF_NUM_INTEROP_THREADS'] = '7' # Force the number of logical cores.
os.environ['TF_NUM_INTRAOP_THREADS'] = '7'
os.environ["OMP_NUM_THREADS"] = '7'
os.environ["MKL_NUM_THREADS"] = '7'

# ---------------- Environmental configuration (before any TF import) -----------------------
# Suppress TensorFlow C++ environmental logs
# '0' = all logs (default), '1' = filter INFO, '2' = filter INFO & WARNING, '3' = filter all including ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable oneDNN custom operations messages
# This prevents warnings regarding slight numerical differences due to floating-point round-off
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ----------------- Standard imports --------------------------------------------------------
import signal
import sys
import logging
import gc

# ----------------- Project-specific utility imports (No TF here) ---------------------------
from src.config.config_loader import ConfigLoader

# 3. Load the configuration EARLY
# This allows to use config values for OS environment variables if needed
config_loader = ConfigLoader("src/config/lumbar_spine_config.yaml")
config = config_loader.get()

# ----------------- TensorFlow and projects imports -----------------------------------------
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(7)  
tf.config.threading.set_inter_op_parallelism_threads(7)

# Apply global TF settings from the config
training_cfg = config.get('training', None)
if training_cfg is None:
    error_msg = (
        "Fatal error in load_model: "
        "the setting variable 'training' is required "
        "but was not found. Please check your YAML file structure."
    )
    raise ValueError(error_msg)

compilation_cfg = training_cfg.get('compilation', None)
if compilation_cfg is None:
    error_msg = (
        "Fatal error in load_model: "
        "the setting variable 'training -> compilation' is required "
        "but was not found. Please check your YAML file structure."
    )
    raise ValueError(error_msg)

hyperparam_cfg = training_cfg.get('hyperparameters', None)
if hyperparam_cfg is None:
    error_msg = (
        "Fatal error in load_model: "
        "the setting variable 'training -> hyperparameters' is required "
        "but was not found. Please check your YAML file structure."
    )
    raise ValueError(error_msg)

run_eagerly_val = compilation_cfg.get('run_eagerly', False)

if run_eagerly_val not in [True, False]:
    error_msg = (
        "Fatal error: the parameter 'training -> compilation -> run_eagerly' "
        "is  not a boolean. Please check your YAML file structure."
    )
    raise ValueError(error_msg)

tf.config.run_functions_eagerly(run_eagerly_val)
tf.data.experimental.enable_debug_mode() # Optionnel mais aide pour tf.data

# Mute Python-level TensorFlow/Keras warnings
# This handles warnings such as deprecated 'tf.placeholder' calls within internal Keras modules
tf.get_logger().setLevel(logging.ERROR)

from src.core.utils.logger import setup_logger, get_current_logger
from src.core.utils.system_stream_tee import SystemStreamTee
from src.core.utils.clean_logs import clean_old_logs
from src.core.models.model_factory import ModelFactory
from src.projects.lumbar_spine.model_trainer import ModelTrainer
from src.projects.lumbar_spine.RSNA_lumbar_losses_and_metric import rsna_weighted_log_loss, RSNAKaggleMetric
from src.projects.lumbar_spine.tfrecord_files_manager import TFRecordFilesManager
from pathlib import Path


def handle_interrupt(signum, frame):
    """Handles interrupt signals (Ctrl+C) to ensure proper log closure."""
    try:
        logger = get_current_logger()
        logger.info("\nInterruption detected (Ctrl+C). Exiting gracefully...")

    except RuntimeError:
        print("\nInterruption detected (Ctrl+C). Exiting gracefully...")

    sys.exit(0)


def load_model(
    depth: int,
    config: dict,
    logger: logging.Logger
) -> tf.keras.Model:

    """
    Retrieves a pre-existing model from a checkpoint or initializes a new one, 
    with an automated weight restoration fallback.

    This function implements a robust "Resume Training" logic designed to handle 
    structural breaking changes (e.g., Lambda layer serialization issues). 
    It operates in three priority stages:
    1. Full Restore: Attempts to load the complete Keras model (architecture + weights + optimizer state).
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

    if depth is None or depth <= 0 :
        error_msg = (
            "Function load_model failed. Null depth parameter"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Extract configs locally within the function to ensure independence
    training_cfg = config.get('training', {})
    hyperparam_cfg = training_cfg.get('hyperparameters', {})
    compilation_cfg = training_cfg.get('compilation', {})

    try:
        max_records = int(config['data_specs'].get('max_records_per_frame', -1))
        if max_records < 0:
            error_msg = (
                "Fatal error: the setting variable 'data_specs -> max_records_per_frame' "
                "is requird but was not found. Please check your YAML file structure."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        learning_rate = float(hyperparam_cfg.get('learning_rate', -1))
        clip_norm = float(hyperparam_cfg.get('clipnorm', -1))

    except (ValueError, TypeError) as e:
        raise ValueError(f"Configuration type error: {e}")

    # Explicit check to prevent logic errors later in the pipeline
    if max_records < 0 or learning_rate < 0 or clip_norm < 0:
        error_msg = (
            "Fatal error: the setting variables 'learning_rate' and/or 'clipnorm' "
            "were not properly defined. Please check your configuration file."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Select the checkpoint loading policy: 'best' for the lowest validation loss 
    # or 'last' to continue from the most recent epoch.
    checkpoint_dir = Path(config["paths"].get("checkpoint", "checkpoints")).resolve()
    resume_mode = hyperparam_cfg.get("resume_mode", "last")

    if resume_mode not in ["best", "last"]:
        error_msg = (
            "Fatal error: the parameter 'training -> hyperparameters -> resume_mode' "
            "was not properly set. Only two values are permitted: 'best' or 'last'. "
            "Please check your YAML file structure."
        )
        logger.critical(error_msg)
        raise ValueError(error_msg)
    
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
            return model

        except Exception as e_1:
            logger.warning(f"Failed to load checkpoint: {e_1}. Trying to restore weights.")

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

    except Exception as e_2:
        msg_error = f"Fatal error. Failed to build new model: {e_2}"
        logger.error(msg_error, extra ={"status": "failed", "error": str({e_2})}, exc_info=True)
        raise RuntimeError(msg_error)

    try:
        if checkpoint_full_path:
            model.load_weights(checkpoint_full_path, skip_mismatch=True)
            logger.info("Weights loaded directly with skip_mismatch=True")

            new_file = checkpoint_dir / "model_restored_fixed.keras"
            model.save(new_file)
            logger.info("Model saved as 'model_restored_fixed.keras'", extra={"status": "recovered"})

    except Exception as e_3:
        msg_warning = f"Warning:  Unable to restore the weights. Falling back to factory. {e_3}"
        logger.warning(
            msg_warning,
            extra={"status": "restart", "warning": str({e_3})},
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

        loss_weights_settings = compilation_cfg.get('loss_weights', None)

        if loss_weights_settings is None:
            error_msg = (
                "Fatal error in load_model: "
                "the setting variable 'training -> compilation -> loss_weights' is required "
                "but was not found. Please check your YAML file structure."
            )

            raise ValueError(error_msg)

        loss_weights = {
            "severity_output": loss_weights_settings["severity_output"],
            "location_output": loss_weights_settings["location_output"] 
        }

        # --- Define Metrics ---
        rsna_score = RSNAKaggleMetric()
        metrics = {
            "severity_output": [rsna_score, "accuracy"],  # Added accuracy for a quick baseline
            "location_output": [tf.keras.metrics.MeanAbsoluteError(name="mae")]
        }

        # --- Model Compilation ---
        # Adam optimizer with gradient clipping to prevent exploding gradients 
        # during multi-task learning.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipnorm=clip_norm
            ),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            run_eagerly=run_eagerly_val
        )

        model.summary()
        logger.info(
            "New model compiled with weighted losses.",
            extra={
                "optimizer": "Adam",
                "learning_rate": learning_rate,
                "clip_norm":clip_norm
            }
        )
        return model

    except Exception as e_4:
        critical_msg = f"Fatal error : {e_4}"
        logger.critical(critical_msg)
        raise e_4


def main():
    """
        Main function to load configuration, set up the TensorFlow dataset pipeline,
        load and compile the 3D model, and start the training process.
    """
    # Define the policy: float16 for computations, float32 for critical variables.

    # Setup signal handler for graceful interruption
    signal.signal(signal.SIGINT, handle_interrupt)

    # 1. Load the project configuration
    config_loader = ConfigLoader("src/config/lumbar_spine_config.yaml")
    config: dict = config_loader.get()

    # 2. Allow to parallelize operations on the requested number of cores
    #nb_cores = config['systems'].get('nb_cores', 0)
    #tf.config.threading.set_intra_op_parallelism_threads(nb_cores)  
    #tf.config.threading.set_inter_op_parallelism_threads(nb_cores)

    # Mirror all the console and error output toward a dedicated file
    system_stream_tee_path = config['paths'].get('log_mirror', None)
    if system_stream_tee_path is None:
        error_msg = (
            "Fatal error: the parameter 'paths -> log_mirror' is required but was not found. "
            "Please check your YAML file structure."
        )
        raise ValueError(error_msg)
    
    mirror = SystemStreamTee(system_stream_tee_path)

    sys.stdout = mirror
    sys.stderr = mirror

    # 2. Initialize logger with process-specific context
    log_dir = config["paths"].get("output", "logs")  # use "logs" as default if not in config.
    log_dir += "/logs"

    with setup_logger("train", log_dir=log_dir, use_json=True) as logger:
        # The logger is now set up available globally via get_current_logger().
        # It will automatically close at the end of this block.
        logger.info(f"Configuration loaded successfully. Loaded_values: {config}")
        
        try:
            # Generate TFRecord files if they don't exist
            file_manager = TFRecordFilesManager(config=config, logger=logger)
            file_manager.generate_tfrecord_files()
            depth = file_manager.get_max_series_depth()

            logger.info(f"Max TFRecord files depth (number of DICOM files per series): {depth}")

            # At this step, file_manager is no more usefull and must be removed.
            del file_manager
            gc.collect()

            # Clear any background memory from the model building phase 
            # to free up RAM before the data pipeline starts prefetching.
            tf.keras.backend.clear_session()

            # Load the compiled model if it already exists. In the other case, generate
            # a new model and compile it.
            model: tf.keras.Model = load_model(depth, config, logger)

            logger.info("Starting training process.",
                    extra={"status": "started", "log_dir": log_dir})
            
            trainer = ModelTrainer(config=config, logger=logger, model=model, model_depth = depth)
            trainer.prepare_training_and_validation_datasets()
            trainer.train_model()

        except Exception as e:
            logger.error(f"Critical error during training: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            raise

        finally:
            logger.info("Training process completed. Log file will be closed automatically.",
                        extra={"status": "completed"})

        # Remove log files older than 30 days
        log_retention_days = config['system'].get('log_retention_days', None)

        if log_retention_days is None:
            error_msg = (
                "Fatal error: the setting variable 'system -> log_retention_days' "
                "is raquired but was not found. Please check your YAML file structure."
            )
            raise ValueError(error_msg)

        clean_old_logs(days=int(log_retention_days))
        mirror.close()


if __name__ == "__main__":    
    main()
