# coding: utf-8

import os

# Create a temp folder on your F: drive if it doesn't exist
os.environ['TMPDIR'] = 'F:\\temp_tf'
os.environ['TEMP'] = 'F:\\temp_tf'
os.environ['TMP'] = 'F:\\temp_tf'

os.environ['TF_NUM_INTEROP_THREADS'] = '4' # Force the number of logical cores.
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'

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

# ----------------- TensorFlow and projects imports -----------------------------------------
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(0)  
tf.config.threading.set_inter_op_parallelism_threads(0)

tf.config.run_functions_eagerly(True)
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
from src.config.config_loader import ConfigLoader
from src.projects.lumbar_spine.tfrecord_files_manager import TFRecordFilesManager
from src.core.models.temporal_padding_layer import TemporalPaddingLayer
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
    Retrieves a pre-existing model from a checkpoint or initializes a new one.

    This function implements a "Resume Training" logic. It checks for a saved 
    Keras model at the specified checkpoint path. If found, it restores the 
    model using custom objects (layers and losses). If no checkpoint is found 
    or if loading fails, it utilizes the ModelFactory to build and compile 
    a fresh architecture from scratch.

    Args:
        depth (int): The maximum number of slices per imaging series, 
            used for 3D convolution input shapes.
        config (dict): Global configuration dictionary containing paths, 
            hyperparameters, and model settings.
        logger (logging.logger): Logger instance for status 
            tracking and error reporting.

    Returns:
        tf.keras.Model: A compiled TensorFlow Keras model ready for training.

    Raises:
        ValueError: If the series depth is invalid (None or <= 0).
        Exception: If a fatal error occurs during model initialization.
    """

    if depth is None or depth <= 0 :
        error_msg = (
            "Function load_model failed. Null depth parameter"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    checkpoint_path = config["paths"].get("checkpoint", None)

    if checkpoint_path is None:
        error_msg = (
            "Fatal error: the parameter 'paths -> checkpoint' is required but was not found. "
            "Please check your YAML file structure."
        )
        logger.critical(error_msg)
        raise ValueError(error_msg)
    
    checkpoint_full_path = Path(checkpoint_path).resolve()

    training_settings = config.get('training', None)

    if training_settings is None:
        error_msg = (
            "Fatal error in load_model: "
            "the setting variable 'training' is required "
            "but was not found. Please check your YAML file structure."
        )
        raise ValueError(error_msg)

    hyperparameters = training_settings.get('hyperparameters', None)
    if hyperparameters is None:
        error_msg = (
            "Fatal error in load_model: "
            "the setting variable 'training -> hyperparameters' is required "
            "but was not found. Please check your YAML file structure."
        )
        raise ValueError(error_msg)

    # Ensure types
    try:
        max_records = int(config['data_specs'].get('max_records_per_frame', -1))
        if max_records < 0:
            error_msg = (
                "Fatal error: the setting variable 'data_specs -> max_records_per_frame' "
                "is requird but was not found. Please check your YAML file structure."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        learning_rate = float(hyperparameters.get('learning_rate', -1))
        clip_norm = float(hyperparameters.get('clipnorm', -1))

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

    if checkpoint_full_path.is_file():
        info_msg = f"Existing model found at {checkpoint_full_path}. Loading for resume..."
        logger.info(info_msg)

        try:
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(
                checkpoint_full_path, 
                custom_objects={
                    "TemporalPaddingLayer": TemporalPaddingLayer,
                    "rsna_weighted_log_loss": rsna_weighted_log_loss,
                    "RSNAKaggleMetric": RSNAKaggleMetric
                },
                safe_mode=False,
            )

            logger.info("Model loaded successfully", extra={"model_architecture": str(model.summary())})
            return model

        except Exception as e:
            warning_msg = f"Failed to load existing model: {e}. Falling back to factory."
            logger.warning(warning_msg)

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

        # --- Define Losses ---
        # Using MSE for location to penalize large spatial errors more heavily
        losses = {
            "severity_output": rsna_weighted_log_loss,
            "location_output": "mse"
        }

        # --- Define Loss Weights ---
        # We give more weight to location (5.0) because the coordinate regression 
        # is currently struggling to converge compared to classification.
        
        compilation_settings = training_settings.get('compilation', None)

        if compilation_settings is None:
            error_msg = (
                "Fatal error in load_model: "
                "the setting variable 'training -> compilation' is required "
                "but was not found. Please check your YAML file structure."
            )

            raise ValueError(error_msg)

        run_eagerly_settings = compilation_settings.get('run_eagerly', False)

        if run_eagerly_settings not in (True, False):
            error_msg = (
                "Fatal error in load_model: "
                "the setting variable 'training -> compilation -> run_eagerly' is required "
                "but is corrupted or was not found. Please check your YAML file structure."
            )

            raise ValueError(error_msg)

        loss_weights_settings = compilation_settings.get('loss_weights', None)

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
            run_eagerly=run_eagerly_settings
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

    except Exception as e:
        critical_msg = f"Fatal error : {e}"
        logger.critical(critical_msg)
        raise e


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
