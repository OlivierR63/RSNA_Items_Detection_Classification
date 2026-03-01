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
            "Function TFRecordFilesManager.get_max_series_depth() failed. "
            "Attribute TFRecordFilesManager._max_series_depth was not properly set."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    checkpoint_path = config.get("checkpoint_path", None)

    if checkpoint_path is None:
        error_msg = (
            "Fatal error: the setting variable 'checkpoint_path' is missing."
            "Please check your configuration file."
        )
        logger.critical(error_msg)
        raise ValueError(error_msg)
    
    checkpoint_full_path = Path(checkpoint_path).resolve()

    # Ensure types
    try:
        max_records = int(config.get('max_records', -1))
        learning_rate = float(config.get('learning_rate', -1))
        clip_norm = float(config.get('clipnorm', -1))

    except (ValueError, TypeError) as e:
        raise ValueError(f"Configuration type error: {e}")

    # Explicit check to prevent logic errors later in the pipeline
    if max_records < 0 or learning_rate < 0 or clip_norm < 0:
        error_msg = (
            "Fatal error: the setting variables 'max_records', 'learning_rate' and/or 'clipnorm' "
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

        losses = {
            "severity_output": rsna_weighted_log_loss,
            "location_output": "mse"
        }

        rsna_score = RSNAKaggleMetric()
        metrics = {
            "severity_output": [rsna_score, "accuracy"],  # Added accuracy for a quick baseline
            "location_output": [tf.keras.metrics.MeanAbsoluteError(name="mae")]
        }

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipnorm=clip_norm
            ),
            loss=losses,
            metrics=metrics,
            run_eagerly=True
        )

        model.summary()
        logger.info("New model compiled and ready for training.", extra={"optimizer": "Adam", "learning_rate": learning_rate, "clip_norm":clip_norm})
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
    #nb_cores = config.get('nb_cores', 4)
    #tf.config.threading.set_intra_op_parallelism_threads(nb_cores)  
    #tf.config.threading.set_inter_op_parallelism_threads(nb_cores)

    # Mirror all the console and error output toward a dedicated file
    system_stream_tee_path = config.get('system_stream_mirror_path', None)
    if system_stream_tee_path is None:
        error_msg = (
            "Fatal error: the setting variable 'system_stream_mirror_path' is missing."
            "Please check your configuration file."
        )
        raise ValueError(error_msg)
    
    mirror = SystemStreamTee(system_stream_tee_path)

    sys.stdout = mirror
    sys.stderr = mirror

    # 2. Initialize logger with process-specific context
    log_dir = config.get("output_dir", "logs")  # use "logs" as default if not in config.
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
        log_retention_days = config.get('log_retention_days', None)

        if log_retention_days is None:
            error_msg = (
                "Fatal error: the setting variable 'log_retention_days' is missing."
                "Please check your configuration file."
            )
            raise ValueError(error_msg)

        clean_old_logs(days=int(log_retention_days))
        mirror.close()


if __name__ == "__main__":    
    main()
