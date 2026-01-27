# coding: utf-8

import logging
import os

# ---------------- Environmental configuration (before any TF import) -----------------------
# Suppress TensorFlow C++ environmental logs
# '0' = all logs (default), '1' = filter INFO, '2' = filter INFO & WARNING, '3' = filter all including ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable oneDNN custom operations messages
# This prevents warnings regarding slight numerical differences due to floating-point round-off
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ----------------- Standard imports -------------------------------------------------------
import signal
import sys
import logging

# ----------------- TensorFlow and projects imports -----------------------------------------
import tensorflow as tf
# Mute Python-level TensorFlow/Keras warnings
# This handles warnings such as deprecated 'tf.placeholder' calls within internal Keras modules
tf.get_logger().setLevel(logging.ERROR)

from src.core.utils.logger import setup_logger, get_current_logger
from src.core.utils.clean_logs import clean_old_logs
from src.core.models.model_factory import ModelFactory
from src.projects.lumbar_spine.model_trainer import ModelTrainer
from src.projects.lumbar_spine.custom_losses import rsna_weighted_log_loss, zero_loss
from src.config.config_loader import ConfigLoader
from src.projects.lumbar_spine.tfrecord_files_manager import TFRecordFilesManager
from src.core.models.temporal_padding_layer import TemporalPaddingLayer
from typing import Optional


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
    logger: Optional[logging.Logger] = None
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
        logger (Optional[logging.Logger]): Logger instance for status 
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
            "Attribute TFRecordFilesManager.max_series_depth was not properly set."
        )
        raise ValueError(error_msg)

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    last_model_path = os.path.join(checkpoint_dir, "last_model.keras")

    MAX_RECORDS = config.get('max_records', 0)

    if os.path.exists(last_model_path):
        info_msg = f"Existing model found at {last_model_path}. Loading for resume..."
        logger.info(info_msg)

        try:
            model = tf.keras.models.load_model(
                last_model_path, 
                custom_objects={
                    "TemporalPaddingLayer": TemporalPaddingLayer,
                    "rsna_weighted_log_loss": rsna_weighted_log_loss,
                    "zero_loss": zero_loss
                }
            )

            logger.info("Model loaded successfully", extra={"model_architecture": str(model.summary())})
            return model

        except Exception as e:
            warning_msg = f"Failed to load existing model: {e}. Falling back to factory."
            logger.warning(warning_msg)

    try:
        info_msg = "No existing model found. Building a new model from factory."
        logger.info(info_msg)
        factory_model = ModelFactory(series_depth=depth, config=config, logger=logger, nb_output_records=MAX_RECORDS)
        model = factory_model.build_multi_series_model()

        losses = {"study_id_output": zero_loss}
        metrics = {"study_id_output": []}

        for idx in range(MAX_RECORDS):
            losses[f"severity_row_{idx}"] = rsna_weighted_log_loss
            metrics[f"severity_row_{idx}"] = ["accuracy"]
            losses[f"location_row_{idx}"] = "mse"
            metrics[f"location_row_{idx}"] = ["mae"]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
            loss=losses,
            metrics=metrics
        )

        logger.info("New model compiled and ready for training.", extra={"optimizer": "Adam", "learning_rate": 1e-4})
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

    # Setup signal handler for graceful interruption
    signal.signal(signal.SIGINT, handle_interrupt)

    # 1. Load the project configuration
    config_loader = ConfigLoader("src/config/lumbar_spine_config.yaml")
    config: dict = config_loader.get()

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

            # Load the compiled model if it already exists. In the other case, generate
            # a new model and compile it.
            model: tf.keras.Model = load_model(depth, config, logger)

            # Clear any background memory from the model building phase 
            # to free up RAM before the data pipeline starts prefetching.
            tf.keras.backend.clear_session()

            logger.info("Starting training process.",
                    extra={"status": "started", log_dir: "config_dir"})
            
            trainer = ModelTrainer(config=config, logger=logger, model=model, model_depth = depth)
            trainer.train_model()

        except Exception as e:
            logger.error(f"Critical error during training: {str(e)}", exc_info=True,
                         extra={"status": "failed", "error": str(e)})
            raise

        finally:
            logger.info("Training process completed. Log file will be closed automatically.",
                        extra={"status": "completed"})

        # Remove log files older than 30 days
        clean_old_logs(days=30)


if __name__ == "__main__":    
    main()
