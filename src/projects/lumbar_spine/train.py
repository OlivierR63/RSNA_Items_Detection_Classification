# coding: utf-8

from typing import Optional, Dict, Any
import logging
import tensorflow as tf
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
from src.core.models.model_factory import ModelFactory
from src.core.utils.logger import get_current_logger, log_method
from pathlib import Path


@log_method()
def train_model(*, config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    """Training function with automatic logger injection and structured logging."""
    if logger is None:
        logger = get_current_logger()

    logger.info("Loading 3D model...",
                    extra={"action": "load_model", "model_type": config['model_3d']['type']})

    try:
        # Create dataset with logger
        dataset = create_tf_dataset(config, logger=logger)

        # Load and compile model
        model = ModelFactory.create_model(config["model_3d"])
        logger.info("Model loaded.", extra={"model_architecture": str(model.summary())})

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        logger.info("Model compiled.", extra={"optimizer": "Adam", "learning_rate": 1e-4})

        # Train the model
        _ = train_with_callbacks(model, dataset, config, logger=logger)

        # Save the model
        model.save(str(Path(config["output_dir"]) / "model"))
        logger.info(f"Model saved to {config['output_dir']}/model",
                        extra={"status": "success",
                                "model_path": str(Path(config["output_dir"]) / "model")})

    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}", exc_info=True,
                     extra={"status": "failed", "error": str(e)})
        raise


@log_method()
def create_tf_dataset(config: Dict[str, Any], *, logger: Optional[logging.Logger] = None):
    """Creates TensorFlow dataset with logging."""
    if logger is None:
        logger = get_current_logger()

    logger.info("Setting up TensorFlow dataset pipeline...",
                    extra={"action": "create_dataset", "batch_size": config["batch_size"]})

    try:
        dataset = LumbarDicomTFRecordDataset(config, logger).create_tf_dataset(
            batch_size=config["batch_size"]
        )
        logger.info("Dataset created successfully.",
                        extra={"status": "success", "batch_size": config["batch_size"]})
        return dataset
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}", exc_info=True,
                     extra={"status": "failed", "error": str(e)})
        raise


@log_method()
def train_with_callbacks(model, dataset, config: Dict[str, Any], *,
                                logger: Optional[logging.Logger] = None) -> None:
    """Trains model with callbacks and logging."""
    if logger is None:
        logger = get_current_logger()

    logger.info("Setting up training callbacks...", extra={"action": "setup_callbacks"})

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(config["output_dir"]) / "model_checkpoint"),
            save_best_only=True,
            monitor="val_loss"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(Path(config["output_dir"]) / "tensorboard_logs"),
            histogram_freq=1
        )
    ]
    logger.info("Training callbacks configured.",
                    extra={"callbacks": [c.__class__.__name__ for c in callbacks]})

    logger.info("Starting model training...",
                    extra={"epochs": config["epochs"], "steps_per_epoch": 1000})

    try:
        history = model.fit(
            dataset,
            epochs=config["epochs"],
            steps_per_epoch=1000,
            validation_data=None,
            callbacks=callbacks
        )
        logger.info("Model training completed successfully.",
                       extra={
                               "final_loss": history.history['loss'][-1],
                               "final_accuracy": history.history['accuracy'][-1]
                           })
        return history
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True,
                     extra={"status": "failed", "error": str(e)})
        raise
