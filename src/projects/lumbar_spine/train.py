# coding: utf-8

import tensorflow as tf
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
from src.projects.lumbar_spine.csv_metadata import CSVMetadata
from src.core.models.model_factory import ModelFactory
from src.config.config_loader import ConfigLoader
import os
from pathlib import Path # Required for path manipulation in callbacks and saving

def main():
    """
    Main function to load configuration, set up the TensorFlow dataset pipeline, 
    load and compile the 3D model, and start the training process.
    """
    # 1. Load the Configuration
    # Reads settings (like paths, model parameters, batch size) from the YAML file.
    config = ConfigLoader("src/config/lumbar_spine_config.yaml").get()

    # 5. Build TFRecord Files (if needed) and Create the TensorFlow Dataset
    # This calls the create_tf_dataset function, which handles reading and optimizing 
    # the data loading pipeline from the generated TFRecords.
    dataset = LumbarDicomTFRecordDataset(config).create_tf_dataset(
        batch_size=config["batch_size"]
    )

    # 6. Add Data Augmentation (Optional)
    # Define a function to apply common augmentation techniques (applied on the fly).
    def augment_image(image, metadata):
        # Apply random horizontal flip.
        image = tf.image.random_flip_left_right(image)
        # Apply random brightness changes (max 10% delta).
        image = tf.image.random_brightness(image, max_delta=0.1)
        # Metadata must be returned alongside the augmented image.
        return image, metadata

    # Apply the augmentation map function to the dataset using parallel processing.
    dataset = dataset.map(
        augment_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 7. Load the 3D Model
    # Uses a factory pattern to instantiate the specific 3D model defined in the configuration.
    model = ModelFactory.create_model(config["model_3d"])

    # 8. Compile the Model
    # Configure the learning process with an optimizer, loss function, and metrics.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        # Assuming this is a multi-class classification problem where targets are integers.
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 9. Callbacks for Training
    # Define actions to be taken during training (e.g., saving models, stopping early).
    callbacks = [
        # ModelCheckpoint: Saves the model based on the monitored metric.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(config["output_dir"]) / "model_checkpoint"),
            save_best_only=True,
            monitor="val_loss" # Tracks validation loss to save the best model weights.
        ),
        # EarlyStopping: Stops training if the monitored metric doesn't improve.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5, # Number of epochs with no improvement after which training will be stopped.
            restore_best_weights=True # Loads the weights from the epoch with the best monitored value.
        )
    ]

    # 10. Train the Model
    # Fit the model using the optimized TensorFlow Dataset pipeline.
    history = model.fit(
        dataset,
        epochs=config["epochs"],
        steps_per_epoch=1000,  # A fixed number of steps per epoch (should be adjusted based on dataset size).
        validation_data=None,  # Placeholder: Should be replaced with a validation dataset.
        callbacks=callbacks
    )

    # 11. Save the Final Model
    model.save(str(Path(config["output_dir"]) / "model"))
    print("Model saved successfully.")

if __name__ == "__main__":
    main()