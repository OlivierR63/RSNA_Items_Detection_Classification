# coding: utf-8

import tensorflow as tf
from src.projects.lumbar_spine.lumbar_dicom_tfrecord_dataset import LumbarDicomTFRecordDataset
from src.core.data_handlers.csv_metadata import CSVMetadata
from src.core.models.model_factory import ModelFactory
from src.config.config_loader import ConfigLoader
import os

def main():
    # 1. Charge la configuration
    config = ConfigLoader("src/config/lumbar_spine_config.yaml").get()

    # 5. Construit les fichiers TFRecord s'ils n'existent pas déjà 
    #    et crée le dataset TensorFlow à partir des TFRecords
    dataset = LumbarDicomTFRecordDataset(config).create_tf_dataset(
        batch_size=config["batch_size"]
    )

    # 6. Ajoute une augmentation de données (optionnel)
    def augment_image(image, metadata):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        return image, metadata

    dataset = dataset.map(
        augment_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 7. Charge le modèle 3D
    model = ModelFactory.create_model(config["model_3d"])

    # 8. Compile le modèle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 9. Callbacks pour l'entraînement
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
        )
    ]

    # 10. Entraîne le modèle
    history = model.fit(
        dataset,
        epochs=config["epochs"],
        steps_per_epoch=1000,  # À ajuster selon la taille de ton dataset
        validation_data=None,  # À remplacer par ton jeu de validation
        callbacks=callbacks
    )

    # 11. Sauvegarde le modèle final
    model.save(str(Path(config["output_dir"]) / "model"))
    print("Modele sauvegarde avec succes.")

if __name__ == "__main__":
    main()