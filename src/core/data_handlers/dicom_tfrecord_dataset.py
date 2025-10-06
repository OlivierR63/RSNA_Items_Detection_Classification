# coding: utf-8

# src/core/data_handlers/dicom_tfrecord_dataset.py
import tensorflow as tf
from typing import Dict, Tuple
from abc import abstractmethod
from pathlib import Path

class DicomTFRecordDataset:
    """Dataset TensorFlow pour charger les TFRecords de DICOM."""

    def __init__(self, config: dict)-> None:
        self._config = config
        self._tfrecord_dir = Path(config["output_dir"]) / "tfrecords"
        self._tfrecord_pattern = str(Path(config["output_dir"]) / "tfrecords" / "*.tfrecord")
        self.generate_tfrecord_files()  # Génère les TFRecords si nécessaire


    @abstractmethod
    def generate_tfrecord_files(self) -> None:
        pass


    @abstractmethod
    def _parse_tfrecord(self, example_proto: tf.Tensor) -> Tuple[tf.Tensor, Dict]:
        pass


    def create_tf_dataset(self, batch_size: int = 8) -> tf.data.Dataset:
        """Crée un dataset TensorFlow optimisé à partir des TFRecords."""
        # 1. Liste tous les fichiers TFRecord
        tfrecord_files = tf.data.Dataset.list_files(self._tfrecord_pattern, shuffle=True)

        # 2. Utilise interleave pour lire plusieurs fichiers en parallèle
        dataset = tfrecord_files.interleave(
            lambda x: tf.data.TFRecordDataset(x).map(
                self._parse_tfrecord,
                num_parallel_calls=tf.data.AUTOTUNE
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # 3. Mélange et optimise le dataset
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset