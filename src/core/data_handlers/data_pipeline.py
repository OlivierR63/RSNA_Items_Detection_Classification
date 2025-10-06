# coding: utf-8

from src.core.data_handlers.dicom_dataset import DicomTFDataset
from src.core.data_handlers.csv_metadata import CSVMetadata
import tensorflow as tf

class DataPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.dicom_dataset = DicomTFDataset(config["dicom_root_dir"])
        self.metadata = CSVMetadata(**config["csv_files"])

    def create_paired_dataset(self) -> tf.data.Dataset:
        """Cree un dataset de paires (image, metadata)."""
        # Charge les métadonnées dans un dictionnaire TensorFlow
        metadata_dict = self.metadata.to_tf_lookup()

        # Fonction pour associer une image à ses métadonnées
        def pair_image_metadata(file_path: str) -> Tuple[tf.Tensor, dict]:
            img = self.dicom_dataset._load_dicom(file_path)
            # Extrait l'ID unique pour la lookup (ex: "study_1_series_1_1.dcm")
            file_name = tf.strings.split(file_path, "/")[-1]  # Ex: "1.dcm"
            instance_number = tf.strings.split(file_name, ".")[0]  # Ex: "1"
            study_series = tf.strings.split(file_path, "/")[-3:-1]  # Ex: ["study_1", "series_1"]
            key = tf.strings.join([study_series[0], study_series[1], instance_number], separator="_")
            metadata = metadata_dict.lookup(key)
            return img, metadata

        # Crée le dataset final
        return self.dicom_dataset.file_paths.map(pair_image_metadata, num_parallel_calls=tf.data.AUTOTUNE)
