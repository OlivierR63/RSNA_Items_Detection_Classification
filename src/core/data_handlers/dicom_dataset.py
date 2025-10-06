# coding: utf-8

import tensorflow as tf
import SimpleITK as sitk
from pathlib import Path
from typing import List, Tuple

class DicomTFDataset:
    """Dataset TensorFlow pour charger des fichiers DICOM sans boucles Python."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

        # Crée une liste de tous les chemins de fichiers DICOM (sans boucles explicites)
        self.file_paths = self._generate_file_paths()

    def _generate_file_paths(self) -> tf.data.Dataset:
        """Genere dynamiquement les chemins des fichiers DICOM avec tf.data."""
        # Crée un dataset de paires (study_id, series_id)
        study_series_pairs = tf.data.Dataset.from_tensor_slices(
            [(study_id, series_id) for study_id in self.study_ids for series_id in self.series_ids]
        )

        # Applique une fonction pour générer les chemins complets
        def get_dicom_paths(study_id: str, series_id: str) -> List[str]:
            series_path = str(Path(self.root_dir) / study_id / series_id)
            # Utilise tf.io.gfile pour lister les fichiers (compatible avec les liens symboliques)
            file_paths = tf.io.gfile.glob(f"{series_path}/*.dcm")
            return file_paths

        # Utilise tf.py_function pour appliquer la fonction Python (nécessaire pour gfile)
        file_paths_dataset = study_series_pairs.map(
            lambda study_id, series_id: tf.py_function(
                func=get_dicom_paths,
                inp=[study_id, series_id],
                Tout=tf.string
            )
        )

        # Aplatit la liste des chemins (chaque élément est une liste de chemins)
        return file_paths_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

    def _load_dicom(self, file_path: str) -> tf.Tensor:
        """Charge un fichier DICOM en tensor (utilise pydicom via tf.py_function)."""
        def _py_load_dicom_tf(path: str) -> tf.Tensor:
            img = sitk.ReadImage(path.decode('utf-8'))
            img_array = sitk.GetArrayFromImage(img)
            return tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Utilise tf.py_function pour encapsuler pydicom (non-TensorFlow)
        return tf.py_function(_py_load_dicom_tf, [file_path], tf.float32)

    def create_tf_dataset(self, batch_size: int = 8) -> tf.data.Dataset:
        """Cree un dataset TensorFlow optimise pour les fichiers DICOM."""
        dataset = self.file_paths
        dataset = dataset.map(self._load_dicom, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)