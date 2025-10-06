#coding: utf-8

from ast import Dict
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

def create_string_to_int_mapper(strings: list) -> callable:
    """Crée une fonction de mapping entre des chaînes et des entiers.

    Args:
        strings: Liste des chaînes à mapper (ex: ["Normal", "Stenosis"]).

    Returns:
        Une fonction qui mappe une chaîne vers un entier.
    """
    # Crée un dictionnaire {chaîne: entier}
    mapping = {s: i for i, s in enumerate(strings)}
    reverse_mapping = {i: s for i, s in enumerate(strings)}  # Optionnel: pour le mapping inverse

    def mapper(s: str) -> int:
        """Mappe une chaîne vers un entier."""
        return mapping.get(s, -1)  # Retourne -1 si la chaîne est inconnue

    # Ajoute des attributs pour accéder aux mappings (optionnel)
    mapper.mapping = mapping
    mapper.reverse_mapping = reverse_mapping

    return mapper


# Fonction générique pour mapper une valeur
def map_value(value: str, mapping: dict) -> int:
    return mapping.get(value, -1)  # Retourne -1 si la valeur est inconnue


def convert_metadata_to_digit(metadata: dict) -> dict:
    """Convertit les métadonnées textuelles en valeurs numériques.

    Args:
        metadata: Dictionnaire contenant les métadonnées à convertir.
                  Ex: {"condition": "Spinal Canal Stenosis", "level": "L1-L2", ...}

    Returns:
        dict: Dictionnaire avec les valeurs numériques.
              Ex: {"condition": 0, "level": 0, "description": -1, "severity": 1}
    """
    # Dictionnaires de mapping pour chaque champ
    condition_values = metadata["condition"].unique().tolist()
    CONDITION_MAP = create_string_to_int_mapper(condition_values).mapping

    level_values = metadata["level"].unique().tolist()
    LEVEL_MAP = create_string_to_int_mapper(level_values).mapping

    description_values = metadata["description"].unique().tolist()
    DESCRIPTION_MAP = create_string_to_int_mapper(description_values).mapping

    severity_values = metadata["severity"].unique().tolist()
    SEVERITY_MAP = create_string_to_int_mapper(severity_values).mapping

    # Crée le dictionnaire de sortie
    output_data = {
        "condition": map_value(metadata.get("condition", ""), CONDITION_MAP),
        "level": map_value(metadata.get("level", ""), LEVEL_MAP),
        "description": map_value(metadata.get("description", ""), DESCRIPTION_MAP),
        "severity": map_value(metadata.get("severity", ""), SEVERITY_MAP),
    }

    return output_data


def get_metadata_for_file(file_path: str, metadata_df: pd.DataFrame) -> dict:
    """Retourne les métadonnées pour un fichier DICOM donné."""
    if metadata_df is None:
        return {}

    parts = Path(file_path).parts
    study_id = parts[-3]
    series_id = parts[-2]
    instance_number = Path(file_path).stem

    mask = (
        (metadata_df["series_id"] == int(series_id)) &
        (metadata_df["instance_number"] == int(instance_number))
    )
    records = metadata_df[mask]
    return records.iloc[0].to_dict() if not records.empty else {}


def convert_dicom_to_tfrecords(root_dir: str, metadata_df: pd.DataFrame, output_dir: str):
    """Convertit les fichiers DICOM en TFRecords par étude."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for study_path in tqdm(list(Path(root_dir).iterdir())):
        if not study_path.is_dir():
            continue

        study_id = study_path.name
        output_path = output_dir / f"{study_id}.tfrecord"

        with tf.io.TFRecordWriter(str(output_path)) as writer:
            for series_path in study_path.iterdir():
                if not series_path.is_dir():
                    continue

                series_id = int(series_path.name)
                for dicom_path in series_path.glob("*.dcm"):
                    # Charge l'image DICOM en conservant le type d'origine
                    img = sitk.ReadImage(str(dicom_path))
                    img_array = sitk.GetArrayFromImage(img)

                    # Convertit en tenseur avec le type optimal (int16 ou uint16)
                    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint16)
                    img_bytes = tf.io.serialize_tensor(img_tensor).numpy()

                    # Récupère les métadonnées (sans chemin complet)
                    metadata = get_metadata_for_file(str(dicom_path), metadata_df)

                    # Extrait instance_number et le stocke sur 1 ou 2 octets selon les cas,
                    # en partant du principe que le numéro d'instance comporte un maximum de 4 chiffres 
                    instance_number = int(dicom_path.stem)
                    nb_bytes = 1 if instance_number <= 255 else 2
                    instance_bytes = instance_number.to_bytes(nb_bytes, byteorder='big')

                    # Crée la feature
                    feature = {
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                        # Inutile de stocker le study_id puisque c'est le nom du fichier.
                        "series_id": tf.train.Feature(bytes_list=tf.train.Int64List(value=[series_id])),
                        "instance_number": tf.train.Feature(bytes_list=tf.train.BytesList(value=[instance_bytes])),
                        # condition =  {0:"Spinal Canal Stenosis", 1:"Right Neural Foraminal Narrowing", 2:"Left Neural Foraminal Narrowing", 3:"Right Subarticular Stenosis", 4:"Left Subarticular Stenosis"}
                        "condition": tf.train.Feature(bytes_list=tf.train.BytesList(value=[instance_number.to_bytes(1, byteorder='big')])),
                        # level = {0: "L1-L2", 1: "L2-L3", 2: "L3-L4", 3: "L4-L5", 4: "L5-S1"}
                        "level": tf.train.Feature(bytes_list=tf.train.BytesList(value=[instance_number.to_bytes(1, byteorder='big')])),
                        # description = {0: "Sagittal T1", 1:"Sagittal T2/STIR", 2:"Axial T2"} 
                        "description": tf.train.Feature(bytes_list=tf.train.BytesList(value=[instance_number.to_bytes(1, byteorder='big')])),
                        "x": tf.train.Feature(bytes_list=tf.train.float64List(value=[metadata["x"]])),
                        "y": tf.train.Feature(bytes_list=tf.train.float64List(value=[metadata["y"]])),
                        # severity = {0: "Mild/Normal", 1: "Moderate", 2: "Severe"}
                        "Severity": tf.train.Feature(bytes_list=tf.train.BytesList(value=[instance_number.to_bytes(1, byteorder='big')])),
                    }

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
