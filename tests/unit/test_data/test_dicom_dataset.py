# tests/unit/test_data/test_dicom_dataset.py
import tensorflow as tf
import pytest
from src.core.data.dicom_dataset import DicomTFDataset

def test_dicom_tf_dataset():
    dataset = DicomTFDataset(
        root_dir="tests/fixtures/dicom_samples",
        study_ids=["study_1"],
        series_ids=["series_1"]
    )
    tf_dataset = dataset.create_tf_dataset(batch_size=2)
    for batch in tf_dataset.take(1):
        assert batch.shape == (2, 256, 256, 1)  # Exemple pour des images 256x256
