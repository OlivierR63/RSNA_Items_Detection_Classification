from core.pipeline import BasePipeline
from core.data_handlers import LumbarSpineDicomDataset, CSVMetadata
from core.models import YoloSegmentation2D, CNN3D

class LumbarSpinePipeline(BasePipeline):
    """Pipeline spécifique pour la colonne vertébrale."""

    def _init_dicom_dataset(self):
        return LumbarSpineDicomDataset(self.config['dicom_root_dir'])

    def _init_metadata(self):
        return CSVMetadata(
            self.config['series_desc_path'],
            self.config['label_coords_path'],
            self.config['train_path']
        )

    def _init_2d_model(self):
        return YoloSegmentation2D(
            input_shape=(256, 256, 1),
            num_classes=2  # Exemple : binaire (pathologie/non-pathologie)
        )

    def _init_3d_model(self):
        return CNN3D(
            input_shape=(64, 64, 64, 1),
            num_classes=3  # Normal/Mild, Moderate, Severe
        )
