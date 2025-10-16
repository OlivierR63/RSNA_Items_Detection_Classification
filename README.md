 ## **Project Structure**

 The project is organized into several key directories and files to facilitate development, testing, and deployment. Below is an overview of the main components:

 ```
RSNA_Items_Detection_Classification/
│
├── src/
│   │
│   ├── config/                    # Configurations globales et par défaut
│   │   ├── __init__.py
│   │   ├── config_loader_.py
│   │   ├── brain_aneuvrysm_config.yaml       # Configuration par défaut
│   │   └── lumbar_spine_config_.yaml         # Chargeur de configuration
│   │
│   ├── core/                     # Classes de base génériques (abstraites)
│   │   │
│   │   ├── data_handlers_/                 # Gestion des données (DICOM, CSV, métadonnées)
│   │   │   ├── __init__.py
│   │   │   ├── data_pipeline.py
│   │   │   ├── dicom_dataset_.py			# Classe `DicomDataset` (abstraite + implémentations)
│   │   │   └── dicom_tfrecord_dataset.py	# Classe `DicomTFRecordDataset` (abstraite + implémentations)
│   │   │
│   │   ├── models/               # Modèles de deep learning (2D/3D)
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py     # Classe `BaseSegmentationModel` (abstraite)
│   │   │   ├── model_2d.py       # Modèles 2D (YOLO, ResNet)
│   │   │   ├── model_3d.py       # Modèles 3D (CNN 3D)
│   │   │   └── model_factory.py  # Fabrique pour instancier les modèles
│   │   │
│   │   ├── pipeline/              # Pipeline de traitement (TFX, orchestration)
│   │   │   ├── __init__.py
│   │   │   ├── base_pipeline.py  # Classe `BasePipeline` (abstraite)
│   │   │   ├── tfx_pipeline.py   # Intégration avec TFX
│   │   │   └── airflow_pipeline.py # Intégration avec Airflow (optionnel)
│   │   │
│   │   └── utils/                 # Utilitaires (logs, helpers)
│   │       ├── __init__.py
│   │       ├── clean_logs.py      
│   │       ├── logger.py          # Gestion des logs
│   │       ├── packing_utils.py      
│   │       └── visualization.py   # Visualisation des résultats
│   │
│   ├── projects/                  # Implémentations spécifiques par projet
│   │    │
│   │    ├── lumbar_spine/          # Projet colonne vertébrale
│   │    │   ├── __init__.py
│   │    │   ├── csv_metadata.py								# Classe `CSVMetadata` (fusion des CSV)
│   │    │   ├── lumbar_dicom_tfrecord__dataset.py			# Implémentation de `DicomDataset` pour ce projet
│   │    │   ├── pipeline.py									# Implémentation de `BasePipeline`
│   │    │   └── train.py									# Script d'entraînement
│   │    │
│   │    └── brain_aneurysm/        # Projet anévrismes cérébraux
│   │        └── __init__.py
│   │
│   ├── RSNA_2024_Lumbar_Spine_Degenerative_Classification.py  # Main script  
│	└── RSNA_Intracranial_Aneurysm_Detection.py
│
├── data/                          # Lien symbolique vers les données (non versionné)
│
├── tests/                         # Tests (structure miroir de `src/`)
│   │
│   ├── e2e/ 
│   │   └── test_submission/
│   │
│   ├── fixtures/
│   │   ├── csv_samples_/
│   │   ├── dicom_samples_/
│   │   └── expected_outputs_/
│   │
│   ├── integration/
│   │   ├── test_pipeline/
│   │   └── test_data_flow/
│   │
│   ├── unit/
│   │   ├── test_data/
│   │   │   └──test_dicom_dataset.py
│   │   ├── test_models/
│   │   ├── test_utils/
│   │   │   └── test_logger.py
│   │   └── test_lumbar_spine_/
│   │       ├── test_lumbar_dicom_tfrecord_dataset.py
│   │       └── test_train.py
│   ├── test_rsna_lumbar_spine_degenerative_classification.py  # Test du script principal
│   └── conftest.py  # Fixtures partagées  
│
├── logs/                          # Logs (exclus du versioning)
│
├── scripts/                       # Scripts utilitaires
│   ├── create_symlink.ps1        # Crée le lien symbolique `data`
│   ├── setup_environment.ps1     # Configure l'environnement
│   └── run_pipeline.ps1          # Lance le pipeline
│
├── .gitignore                     # Exclut `data/`, `logs/`, `.vs/`, etc.
├── .vsconfig                      # Exclut `data/` de l'indexation Visual Studio
├── README.md                      # Documentation du projet
└── RSNA_2024_Lumbar_Spine_Degenerative_Classification.pyproj  # Projet Visual Studio
```
### **Description of the files and folders**
TBD