 ## **Project Structure**

 The project is organized into several key directories and files to facilitate development, testing, and deployment. Below is an overview of the main components:

 ```
RSNA_2024_Lumbar_Spine_Degenerative_Classification/
│
├── src/
│   ├── core/                     # Classes de base génériques (abstraites)
│   │   ├── data/                 # Gestion des données (DICOM, CSV, métadonnées)
│   │   │   ├── __init__.py
│   │   │   ├── dicom_dataset.py  # Classe `DicomDataset` (abstraite + implémentations)
│   │   │   ├── csv_metadata.py   # Classe `CSVMetadata` (fusion des CSV)
│   │   │   └── data_loader.py    # Chargement des données (utilise `DicomDataset` et `CSVMetadata`)
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
│   │       ├── logger.py          # Gestion des logs
│   │       ├── file_utils.py      # Manipulation de fichiers
│   │       └── visualization.py   # Visualisation des résultats
│   │
│   ├── projects/                  # Implémentations spécifiques par projet
│   │   ├── lumbar_spine/          # Projet colonne vertébrale
│   │   │   ├── __init__.py
│   │   │   ├── config.yaml        # Configuration spécifique
│   │   │   ├── dataset.py         # Implémentation de `DicomDataset` pour ce projet
│   │   │   ├── pipeline.py        # Implémentation de `BasePipeline`
│   │   │   └── train.py           # Script d'entraînement
│   │   │
│   │   └── brain_aneurysm/        # Projet anévrismes cérébraux
│   │       ├── __init__.py
│   │       ├── config.yaml
│   │       ├── dataset.py
│   │       ├── pipeline.py
│   │       └── train.py
│   │
│   └── config/                    # Configurations globales et par défaut
│       ├── __init__.py
│       ├── default_config.yaml    # Configuration par défaut
│       └── config_loader.py       # Chargeur de configuration
│
├── data/                          # Lien symbolique vers les données (non versionné)
│
├── tests/                         # Tests (structure miroir de `src/`)
│   ├── unit/
│   │   ├── test_data/
│   │   ├── test_models/
│   │   └── test_utils/
│   │
│   ├── integration/
│   │   ├── test_pipeline/
│   │   └── test_data_flow/
│   │
│   └── e2e/
│       └── test_submission/
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