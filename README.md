 ## **Project Structure**

 The project is organized into several key directories and files to facilitate development, testing, and deployment. Below is an overview of the main components:

 ```
 project_root/
│
├── src/                          # Code source principal
│   ├── data/                     # Scripts de gestion des données
│   ├── models/                   # Modèles TensorFlow
│   ├── pipeline/                 # Pipeline TFX
│   └── utils/                    # Utilitaires (lecture DICOM, fusion CSV, etc.)
│
├── tests/                        # Répertoire principal des tests
│   ├── unit/                     # Tests unitaires (fonctions isolées)
│   │   ├── test_data/            # Tests sur la gestion des données
│   │   ├── test_models/          # Tests sur les modèles
│   │   └── test_utils/           # Tests sur les utilitaires
│   │
│   ├── integration/              # Tests d'intégration (interactions entre composants)
│   │   ├── test_pipeline/        # Tests du pipeline TFX
│   │   └── test_data_flow/       # Tests du flux de données (DICOM → CSV → modèle)
│   │
│   ├── e2e/                      # Tests end-to-end (bout en bout)
│   │   └── test_submission/      # Tests sur la génération du fichier submission.csv
│   │
│   ├── fixtures/                 # Données de test (DICOM/CSV mockés)
│   │   ├── dicom_samples/        # Exemples de fichiers DICOM simplifiés
│   │   ├── csv_samples/          # Exemples de fichiers CSV réduits
│   │   └── expected_outputs/     # Résultats attendus (ex. : submission.csv)
│   │
│   ├── conftest.py               # Configuration pytest (fixtures globales)
│   └── requirements.txt          # Dépendances spécifiques aux tests
│
├── logs/                         # Logs des tests et warnings
├── .github/workflows/            # CI/CD (GitHub Actions)
└── README.md                     # Documentation des tests
```
### **Description of the files and folders**
TBD