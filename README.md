 ## **Project Structure**

 The project is organized into several key directories and files to facilitate development, testing, and deployment. Below is an overview of the main components:

 ```
RSNA_Items_Detection_Classification/
│
├── src/
│   │
│   ├── config/                    # Default global settings
│   │   ├── __init__.py
│   │   ├── config_loader_.py                   # settings loader
|   |   ├── lumbar_spine_config_kaggle.yaml     # kaggle environment default setting
│   │   ├── lumbar_spine_config_windows.yaml    # Windows environment default setting
│   │   └── lumbar_spine_config.yaml            # Current setting
│   │
│   ├── core/
│   │   │
│   │   ├── models/               # Deep learning models (2D/3D)
│   │   │   ├── __init__.py
│   │   │   ├── backbone_2d.py             # 2D models (YOLO, ResNet, MobileNetV2 ...)
│   │   │   ├── conv3d_aggregator.py       # 3D aggregation logic
│   │   │   ├── temporal_padding_layer.py  # Custom Keras layers
│   │   │   └── model_factory.py           # Factory pattern for model instantiation
│   │   │
│   │   └── utils/                 # Shared helpers
│   │       ├── __init__.py
│   │       ├── clean_logs.py                   # Log rotation: automates deletion of outdated logs (>30 days)
│   │       ├── dataset_utils.py                        # TFRecord parsing, Normalization, Augmentation
│   │       ├── monitoring_utils.py
│   │       ├── log_training_callbacks.py    # Training monitor: logs metrics (Loss/Acc), RAM usage, and step timing
│   │       ├── logger.py                               # Unified_logging_system
│   │       ├── system_resource_monitor_callbacks.py    # Safety circuit: monitors RAM/CPU to trigger emergency stop (OOM prevention)
│   │       └── system_stream_tee.py                    # Redirect both standard and error output toward a log file
│   │
│   ├── projects/                  # Project-specific implementations
│   │    │
│   │    └── lumbar_spine/         # Lumbar_spine_project
│   │        ├── __init__.py
│   │        ├── csv_metadata_handler.py                     # Data orchestrator (Joins CSV/Labels)
│   │        ├── lumbar_dicom_tfrecord_dataset.py            # Data pipeline: 3D volumes builder from TFRecords
│   │        ├── model_trainer.py                            # Training & Validation loops
|   |        ├── RSNA_lumbar_losses_and_metric.py            # Optimization: Weighted Log Loss & Kaggle competition metrics
|   |        └── tfrecord_files_manager.py            # I/O & Sharding manager
│   │
│   ├── RSNA_2024_Lumbar_Spine_Degenerative_Classification.py  # Main script  
│   └── RSNA_input_data_survey.py  # Data Profiler: analyzes DICOM consistency (spacing, format) and series depth
│
├── data/                          # Symlink to data (not versioned)
│
├── tests/                         # Tests repository
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
    │   ├── __init__.py
│   │   ├── test_data_handlers/
│   │   │   └──test_dicom_dataset.py
│   │   ├── test_models/
│   │   ├── test_utils/
│   │   │   └── test_logger.py
│   │   ├── test_lumbar_spine_/
│   │   ├── test_lumbar_dicom_tfrecord_dataset.py
│   │   │       └── test_train.py
│   │   └── test_rsna_lumbar_spine_degenerative_classification.py  # Test du script principal
|   |
│   └── conftest.py  # Shared fixtures
│
├── logs/                          # Logs (not versioned)
│
├── scripts/                      # 
│   ├── create_symlink.ps1        # 
│   ├── setup_environment.ps1     # 
│   └── run_pipeline.ps1          #
│
├── .gitignore                     # Exclude `data/`, `logs/`, `.vs/`, etc.
└── README.md                      # Project documentation

### **Files and folders description**
TBD