# RSNA 2024 Lumbar Spine Degenerative Classification

This repository hosts a modular, production-ready Deep Learning pipeline built with **TensorFlow / tf_keras** for the RSNA 2024 Lumbar Spine Degenerative Classification challenge. The goal is to detect and classify the severity of lumbar spine degenerative conditions (e.g., neural foraminal narrowing, canal stenosis, subarticular stenosis) and predict coordinate locations for medical annotations from volumetric DICOM imaging data.

The architecture uses a hybrid model: extracting 2D features from series slice images using modern backbones (e.g., MobileNetV2, ResNet50) and aggregating these features into a 3D context using convolutional temporal aggregation layer networks.

---

## Project Directory Structure

The repository is structured into decoupled, single-responsibility modules to isolate configuration management, shared core components, and project-specific execution logic.

```mermaid
graph TD
    Root["RSNA_Items_Detection_Classification (Root)"]
    
    subgraph ConfigAndEntry["Config & Entry"]
        Main["RSNA_2024_..._Classification.py (Entry)"]
        Survey["RSNA_input_data_survey.py (Data Survey)"]
        cfg_folder["src/config/"]
        cfg_folder --> cfg_files["config_loader.py<br/>schema.py<br/>lumbar_spine_config_*.yaml"]
    end
    
    subgraph CoreLibrary["Core Library"]
        core_folder["src/core/"]
        core_folder --> core_models["models/<br/>- backbone_2d.py<br/>- conv3d_aggregator.py<br/>- temporal_padding_layer.py<br/>- model_factory.py"]
        core_folder --> core_callbacks["callbacks/<br/>- system_resource_monitor_callback.py<br/>- dynamic_loss_balancer_callback.py<br/>- log_training_callback.py"]
        core_folder --> core_utils["utils/<br/>- dataset_utils.py<br/>- logger.py<br/>- system_stream_tee.py"]
    end
    
    subgraph ProjectsAndLogic["Projects & Logic"]
        proj_folder["src/projects/lumbar_spine/"]
        proj_folder --> proj_files["model_trainer.py<br/>tfrecord_files_manager.py<br/>csv_metadata_handler.py<br/>lumbar_dicom_tfrecord_dataset.py<br/>RSNA_lumbar_losses_and_metric.py"]
    end
    
    subgraph SupportDirectories["Support Directories"]
        scripts_folder["scripts/<br/>- create_hardlink.ps1<br/>- run_pipeline.ps1"]
        tests_folder["test/<br/>- unit/<br/>- integration/<br/>- conftest.py"]
        data_folder["data/ (DICOM studies & labels)"]
        logs_folder["logs/ (Session run records)"]
    end
    Root --> ConfigAndEntry
    Root --> CoreLibrary
    Root --> ProjectsAndLogic
    Root --> SupportDirectories
```

---

## Software Architecture & Design Patterns
The pipeline uses decoupled modules following single-responsibility principles. The diagram below represents the system architecture and runtime workflow.

```mermaid
graph TD
    subgraph Execution Entry
        Main["RSNA_2024_Lumbar_Spine_Degenerative_Classification.py<br/>(Main Orchestrator)"]
    end
    
    subgraph Data & Configuration Setup
        Config["ConfigLoader<br/>(Loads YAML configs)"]
        CSV["CSVMetadataHandler<br/>(Orchestrates labels & clinical CSVs)"]
        TFRecord["TFRecordFilesManager<br/>(Converts DICOMs to TFRecords)"]
    end
    
    subgraph Model & Training Architecture
        Factory["ModelFactory<br/>(Builds hybrid 2D/3D model)"]
        Losses["RSNA_lumbar_losses_and_metric.py<br/>(RSNALossAndMetricProvider)"]
        Trainer["ModelTrainer<br/>(Runs the fitting loop & logs)"]
    end
    
    subgraph Sub-Components
        Backbone["Backbone2D<br/>(MobileNetV2 / ResNet50)"]
        Aggregator["Conv3DAggregator<br/>(3D CNN Temporal Aggregation)"]
        Padding["TemporalPaddingLayer<br/>(Static shape padding)"]
        Dataset["LumbarDicomTFRecordDataset<br/>(tf.data pipeline builder)"]
        Callbacks["Training Callbacks<br/>(Loss Balancer, Resource Monitor)"]
    end
    Main --> Config
    Main --> CSV
    Main --> TFRecord
    Main --> Factory
    Main --> Losses
    Main --> Trainer
    Trainer --> Dataset
    Trainer --> Callbacks
    Factory --> Backbone
    Factory --> Aggregator
    Factory --> Padding
```

---

## System Initialization & Instantiation Sequence
The following sequence diagram maps the chronological order in which singletons, utility modules, handlers, and custom evaluation metrics are instantiated and invoked by the Main Orchestrator during the session setup phase.

```mermaid
sequenceDiagram
    autonumber
    actor User as Orchestrator / Main
    participant Config as ConfigLoader (Singleton)
    participant MetadataHandler as CSVMetadataHandler
    participant TFRecordMgr as TFRecordFilesManager
    participant ClassCount as DataFrameClassCount (Singleton)
    participant Provider as RSNALossAndMetricProvider
    participant Factory as ModelFactory (_get_or_build_model)
    participant Backbone as Backbone & Head Builder
    participant Metric as RSNAKaggleMetric
    participant DatasetBuilder as TFRecordDatasetBuilder
    participant Augmenter as DataAugmenter / Preprocessor
    participant Callbacks as CallbackFactory
    participant Trainer as ModelTrainer
    participant Evaluator as InferenceEngine / Predictor

    Note over User, TFRecordMgr: STEP 1: Initialization & Data Serialization (ETL Pipeline)
    User->>Config: get() / get_value() (YAML Configuration Loading)
    User->>MetadataHandler: Initialization (CSV Parsing & Label Mapping)
    
    User->>TFRecordMgr: generate_tfrecord_files()
    activate TFRecordMgr
    Note over TFRecordMgr: DICOM Reading, Volume Normalization & TFRecord Writing
    TFRecordMgr-->>User: actual_nb_tfrecord_files (Writes/updates cache.json)
    deactivate TFRecordMgr

    Note over User, ClassCount: STEP 2: Dynamic Balancing Weights Calculation
    User->>ClassCount: set_balancing_weights()
    activate ClassCount
    ClassCount->>ClassCount: _get() (Reloads fresh cache.json from disk)
    ClassCount->>ClassCount: _calculate_balancing_weights() (Inverse frequency strategy)
    deactivate ClassCount

    Note over User, Metric: STEP 3: Model Construction, Assembly & Compilation
    User->>Provider: Initialization ( Involque get_class_weights() )
    User->>Factory: Involkes model building (_get_or_build_model)
    activate Factory
    
    Factory->>Backbone: build_backbone() (EfficientNet / ResNet3D)
    Backbone-->>Factory: Feature Extractor Layers
    Factory->>Backbone: build_spine_classifier_head() (25 Distinct Task Heads)
    Backbone-->>Factory: Multi-task Functional Model Instance
    
    Factory->>Provider: get_loss()
    activate Provider
    Provider->>ClassCount: get_balancing_weights()
    Provider-->>Factory: Returns rsna_weighted_log_loss closure
    deactivate Provider

    Factory->>Provider: get_metrics()
    activate Provider
    Provider->>ClassCount: get_balancing_weights()
    Provider->>Metric: Instantiation of RSNAKaggleMetric(weights)
    activate Metric
    Note over Metric: Strict validation of injected balancing weights
    Metric-->>Provider: Ready metric instance
    deactivate Metric
    Provider-->>Factory: Returns metrics list (Loss Core + Accuracy)
    deactivate Provider

    Note over Factory: Keras Compilation (model.compile with run_eagerly=False)
    Factory-->>User: Fully compiled tf_keras.Model instance
    deactivate Factory

    Note over User, Trainer: STEP 4: Dataset Generation & Training Orchestration
    User->>DatasetBuilder: build_train_and_val_datasets(tfrecord_files)
    activate DatasetBuilder
    DatasetBuilder->>Augmenter: apply_augmentation_and_preprocessing() (3D Rotations, Z-score)
    DatasetBuilder-->>User: tf.data.Dataset (Optimized with Prefetch / Parallel Read)
    deactivate DatasetBuilder

    User->>Callbacks: build_callbacks_list() (Checkpoint, EarlyStopping, TensorBoard)
    Callbacks-->>User: list[tf_keras.callbacks.Callback]

    User->>Trainer: Instantiation of Trainer(model, train_ds, val_ds, callbacks)
    User->>Trainer: train_model()
    activate Trainer
    Note over Trainer: Execution of core Keras loop: model.fit()
    Trainer-->>User: TrainingHistory Object
    deactivate Trainer

    Note over User, Evaluator: STEP 5: Post-Processing, Global Evaluation & Submission
    User->>Evaluator: run_inference(test_ds)
    activate Evaluator
    Note over Evaluator: Best weights restoration & final testing inference
    Evaluator->>Evaluator: format_submission() (Generates competition-compliant submission.csv)
    Evaluator-->>User: Completion status / Output file ready
    deactivate Evaluator
```

---

## Losses & Evaluation Metrics Architecture (UML Class Diagram)
The following UML class diagram illustrates the object-oriented design and relationships of the components managing class counts, dataset balancing, custom losses, and the official Kaggle competition evaluation metric framework.

```mermaid
classDiagram
    direction TD
    class SingletonMeta {
        <<metaclass>>
    }
    class DataFrameClassCount {
        -Path _cache
        -Dict _severity_labels_counts
        -Tensor _balancing_weights
        +__init__()
        -_get() Dict
        -_calculate_balancing_weights() Tensor
        +get_balancing_weights() Tensor
    }
    class tf_keras_metrics_Metric {
        <<external>>
    }
    class RSNALossAndMetricProvider {
        -Dict _config
        -Logger _logger
        -Tensor _balancing_weights
        -Tensor _class_weights
        +__init__(logger: Logger)
        +get_loss() Function
        +get_metrics() RSNAKaggleMetric
    }
    class RSNAKaggleMetric {
        -Dict _config
        -Tensor _class_weights
        -Tensor _balancing_weights
        -Variable total_loss
        -Variable count
        +__init__(class_weights, balancing_weights, logger, name, **kwargs)
        +update_state(y_true, y_pred, sample_weight) None
        +result() Tensor
        +reset_state() None
    }
    SingletonMeta <|-- DataFrameClassCount : instance-of
    tf_keras_metrics_Metric <|-- RSNAKaggleMetric : inherits
    RSNALossAndMetricProvider ..> RSNAKaggleMetric : instantiates
    RSNALossAndMetricProvider ..> DataFrameClassCount : requests weights
```

---

## Key Engine Features

### Dynamic Loss Balancing
Because classification (weighted log loss) and coordinate regression (MSE) have different magnitudes, the model uses a custom DynamicLossBalancerCallback. This callback dynamically adjusts the location_output loss weight variable based on the relative convergence rates of both tasks across epochs.

### OOM Prevention Circuit
Processing large volumetric 3D datasets in batches can cause Out-Of-Memory (OOM) errors. The SystemResourceMonitorCallback actively monitors RAM/CPU usage at the end of each batch/epoch and triggers a graceful emergency training stop if memory exceeds the threshold configured in the YAML (e.g. 90%). This ensures that checkpoints are saved before an OOM occurs.

### Fail-Safe Resume Logic
If a training run is interrupted, the entry point attempts to load the complete saved Keras model. In case loading fails (e.g. serialization issues with custom Keras Layers), a weight salvage fallback builds a fresh architecture from the ModelFactory and maps the saved weights by name before continuing.

---

## Getting Started

1. **Environment and Symlinks**
The pipeline utilizes symlinks to select configurations depending on the environment (Kaggle kernel vs local Windows machine).

Create a hardlink/symlink to point to the correct configuration:

```powershell
.\scripts\create_hardlink.ps1
```

2. **Surveying Dataset**
Inspect your input raw DICOM files to check spacing, image formats, and depths before starting preprocessing:

```bash
python src/RSNA_input_data_survey.py
```

3. **Running Training**
Start the end-to-end pipeline (Preprocessing -> TFRecord generation -> Training -> Evaluation):

```powershell
.\scripts\run_pipeline.ps1
```

---

## Tests
The repository is fully testable using pytest. The test/ directory contains unit tests for utils, models, dataset loaders, as well as full integration tests verifying data flows.

Run the test suite:

```bash
pytest
```
