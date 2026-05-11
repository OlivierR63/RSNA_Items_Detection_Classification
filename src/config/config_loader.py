# coding: utf-8

from pathlib import Path
from typing import Any, Dict
import yaml
import logging
import json
import numpy as np
from src.core.utils.singleton_meta import SingletonMeta


class ConfigLoader(metaclass=SingletonMeta):
    """
    A singleton class that loads configuration settings from a YAML file 
    and resolves relative paths.

    This class handles the parsing of the configuration file and ensures that
    all relevant directory and file paths (e.g., DICOM and CSV files)
    are made absolute relative to the location of the configuration file itself.
    Since it implements the singleton pattern, only one instance of the 
    configuration is maintained throughout the application's lifecycle.
    """

    def __init__(self, config_path: str = None) -> None:
        """
        Initializes the singleton instance by reading the YAML file and resolving paths.
        Subsequent calls to the constructor return the existing instance without
        re-executing this logic.

        Args:
            config_path (str, optional): File system path to the YAML configuration file.
            Required only for the first instantiation

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the YAML file is malformed.
        """
        # Check if the singleton has already been initialized to avoid redundant setup
        if hasattr(self, '_config'):
            return  # Already initialized by a previous call

        # Ensure the configuration path is provided during the initial call
        if config_path is None:
            raise ValueError(
                "ConfigLoader must be initialized with a config_path on its first call."
            )
        
        # Check file existence
        config_file_path = Path(config_path).resolve()
        if not config_file_path.exists():
            raise FileNotFoundError(f"Configuration file {config_path} not found.")

        # Determine the directory containing the configuration file.
        # All relative paths in the config will be resolved based on this directory.
        config_dir = config_file_path.parent

        # Load the configuration data first (Initialization of self._config)
        try:
            with config_file_path.open('r') as f:
                self._config: Dict[str, Any] = yaml.safe_load(f) or {}  # Handles empty YAML files

        except yaml.YAMLError as e:
            raise ValueError(f"Error loading YAML configuration file: {e}")

        # Store the determined root directory (the YAML file's location)
        self._config["root_dir"] = str(config_dir)

        # At this time, just initialize the attribute "series_depth"
        self._config["series_depth"] = None

        # Reference to the paths dictionary to avoid repetitive lookups
        paths_cfg = self._config.get('paths', {})

        # --- Resolve Core Relative Paths ---
        # Iterate through known keys to convert relative paths (starting with '.') to absolute
        core_keys = [
            "dicom_studies",
            "tfrecord",
            "tfrecord_metadata_cache",
            "output",
            "inspection",
            "checkpoint",
            "log_mirror"
        ]

        for key in core_keys:
            if key in paths_cfg:
                val = paths_cfg[key]
                # Check if the value is a string and starts with a dot (relative path)
                if isinstance(val, str) and val.startswith('.'):
                    # Resolve path relative to the config file location
                    resolved_path = (config_dir / val).resolve()
                    paths_cfg[key] = str(resolved_path)

        # --- Resolve CSV File Paths ---
        # Handle nested dictionary for CSV file locations
        if "csv" in paths_cfg:
            csv_dict = paths_cfg["csv"]
            for csv_key in csv_dict:
                val = csv_dict[csv_key]
                if isinstance(val, str) and val.startswith('.'):
                    resolved_csv = (config_dir / val).resolve()
                    csv_dict[csv_key] = str(resolved_csv)

        # Final check: Verify the compliance of the config dictionary
        self._check_config_compliance()

    def set_value(self, key: str, value: Any) -> None:
        """
        Sets a configuration value for a specific key.

        Args:
            key (str): The configuration key to update.
            value (Any): The value to assign to the key.
        """
        self._config[key] = value

    def get_value(self, key: str, default: str=None) -> Any:
        """
        Retrieves a single configuration value by its key.

        Args:
            key (str): The name of the configuration parameter to retrieve.
            default (Any, optional): Default value if key is not found.

        Returns:
            Any: The configuration value associated with the key.
        """
        return self._config.get(key, default)

    def get(self) -> dict:
        """
        Returns the entire processed configuration dictionary.

        Returns:
            dict: The complete configuration dictionary, with all relative
                  paths resolved to absolute paths.
        """
        return self._config

    def calculate_series_depth(
        self,
        tfrecord_cache_dir: str,
        dicom_studies_dir: str,
        percentile: int,
        logger: logging.Logger
    ) -> int:
        """
        Determines the target depth (number of slices) for image series normalization.

        The depth is calculated as a specific percentile of the distribution of
        slice counts across all available studies. This ensures a consistent input
        shape for the model while limiting the impact of extreme outliers.

        The function implements a smart cache mechanism that persists the
        calculated depth in a JSON file. The cache is automatically invalidated
        if the number of studies changes or if any study directory has been
        modified (based on filesystem timestamps).

        Args:
            config (Dict[str, str]): Configuration dictionary containing paths
                and threshold parameters.
            logger (logging.Logger, optional): Logger for tracking progress and
                cache status. Defaults to None.

        Returns:
            int: The calculated reference depth (number of slices) based on
                the configured percentile.
        """
        if logger:
            logger.info("Starting  function calculate_series_depth")

        TFRecord_cache_dir = Path(tfrecord_cache_dir).resolve()
        dicom_studies_dir = Path(dicom_studies_dir).resolve()

        if logger:
            logger.info(f"dicom_studies_dir = {dicom_studies_dir}")

        cache_file = TFRecord_cache_dir / "cache.json"

        studies_dirs_list = [study for study in dicom_studies_dir.iterdir() if study.is_dir()]
        studies_count = len(studies_dirs_list)

        if logger:
            logger.info(f"Function calculate_series_depth: found {studies_count} studies")

        if studies_count == 0:
            return 0

        series_depth = None

        # Smart Cache Management
        if cache_file.exists():
            try:
                cache_mtime = cache_file.stat().st_mtime
                with cache_file.open('r') as f:
                    cache_data = json.load(f)

                    # Check 1: Invalidate cache if the number of files has changed
                    if cache_data.get('studies_count') == studies_count:

                        # Check 2: Has any study directory been modified since cache creation?
                        # Note: Modifying / adding files inside a folder update sits time
                        is_cache_stale = any(
                            s.stat().st_mtime > cache_mtime for s in studies_dirs_list
                        )

                        if not is_cache_stale:
                            series_depth = cache_data['series_depth']
                            if logger:
                                logger.info("Series depth loaded from cache")

            except Exception as e:
                if logger:
                    warning_msg = f"Cache read failed, recalculating: {e}"
                    logger.warning(warning_msg)
                pass

        # Depth calculation (Only if series_depth has not been recovered from the cache)
        if series_depth is None:
            series_depth = 0

            depth_list = []

            for study in studies_dirs_list:
                # Find the maximum depth across series in this study, defaults to 0
                depth_list.extend(
                    [
                        len(list(series.glob('*.dcm')))
                        for series in study.iterdir()
                        if series.is_dir()
                    ]
                )

            series_depth = int(np.percentile(depth_list, percentile))

            # Save to Cache
            try:
                TFRecord_cache_dir.mkdir(parents=True, exist_ok=True)
                with cache_file.open('w') as cache_file:
                    json.dump(
                        {'studies_count': studies_count, 'series_depth': series_depth},
                        cache_file
                    )

            except Exception as e:
                warning_msg = f"Unable to save the cache file in {TFRecord_cache_dir} : {e}"
                if logger:
                    logger.warning(warning_msg)

        if logger:
            logger.info(
                "Function calculate_series_depth : "
                f"calculated series depth = {series_depth}"
            )

        return series_depth

    def _check_config_compliance(self) -> None:
        """
        Validates the integrity and completeness of the YAML configuration.

        This method performs a structural check of the configuration dictionary. 
        It enforces a 'fail-fast' policy: if a mandatory key is missing or 
        incorrectly nested, the process is terminated immediately to prevent 
        downstream failures in the data pipeline or training loop.

        Raises:
            ValueError: If any required configuration key (e.g., 'paths', 
                'dicom_studies', 'tfrecord') is missing from the YAML file.
        
        Note:
            This is an internal validation step typically called during 
            the initialization of the ConfigLoader.
        """

        paths_cfg = self._config.get("paths", None)
        if paths_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'paths' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        dicom_studies_dir = paths_cfg.get('dicom_studies', None)
        if dicom_studies_dir is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'paths -> dicom_studies' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        tfrecord_dir = paths_cfg.get('tfrecord', None)
        if tfrecord_dir is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'paths -> tfrecord' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        tfrecord_metadata_cache_dir = paths_cfg.get('tfrecord_metadata_cache', None)
        if tfrecord_metadata_cache_dir is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'paths -> tfrecord_metadata_cache' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        output_dir = paths_cfg.get('output', None)
        if output_dir is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'paths -> output' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        checkpoint_dir = Path(paths_cfg.get("checkpoint", None))
        if checkpoint_dir is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'paths -> checkpoint' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        log_mirror_dir = paths_cfg.get('log_mirror', None)
        if log_mirror_dir is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'paths -> log_mirror' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        tf_cache_dir = paths_cfg.get('tf_cache', None)
        if tf_cache_dir is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'paths -> tf_cache' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        inspection_dir = paths_cfg.get('inspection', None)
        if inspection_dir is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'paths -> inspection' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        csv_cfg = paths_cfg.get('csv', None)
        if csv_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'paths -> csv' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        series_description_csv = csv_cfg.get('series_description', None)
        if series_description_csv is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'paths -> csv -> series_description' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        label_coordinates_csv = csv_cfg.get('label_coordinates', None)
        if label_coordinates_csv is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'paths -> csv -> label_coordinates' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        instances_series_format_csv = csv_cfg.get('instances_series_format', None)
        if instances_series_format_csv is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'paths -> csv -> instances_series_format' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        train_csv = csv_cfg.get('train', None)
        if train_csv is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'paths -> csv -> train' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        data_specs_cfg = self._config.get('data_specs', None)
        if data_specs_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'data_specs' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        series_depth_percentile = data_specs_cfg.get('series_depth_percentile', None)
        if series_depth_percentile is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'data_specs -> series_depth_percentile' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        max_records_per_frame = data_specs_cfg.get('max_records_per_frame', None)
        if max_records_per_frame is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'data_specs -> max_records_per_frame' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        dataset_buffer_size = data_specs_cfg.get('dataset_buffer_size_mb', None)
        if dataset_buffer_size is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'data_specs -> dataset_buffer_size_mb' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        models_cfg = self._config.get('models', None)
        if models_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'models' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        backbone_2d_cfg = models_cfg.get('backbone_2d', None)
        if backbone_2d_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'models -> backbone_2d' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        type_2d = backbone_2d_cfg.get('type', None)
        if type_2d is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'models -> backbone_2d -> type' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        img_shape = backbone_2d_cfg.get('img_shape', None)
        if img_shape is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'models -> backbone_2d -> img_shape' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        freeze = backbone_2d_cfg.get('freeze', None)
        if freeze is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'models -> backbone_2d -> freeze' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        scaling_dict = backbone_2d_cfg.get('scaling', None)
        if scaling_dict is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'models -> backbone_2d -> scaling' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        min_scaling_value, max_scaling_value = (
            scaling_dict.get("min", None),
            scaling_dict.get("max", None)
        )

        if None in (min_scaling_value, max_scaling_value):
            critical_msg = (
                "Fatal error in normalize_image: "
                "the setting variable 'models -> backbone_2d -> scaling' is required "
                "but the dictionary values are invalid. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        if not (
            isinstance(min_scaling_value, (int, float))
            and isinstance(max_scaling_value, (int, float))
        ):
            raise ValueError("Scaling values must be numeric (int or float).")

        if min_scaling_value > max_scaling_value:
            critical_msg = (
                "Fatal error in normalize_image: 'min' cannot be greater than 'max' "
                "in scaling configuration."
            )
            raise ValueError(critical_msg)

        head_3d_cfg = models_cfg.get('head_3d', None)
        if head_3d_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'models -> head_3d' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        type_3d = head_3d_cfg.get('type', None)
        if type_3d is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'models -> head_3d -> type' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        filters = head_3d_cfg.get('filters', None)
        if filters is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'models -> head_3d -> filters' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        training_cfg = self._config.get('training', None)
        if training_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'training' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        batch_size = training_cfg.get('batch_size', None)
        if batch_size is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'training -> batch_size' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        if batch_size <= 0:
            critical_msg = (
                "Fatal error in prepare_training_and_validation_datasets: "
                "'batch_size' shall be strictly positive"
            )
            raise ValueError(critical_msg)

        nb_epoch = training_cfg.get('epochs', None)
        if nb_epoch is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'training -> epochs' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        train_split_ratio = training_cfg.get('train_split_ratio', None)
        if train_split_ratio is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'training -> train_split_ratio' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        loss_balancer_cfg = training_cfg.get('loss_balancer', None)
        if loss_balancer_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'training -> loss_balancer' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        momentum = loss_balancer_cfg.get('momentum', None)
        if momentum is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'training -> loss_balancer -> momentum' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        min_weight = loss_balancer_cfg.get('min_weight', None)
        if min_weight is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'training -> loss_balancer -> min_weight' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        max_weight = loss_balancer_cfg.get('max_weight', None)
        if max_weight is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'training -> loss_balancer -> max_weight' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        optimizer_cfg = self._config.get('optimizer', None)
        if optimizer_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'optimizer' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        optim_type = optimizer_cfg.get('type', None)
        if optim_type is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'optimizer -> type' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        learning_rate = optimizer_cfg.get('learning_rate', None)
        if learning_rate is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'optimizer -> learning_rate' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        clipnorm = optimizer_cfg.get('clipnorm', None)
        if clipnorm is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'optimizer -> clipnorm' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        callbacks_cfg = self._config.get('callbacks', None)
        if callbacks_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'callbacks' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        patience = callbacks_cfg.get('patience', None)
        if patience is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'callbacks -> patience' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        resume_mode = callbacks_cfg.get('resume_mode', None)
        if resume_mode is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'callbacks -> resume_mode' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        if resume_mode not in ["best", "last"]:
            critical_msg = (
                "Fatal error: the parameter 'callbacks -> resume_mode' "
                "was not properly set. Only two values are permitted: 'best' or 'last'. "
                "Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        compilation_cfg = self._config.get('compilation', None)
        if compilation_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'compilation' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        loss_weights_cfg = compilation_cfg.get('loss_weights', None)
        if loss_weights_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'compilation -> loss_weights' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        severity_cfg = loss_weights_cfg.get('severity_output', None)
        if severity_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'compilation -> loss_weights -> severity_output' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        location_cfg = loss_weights_cfg.get('location_output', None)
        if location_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'compilation -> loss_weights -> location_output' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        class_weights_cfg = compilation_cfg.get('class_weights', None)
        if class_weights_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'compilation -> class_weights' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        normal_mild_weights_cfg = class_weights_cfg.get('Normal/Mild', None)
        if normal_mild_weights_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'compilation -> class_weights -> Normal/Mild' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        moderate_cfg = class_weights_cfg.get('Moderate', None)
        if moderate_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'compilation -> class_weights -> Moderate' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        severe_cfg = class_weights_cfg.get('Severe', None)
        if severe_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'compilation -> class_weights -> Severe' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        run_eagerly_cfg = compilation_cfg.get('run_eagerly', None)
        if run_eagerly_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'compilation -> run_eagerly' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        system_cfg = self._config.get("system", None)
        if system_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'system' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        nb_cores = system_cfg.get("nb_cores", None)
        if nb_cores is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'system -> nb_cores' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        log_retention_days = system_cfg.get("log_retention_days", None)
        if log_retention_days is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'system -> log_retention_days' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        seed = system_cfg.get("seed", None)
        if seed is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'system -> seed' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        mem_threshold_percent = system_cfg.get("memory_threshold_percent", None)
        if mem_threshold_percent is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'system -> memory_threshold_percent' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        chunksize = system_cfg.get("chunksize", None)
        if chunksize is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'system -> chunksize' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        dataset_steering_cfg = self._config.get("dataset_steering", None)
        if dataset_steering_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'dataset_steering' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        interleave_cfg = dataset_steering_cfg.get("interleave", None)
        if interleave_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'dataset_steering -> interleave' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        nb_parallel_files = interleave_cfg.get("parallel_files", None)
        if nb_parallel_files is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'dataset_steering -> interleave -> parallel_files' "
                "is required but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        block_per_file = interleave_cfg.get("block_per_file", None)
        if block_per_file is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'dataset_steering -> interleave -> block_per_file' "
                "is required but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        deterministic = interleave_cfg.get("deterministic", None)
        if deterministic is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'dataset_steering -> interleave -> deterministic' "
                "is required but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        group_studies = dataset_steering_cfg.get("group_studies", None)
        if group_studies is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'dataset_steering -> group_studies' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        prefetch_batches = dataset_steering_cfg.get("prefetch_batches", None)
        if prefetch_batches is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'dataset_steering -> prefetch_batches' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        num_parallel_calls = dataset_steering_cfg.get("num_parallel_calls", None)
        if num_parallel_calls is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'dataset_steering -> num_parallel_calls' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        use_cache = dataset_steering_cfg.get("use_cache", None)
        if use_cache is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'dataset_steering -> use_cache' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        logging_cfg = self._config.get("logging", None)
        if logging_cfg is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'logging' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        logging_level = logging_cfg.get("level", None)
        if logging_level is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'logging -> level' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        console_display = logging_cfg.get("console_display", None)
        if console_display is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'logging -> console_display' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)

        use_json = logging_cfg.get("use_json", None)
        if use_json is None:
            critical_msg = (
                "Fatal error in check_config_compliance: "
                "the setting variable 'logging -> use_json' is required "
                "but was not found. Please check your YAML file structure."
            )
            raise ValueError(critical_msg)
