# coding: utf-8

import tensorflow as tf
from typing import Dict, Tuple, List, Optional
from src.core.data_handlers.dicom_tfrecord_dataset import DicomTFRecordDataset
from src.core.utils.logger import log_method
from src.projects.lumbar_spine.tfrecord_files_manager import TFRecordFilesManager
import logging
import random


class LumbarDicomTFRecordDataset(DicomTFRecordDataset):
    """
        TensorFlow Dataset for loading DICOM TFRecords.
    """

    def __init__(self, config: dict, logger: Optional[logging.Logger] = None, series_depth = 1) -> None:
        self._MAX_RECORDS = config['max_records']
        self._MAX_RECORDS_FLAT = self._MAX_RECORDS * 3
        self._series_depth = series_depth
        super().__init__(config, logger=logger)  # Pass the logger to parent class
        self._logger.info("Initializing LumbarDicomTfRecordDataset")


    @log_method()
    def build_tf_dataset_pipeline(
        self,
        *,
        batch_size: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[int, tf.data.Dataset, int, tf.data.Dataset]:
        
        """
            Creates an optimized TensorFlow Dataset for 3D analysis.
            This pipeline groups DICOM frames into series, sorts them anatomically,
            and applies symmetric padding.
        """
        logger = logger or self._logger
        logger.info(f"Creating 3D-aware TF Dataset with batch_size={batch_size}")

        # Priority: explicit argument > config file
        final_batch_size = batch_size or self._config.get('batch_size', 8)

        try:
            # 1. List files and shuffle them
            all_tfrecord_files = tf.io.gfile.glob(self._tfrecord_pattern)

            # 2. Shuffle the files
            random.seed(42) 
            random.shuffle(all_tfrecord_files)

            # 3. Calculate the total number of files
            nb_tfrecord_files = len(all_tfrecord_files)

            # 3. Define the train / validation ratio
            train_ratio = self._config.get('train_split_ratio', 0.8)

            # 4. Split the train et validation datasets
            split_idx = int(nb_tfrecord_files * train_ratio)
            train_list = all_tfrecord_files[:split_idx]
            val_list = all_tfrecord_files[split_idx:]

            # 5. The TensorFlow world starts here
            nb_train = len(train_list)
            train_ds = self._generate_tfrecord_dataset(train_list, batch_size=final_batch_size)
            nb_val = len(val_list)
            val_ds = self._generate_tfrecord_dataset(val_list, batch_size=final_batch_size)

            return nb_train, train_ds, nb_val, val_ds

        except Exception as e:
            logger.error(f"Function build_tf_dataset_pipeline failed: {str(e)}", exc_info=True)
            raise

    @log_method()
    def _generate_tfrecord_dataset(
        self,
        tfrecord_list,
        batch_size,
        logger: Optional[logging.Logger]=None
    ) -> tf.data.Dataset:

        """
        Constructs a specialized tf.data.Dataset pipeline for 3D lumbar spine analysis.

        This pipeline handles the transition from flat TFRecord entries to structured 
        3D multi-series volumes. It performs the following sequence:
        1. Interleaves TFRecord files into a flat stream of serialized examples.
        2. Parses examples into image tensors and metadata (with coordinate normalization).
        3. Windows elements by Study ID to gather all frames for T1, T2, and Axial series.
        4. Reconstructs and sorts 3D volumes while ensuring spatial consistency.
        5. Formats the data into the multi-input dictionary required by the model.

        Args:
            tfrecord_list: List of paths to the TFRecord files.
            logger: Optional logger for tracking pipeline initialization.

        Returns:
            A tf.data.Dataset yielding tuples of (inputs_dict, targets_dict).
        """

        logger = logger or self._logger
        logger.info("Starting function _generate_tfrecord_dataset")

        # tfrecord_list = [str(int(p)) for p in tfrecord_list]
        
        try:
            # 1. Extract records from files (Flattening the nested structure)
            buffer_size=1024*1024*4 # buffer_size set to 4MB.
            dataset = tf.data.Dataset.from_tensor_slices(tfrecord_list).interleave(
                lambda x: tf.data.TFRecordDataset(x, buffer_size=buffer_size),
                #cycle_length=tf.data.AUTOTUNE,  # Read several studies in parallel
                cycle_length=1,
                block_length=1,      # Exhausts each file's content before proceeding to the next in the cycle.
                #num_parallel_calls=tf.data.AUTOTUNE,
                num_parallel_calls=1,
                deterministic=True
            )

            # 2. Parse individual TFRecord elements (frames)
            tfrecord_files_manager = TFRecordFilesManager(self._config, self._logger)
            tfrecord_files_manager.set_series_depth(self._series_depth)
            dataset = dataset.map(
                tfrecord_files_manager._parse_tfrecord_single_element,
                #num_parallel_calls=tf.data.AUTOTUNE
                num_parallel_calls=1
            )

            # 3. Group elements by series (using study_id and series_id)
            # We assume series_id is unique enough or combine IDs if necessary.
            # window_size=self._series_depth matches the maximum DICOM files per series.
            dataset = dataset.group_by_window(
                key_func=lambda _1, meta, _2: tf.cast(meta['study_id'], tf.int64),
                
                # Collect all TFRecord content (encompassing all 3 series) into a single buffer.
                reduce_func=lambda key, window: window.batch(3*self._series_depth),
                
                # Does not supports parallel processing of several files.
                window_size=1
            )

            # 4. Reconstruct spatial consistency by sorting slices along the Z-axis according
            #  to DICOM Instance Number and perform symmetric padding (Pure TF) when necessary.
            dataset = dataset.map(
                lambda images, metadata, labels: tfrecord_files_manager._process_study_multi_series(images, metadata, labels),
                #num_parallel_calls=tf.data.AUTOTUNE
                num_parallel_calls=1
            )

            dataset = dataset.cache()

            # 5. Final formatting for ModelFactory and batching
            dataset = dataset.map(
                lambda image, series_id, label: self._format_for_model(image, series_id, label),
                #num_parallel_calls=tf.data.AUTOTUNE
                num_parallel_calls=1
            )

            # Repeat before batching to ensure a continuous stream of data
            # and prevent partial batches at the end of an epoch.
            #dataset=dataset.repeat() # Repeat first to ensure an infinite flow.
            
            # Batching several studies (patients) together. 
            # Note: Set batch_size carefully based on GPU VRAM (3D volumes are heavy).
            dataset = dataset.batch(batch_size, drop_remainder=True)
            
            # Prefetch the next batch in the background to hide latency
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset

        except Exception as e:
            logger.error(f"Error creating dataset pipeline: {str(e)}", exc_info=True)
            raise

    def _format_for_model(
        self, 
        study_volumes: Tuple[
                                Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
                                Tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
                                Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
                                ], 
        study_id_tf: tf.Tensor, 
        labels: Dict[str, tf.Tensor]
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """
        Final mapping stage that adapts study-level volumes and metadata 
        to the specific multi-input/multi-output signature of the model.

        Args:
            study_volumes (tuple): Tensors for (Sagittal T1, Sagittal T2, Axial T2) 
                                   each with shape (self._series_depth, H, W, C).
            study_id_tf (tf.Tensor): The unique identifier for the study.
            labels (dict): Dictionary containing the 'records' tensor for diagnosis.
    
        Returns:
            tuple: (inputs_dict, targets_dict) ready for model.fit().
        """

        # Retrieve the config of the expected shapes
        img_cfg = self._config['model_2d']['img_shape']
        target_shape = [self._series_depth, img_cfg[0], img_cfg[1], img_cfg[2]]

        # Unpack processed volumes from the previous study-level processing step
        sag_t1, sag_t2, axial = study_volumes

        sag_t1[0].set_shape(target_shape)
        sag_t2[0].set_shape(target_shape)
        axial[0].set_shape(target_shape)

        # --- 2. Build Inputs Dictionary ---
        # These keys MUST exactly match the names defined in ModelFactory.build_multi_series_model()
        features = {
            "study_id": tf.reshape(tf.cast(study_id_tf, tf.float32), [1]),
            "img_sag_t1": tf.cast(sag_t1[0], tf.float32),
            "series_sag_t1": tf.reshape(tf.cast(sag_t1[1], tf.float32), [1]),
            "desc_sag_t1": tf.reshape(tf.cast(sag_t1[2], tf.float32), [1]),
            "img_sag_t2": tf.cast(sag_t2[0], tf.float32),
            "series_sag_t2": tf.reshape(tf.cast(sag_t2[1], tf.float32), [1]),
            "desc_sag_t2": tf.reshape(tf.cast(sag_t2[2], tf.float32), [1]),
            "img_axial_t2": tf.cast(axial[0], tf.float32),   
            "series_axial_t2": tf.reshape(tf.cast(axial[1], tf.float32), [1]),
            "desc_axial_t2": tf.reshape(tf.cast(axial[2], tf.float32), [1])
        }

        # --- 3. Build Targets Dictionary ---
        labels_dict = {}

        # Traceability: Pass the study_id back out to verify data integrity during inference
        # Expanded to (1,) or (batch, 1) to match the Lambda layer output shape
        labels_dict["study_id_output"] = tf.reshape(tf.cast(study_id_tf, tf.float32), [1])

        # Diagnosis: Reshape and map the 25 level records
        # records shape: (self._MAX_RECORDS, 4) -> [condition_id, severity, x, y]
        records = tf.reshape(labels["records"], (self._MAX_RECORDS, 4))

        for idx_on_row in range(self._MAX_RECORDS): 
            # Classification target (Severity: 0, 1, or 2)
            labels_dict[f"severity_row_{idx_on_row}"] = tf.reshape(tf.cast(records[idx_on_row, 1], tf.int32), [1])
    
            # Regression target (Coordinates: Normalized X, Y)
            # Remark: No need there to append [tf.newaxis], because records[idx_on_row, 2:4] is already
            # a Rank-1 tensor (vector of size 2) and not a Rank-0 tensor (scalar)..
            loc_data = tf.cast(records[idx_on_row, 2:4], tf.float32)
            loc_data.set_shape([2]) # Force the size of the coordinates vector
            labels_dict[f"location_row_{idx_on_row}"] = loc_data

        return features, labels_dict





