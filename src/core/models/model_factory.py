# coding: utf-8

import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, Model
from src.core.models.conv3d_aggregator import Conv3DAggregator
from src.core.models.model_2d import Backbone2D


def minimal_parse(proto):
    """
    Parses only the metadata field from a TFRecord proto to minimize I/O overhead.
    """
    feature_description = {
        'metadata': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(proto, feature_description)

class ModelFactory:
    """
    Factory class to build and configure the multi-modal RSNA Lumbar Spine model.
    Handles dynamic depth calculation and shared backbone orchestration.
    """
    def __init__(self, *, config, logger, nb_output_records, series_depth = 0, num_meta=3, aggregator_class=Conv3DAggregator):
        self._num_meta = num_meta
        self._config = config
        self._logger = logger
        self._aggregator_class = aggregator_class
        self._nb_output_records = nb_output_records
        self._tfrecord_dir = Path(self._config['tfrecord_dir'])
        
        # Calculate or retrieve max_series_depth (max slices per series)
        if series_depth == 0:
            class_name = self.__class__.__name__
            error_msg = f"{class_name} initialization failed: the argument 'series_depth' was not properly filled."
            raise ValueError(error_msg)

        self._series_depth = series_depth

    def _process_branch(
        self,
        img_input: tf.Tensor,
        series_input: tf.Tensor,
        desc_input: tf.Tensor,
        backbone: tf.keras.layers.Layer,
        suffix: str
    ) -> tf.Tensor:
        """
        Processes a single imaging series branch by extracting visual features 
        and concatenating them with series-specific metadata.

        Args:
            img_input: Input tensor for the image sequence (Batch, Time, H, W, C).
            series_input: Input tensor for the metadata associated with the series.
            desc_input: Input tensor for the metadata associated with the "description" (ie. the observed pathology).
            backbone: The TimeDistributed backbone shared across branches.
            suffix: Identifier for naming layers (e.g., 'sag_t1').

        Returns:
            tf.Tensor: Merged features of visual and metadata information for the branch.
        """
        # Extract temporal/spatial features using the shared 2D backbone
        x = backbone(img_input)
    
        # Aggregate sequence features into a single vector (3D context)
        # This uses the aggregator class defined in your config (e.g., Conv3D, LSTM, or GlobalAverage)
        aggregator = self._aggregator_class(
            config=self._config,
            series_depth=self._series_depth,
            logger=self._logger
        )
        x = aggregator.build(x, suffix=suffix)
    
        # Merge visual features with series-specific metadata
        combined_branch = layers.concatenate([x, series_input, desc_input], name=f"merged_feat_{suffix}")
    
        return combined_branch

    def build_multi_series_model(self):
        """
        Constructs a multi-input model capable of processing Sagittal T1, 
        Sagittal T2, and Axial T2 series simultaneously with their respective metadata.

        Returns:
            tf.keras.Model: The compiled multi-output Keras model.
        """

        # --- 1. Define Inputs for study Id ---
        study_id_layer = layers.Input(shape=(1,), name="study_id" )

        # --- 2. Define Inputs for each series ---
        # Images shapes are (self._max_series_depth, height, width, channels)

        self._logger.info("Start building model")
        height, width, channels = self._config['model_2d']['img_shape']

        # Sagittal T1
        img_sag_t1 = layers.Input(shape=(self._series_depth, height, width, channels), name="img_sag_t1")
        series_sag_t1 = layers.Input(shape=(1,), name="series_sag_t1")
        desc_sag_t1 = layers.Input(shape=(1,), name="desc_sag_t1")
    
        # Sagittal T2
        img_sag_t2 = layers.Input(shape=(self._series_depth, height, width, channels), name="img_sag_t2")
        series_sag_t2 = layers.Input(shape=(1,), name="series_sag_t2")
        desc_sag_t2 = layers.Input(shape=(1,), name="desc_sag_t2")
    
        # Axial T2
        img_axial_t2 = layers.Input(shape=(self._series_depth, height, width, channels), name="img_axial_t2")
        series_axial_t2 = layers.Input(shape=(1,), name="series_axial_t2")
        desc_axial_t2 = layers.Input(shape=(1,), name="desc_axial_t2")

        # --- 3. Shared Backbone Initialization ---
        # We use a single instance of TimeDistributed to share weights across all three branches
        backbone_obj = Backbone2D(
            model_name=self._config['model_2d']['type'], 
            input_shape=self._config['model_2d']['img_shape']
        )
        shared_td_backbone = layers.TimeDistributed(backbone_obj.get_model(), name="shared_2d_backbone")

        # --- 4. Process each branch independently ---
        feat_sag_t1 = self._process_branch(img_sag_t1, series_sag_t1, desc_sag_t1, shared_td_backbone, "sag_t1")
        feat_sag_t2 = self._process_branch(img_sag_t2, series_sag_t2, desc_sag_t2, shared_td_backbone, "sag_t2")
        feat_ax_t2  = self._process_branch(img_axial_t2, series_axial_t2, desc_axial_t2, shared_td_backbone, "axial_t2")

        # --- 5. Global Fusion ---
        # Concatenate features from all imaging planes
        global_features = layers.concatenate([feat_sag_t1, feat_sag_t2, feat_ax_t2], name="global_fusion")

        lay_z = layers.Dense(512, activation='relu')(global_features)
        lay_z = layers.Dropout(0.3)(lay_z)

        # --- 6. Outputs and Traceability ---
        # Traceability output: Passthrough of study_id (using T1 branch as reference)
        study_id_output = layers.Lambda(lambda x: x, name="study_id_output")(study_id_layer)

        pathology_level_location_output_list = {'study_id_output': study_id_output}

        for idx_on_row in range(self._nb_output_records):
            out_severity = layers.Dense(3, activation='softmax', name=f"severity_row_{idx_on_row}")(lay_z)
            out_location = layers.Dense(2, activation='sigmoid', name=f"location_row_{idx_on_row}")(lay_z)
            pathology_level_location_output_list[f"severity_row_{idx_on_row}"] = out_severity
            pathology_level_location_output_list[f"location_row_{idx_on_row}"] = out_location

        # Build the final functional model
        inputs_dict = {
            "study_id": study_id_layer,
            "img_sag_t1": img_sag_t1,
            "series_sag_t1": series_sag_t1,
            "desc_sag_t1": desc_sag_t1,
            "img_sag_t2": img_sag_t2,
            "series_sag_t2": series_sag_t2,
            "desc_sag_t2": desc_sag_t2,
            "img_axial_t2": img_axial_t2,
            "series_axial_t2": series_axial_t2,
            "desc_axial_t2": desc_axial_t2
        }

        model = Model(
            inputs=inputs_dict,
            outputs=pathology_level_location_output_list,
            name="RSNA_MultiSeries_Model"
        )
        self._logger.info("Building model completed")

        return model