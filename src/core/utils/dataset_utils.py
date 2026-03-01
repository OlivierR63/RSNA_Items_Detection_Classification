# coding: utf-8

import tensorflow as tf
import json
import logging
import numpy as np

from typing import Dict, Tuple
from pathlib import Path
from src.config.config_loader import ConfigLoader


def calculate_max_series_depth(logger: logging.Logger=None) -> int:
    """
    Calculates the maximum number of slices per series using high-performance 
    TF dataset reduction and a localized metadata cache.

    Args:
        logger: Optional logger instance.
    
    Returns:
        int: The maximum number of records (depth) found across all series.
    """
    if logger:
        logger.info("Starting  function calculate_max_series_depth")

    global TFRECORD_DIR
    global DICOM_STUDIES_DIR
    global SERIES_DEPTH_THRESHOLD_PERCENTILE

    TFRecord_dir = Path(TFRECORD_DIR).resolve()
    dicom_studies_dir = Path(DICOM_STUDIES_DIR).resolve()

    if logger:
        logger.info(f"dicom_studies_dir = {dicom_studies_dir}")

    depth_cache_file = TFRecord_dir / "depth_metadata_cache.json"

    studies_dirs_list = [study for study in dicom_studies_dir.iterdir() if study.is_dir() ]
    studies_count = len(studies_dirs_list)

    if logger:
        logger.info( f"Function calculate_max_series_depth: found {studies_count} studies")

    if studies_count == 0:
        return 0

    series_depth = None

    # Smart Cache Management
    if depth_cache_file.exists():
        try:
            cache_mtime = depth_cache_file.stat().st_mtime
            with depth_cache_file.open('r') as f:
                depth_cache_data = json.load(f)

                # Check 1: Invalidate cache if the number of files has changed
                if depth_cache_data.get('studies_count') == studies_count:

                    # Check 2: Has any study directory been modified since cache creation?
                    # Note: Modifying / adding files inside a folder update sits time
                    is_cache_stale = any(
                        s.stat().st_mtime > cache_mtime for s in studies_dirs_list
                    )

                    if not is_cache_stale:
                        series_depth = depth_cache_data['series_depth']
                        if logger:
                            logger.info("Series depth loaded from cache")

        except Exception as e:
            if logger:
                warning_msg = f"Cache read failed, recalculating: {e}"
                logger.debug(warning_msg)
            pass
    
    # Depth calculation (Only if series_depth has not been recovered from the cache)
    if series_depth is None:
        series_depth = 0

        depth_list = []

        for study in studies_dirs_list:
            # Find the maximum depth across series in this study, defaults to 0
            depth_list.extend([len(list(series.glob('*.dcm'))) for series in study.iterdir() if series.is_dir()])
         
        series_depth = int(np.percentile(depth_list, SERIES_DEPTH_THRESHOLD_PERCENTILE))

        # Save to Cache
        try:
            TFRecord_dir.mkdir(parents=True, exist_ok=True)
            with depth_cache_file.open('w') as cache_file:
                json.dump({'studies_count': studies_count, 'series_depth': series_depth}, cache_file)

        except Exception as e:
            warning_msg = f"Unable to save the cache file in {TFRecord_dir} : {e}"
            if logger:
                logger.warning(warning_msg)

    if logger:
        logger.info(f"Function calculate_max_series_depth : calculated series depth = {series_depth}")

    return series_depth


# 2. Configuration Loading
config_loader = ConfigLoader("src/config/lumbar_spine_config.yaml")
config: dict = config_loader.get()

# 3.  Constant assignments (After functions are defined)
MAX_RECORDS = config.get('max_records', None)
if MAX_RECORDS is None:
    raise ValueError("Key 'max_records' is missing. Verify config file.")

MODEL_2D_HEIGHT, MODEL_2D_WIDTH, MODEL_2D_NB_CHANNELS = config['model_2d']['img_shape']
MIN_SCALING_VALUE = config['model_2d']['min_scaling_value']
MAX_SCALING_VALUE = config['model_2d']['max_scaling_value']
TFRECORD_DIR = config['tfrecord_dir']
DICOM_STUDIES_DIR = config['dicom_studies_dir']
SERIES_DEPTH_THRESHOLD_PERCENTILE = config['series_depth_threshold_percentile']

# 4. Final execution
MAX_SERIES_DEPTH = calculate_max_series_depth()

# 5. Define the dataset mapping helper functions
def parse_tfrecord_single_element(
    feature_tf: tf.Tensor,
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:

    """
    Parses and pre-processes a single DICOM frame from a TFRecord.

    The function handles the full pipeline from raw bytes to model-ready tensors, 
    including isometric resizing and coordinate system transformation.

    Operations performed:
    1. Feature Parsing: Extracts image bytes, metadata, and pathology records 
       using a predefined feature description.
    2. Image Decoding & Reshaping: Decodes raw uint16 pixels and reshapes them 
       into a (H, W, 1) tensor.
    3. Isometric Resizing: Resizes the image to target dimensions while 
       calculating the scaling ratio to maintain aspect ratio consistency.
    4. Coordinate Transformation: 
       - Adjusts pathology (x, y) coordinates based on image scaling.
       - Applies translation offsets to center the image within the model canvas.
       - Normalizes coordinates to a relative [0, 1] range based on the 
         target canvas (MODEL_2D_WIDTH/HEIGHT).
    5. Intensity Normalization: Normalizes pixels using series-level min/max values 
       while strictly preserving 0.0 for padding/background pixels.
    6. Memory Optimization: Casts the final image to float16 to reduce 
       RAM footprint and accelerate GPU throughput.

    Args:
        feature_tf: A serialized tf.train.Example proto (as a string Tensor).

    Returns:
        A Tuple containing:
        - normalized_image_tf (tf.float16): Processed (H, W, 1) image frame.
        - metadata_dict (dict): Tensors for identifiers (study_id, series_id, etc.) 
          and spatial context (scaling_ratio, x_crop, y_crop).
        - labels_dict (dict): Dictionary with a 'records' key containing 
          a (MAX_RECORDS, 4) float32 tensor of [condition, severity, x_norm, y_norm].
    """
    global MODEL_2D_HEIGHT, MODEL_2D_WIDTH, MAX_RECORDS

    # Define the structure of the features stored in the TFRecord.
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'is_padding': tf.io.FixedLenFeature([], tf.int64),
        'file_format': tf.io.FixedLenFeature([2], tf.int64),
        'study_id': tf.io.FixedLenFeature([], tf.int64),
        'series_id': tf.io.FixedLenFeature([], tf.int64),
        'series_min': tf.io.FixedLenFeature([], tf.int64),
        'series_max': tf.io.FixedLenFeature([], tf.int64),
        'instance_number': tf.io.FixedLenFeature([], tf.int64),
        'img_height': tf.io.FixedLenFeature([], tf.int64),
        'img_width': tf.io.FixedLenFeature([], tf.int64),
        'description': tf.io.FixedLenFeature([], tf.int64),
        'records': tf.io.FixedLenFeature([100], tf.float32), # nb_max_records * 4
        'nb_records': tf.io.FixedLenFeature([], tf.int64)
    }

    # --- 1. Parse the scalar protocol buffer string into a dictionary of Tensors. -----
    parsed_features_tf: Dict[str, tf.Tensor] = tf.io.parse_single_example(feature_tf, feature_description)

    # Assign the results back to named Tensors
    # As a reminder: condition = observed pathology.
    study_id_t = parsed_features_tf['study_id']
    is_padding_t = parsed_features_tf['is_padding']
    file_format_t = parsed_features_tf['file_format']
    series_id_t = parsed_features_tf['series_id']
    series_min_t = parsed_features_tf['series_min']
    series_max_t = parsed_features_tf['series_max']
    instance_number_t = parsed_features_tf['instance_number']
    img_height_t = parsed_features_tf['img_height']
    img_width_t = parsed_features_tf['img_width']
    description_t = parsed_features_tf['description']
    nb_records_t = parsed_features_tf['nb_records']
    padded_records_t = parsed_features_tf['records']

    # Secure the shapes (essential)
    study_id_t.set_shape([]) # Scalar
    is_padding_t.set_shape([])
    file_format_t.set_shape([2])
    series_id_t.set_shape([])
    series_min_t.set_shape([])
    series_max_t.set_shape([])
    instance_number_t.set_shape([])
    img_height_t.set_shape([])
    img_width_t.set_shape([])
    description_t.set_shape([])
    nb_records_t.set_shape([])
    padded_records_t.set_shape([MAX_RECORDS * 4])

    # --- 2. Deserialize and Reshape the Image Tensor (Pure TF) ---
    image_tf: tf.Tensor = tf.io.decode_raw(parsed_features_tf["image"], out_type=tf.uint16)
    # NOTE: Using tf.uint16, the original Dicom type normalization
    # to float32 happens later.
    # Crucial stage: the image shape is unknown so far.
    # It must be set before applying the "resize" command.
    # A new dimension is added for the channel (grayscale = 1)
    height_t = tf.cast(img_height_t, tf.int32)
    width_t = tf.cast(img_width_t, tf.int32)

    # Standardize to model input dimensions
    # Bilinear + Antialias is safe for BOTH 1x1 upsampling and DICOM downsampling.
    image_tf = tf.cond(
        tf.equal(is_padding_t, 1),

        # If padding image, create right now the black final image
        lambda: tf.zeros(
            [MODEL_2D_HEIGHT, MODEL_2D_WIDTH, 1],
            dtype=tf.float32
        ),

        # Otherly, follow the "normal" process
        lambda: tf.image.resize(
            tf.reshape(image_tf, [height_t, width_t, 1]), 
            [MODEL_2D_HEIGHT, MODEL_2D_WIDTH],
            method='bilinear',
            antialias=True
        )
    )

    # Resizing Ratio Calculation
    # We create tensors for the division
    current_dims = tf.cast(tf.stack([height_t, width_t]), tf.float32)
    target_dims = tf.constant([MODEL_2D_HEIGHT, MODEL_2D_WIDTH], dtype=tf.float32)
    
    ratios = tf.cond(
        tf.equal(is_padding_t, 1),
        lambda: tf.constant(1, tf.float32),
        lambda: current_dims / target_dims # Element-wise division [H_ratio, W_ratio]
    )
    
    min_ratio_t = tf.reduce_min(ratios)
    max_ratio_t = tf.reduce_max(ratios)
    
    scaling_ratio_t = tf.cond(
        min_ratio_t > 1.0, 
        lambda: min_ratio_t, 
        lambda: max_ratio_t
    )

    # Reshape the padded records tensor to (MAX_RECORDS, 4)
    padded_records_t = tf.reshape(padded_records_t, [MAX_RECORDS, 4])

    # Extract categorical metadata : (condition_level, severity)
    # We keep columns 0 and 1
    categorical_data = padded_records_t[:, :2]

    # Extract x and y coordinates
    coords_raw_t = tf.cast(padded_records_t[:, 2:], tf.float32)   

    # Update x and y coordinates after resizing image:
    # index 2 is x, index 3 is y.
    scaled_coords_t = coords_raw_t/scaling_ratio_t

    # Calculate offsets for centering
    # Note: using float32 for division and centering
    canvas_width_t = tf.cast(MODEL_2D_WIDTH, tf.float32)
    canvas_height_t = tf.cast(MODEL_2D_HEIGHT, tf.float32)

    # Define cropping position 
    x_crop_t = tf.constant(0, tf.float32)
    y_crop_t = tf.constant(0, tf.float32)

    actual_width_after_scale_t = tf.cast(width_t, tf.float32) / scaling_ratio_t
    actual_height_after_scale_t = tf.cast(height_t, tf.float32) / scaling_ratio_t

    offset_x_t = (canvas_width_t - actual_width_after_scale_t) / 2.0
    offset_y_t = (canvas_height_t - actual_height_after_scale_t) / 2.0

    # Stack offset [x, y] to match coordinates columns
    offset_t = tf.stack([offset_x_t, offset_y_t])

    # Apply translation
    updated_coords_t = scaled_coords_t + offset_t

    # Normalize the coordinates [0, 1] with regard to the canvas model
    model_dims = tf.cast(tf.stack([MODEL_2D_WIDTH, MODEL_2D_HEIGHT]), tf.float32)
    normalized_coords = updated_coords_t / model_dims

    # Rebuild final records (ensure same dtype for concat)
    final_records_t = tf.concat([
        tf.cast(categorical_data, tf.float32),
        tf.cast(normalized_coords, tf.float32)
    ], axis=1)

    # --- 3. Normalize the image pixel values -----
    # Cast to float16 to save RAM with no detrimental effect
    # on the intermediate calculation
    normalized_image_tf = tf.cast(
        normalize_image(image_tf, series_min_t, series_max_t),
        tf.float16
    )

    # --- 4. Prepare the output values ------
    metadata_dict = {
        "study_id": study_id_t,
        "series_id": series_id_t,
        "instance_number": instance_number_t,
        "is_padding": is_padding_t,
        "scaling_ratio": scaling_ratio_t,
        "x_crop": x_crop_t,
        "y_crop": y_crop_t,
        "description": description_t
    }

    labels_dict = {
        # The records are now a (self._MAX_RECORDS, 4) float32 tensor
        "records": final_records_t
    }

    return_object = (normalized_image_tf, metadata_dict, labels_dict)

    return return_object

def normalize_image(
    image_tf: tf.Tensor,
    series_min_t: tf.Tensor,
    series_max_t: tf.Tensor,
) -> tf.Tensor:

    """
    Normalizes image intensity while preserving 0 as the padding/neutral value.
    
    This ensures that background padding remains at absolute zero even after 
    Min-Max scaling and range shifting.
    """
    global MAX_SCALING_VALUE, MIN_SCALING_VALUE

    # 1. Identify padding pixels before transformation
    # We assume actual anatomical pixels have a value > 0 
    # (or we use the raw image_tf before any cast/shift)
    is_padding = tf.equal(image_tf, 0)

    # 2. Standard Normalization
    s_min = tf.cast(series_min_t, tf.float32)
    s_max = tf.cast(series_max_t, tf.float32)
    
    denom = tf.maximum(s_max - s_min, 1e-8)
    normalized = (tf.cast(image_tf, tf.float32) - s_min) / denom
    
    # Rescale to target range
    rescaled = normalized * (MAX_SCALING_VALUE - MIN_SCALING_VALUE) + MIN_SCALING_VALUE

    # 3. Final Step: Restore 0 for padding pixels
    # If is_padding is true, use 0.0, else use the rescaled value.
    final_image = tf.where(is_padding, tf.constant(0.0, dtype=tf.float32), rescaled)

    return final_image


def process_study_multi_series(
    images: tf.Tensor, 
    meta: Dict[str, tf.Tensor], 
    labels: Dict[str, tf.Tensor]
) -> Tuple[Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], 
                 Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], 
                 Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
                 ], 
           tf.Tensor, 
           Dict[str, tf.Tensor]
        ]:
    """
    Orchestrates the conversion of a raw study-level frame collection into a 
    structured multi-modal input for the neural network.

    This coordinator assumes structural padding (1x1 pixels) has been handled at 
    the TFRecord level. Its primary responsibilities are:
    1. View Decomposition: Splitting the global study tensor into three dedicated 
       anatomical streams (Sagittal T1, Sagittal T2, Axial T2).
    2. Series Resolution: Selecting the most frame-rich series for each view 
       when multiple acquisitions are present.
    3. Temporal/Spatial Ordering: Ensuring all volumes are anatomically sorted 
       before being fed to the model.
    4. Metadata/Label Compression: Reducing redundant frame-level records into 
       single study-level identifiers and diagnostics.

    Args:
        images (tf.Tensor): Flattened batch of all images in the study (N, H, W, C).
        meta (dict): Dictionary of metadata tensors (each of length N).
        labels (dict): Dictionary of label tensors (study-wide diagnostics).

    Returns:
        tuple: (study_data_triplet, study_id, reduced_labels)
               - study_data_triplet: A nested structure containing 4-tuples:
                 (Volume, Slice_Metadata, Series_ID, Description_Code) 
                 for each of the three required anatomical planes.
               - study_id: The unified identifier for the study (scalar).
               - reduced_labels: Study-level diagnostic labels (compacted).
    """

    global MAX_SERIES_DEPTH, MODEL_2D_HEIGHT, MODEL_2D_WIDTH, MODEL_2D_NB_CHANNELS

    # --- 1. Define Search Targets (Integer Codes) ---
    # Based on the internal mapping: 0: "Sagittal T1", 1: "Sagittal T2", 2: "Axial T2"
    # Using constants ensures Graph-compatibility during execution.
    t1_code = tf.constant(0, dtype=tf.int32)
    t2_code = tf.constant(1, dtype=tf.int32)
    ax_code = tf.constant(2, dtype=tf.int32)

    # --- 2. Process Individual Branches ---
    # We extract the 3-tuple (Padded_Images, Selected_Series_ID, Description_Code) for each plane.
    # This handles series selection and black-frame padding internally.
    res_t1 = process_single_description(
        images, meta, t1_code,
        MAX_SERIES_DEPTH,
        MODEL_2D_HEIGHT,
        MODEL_2D_WIDTH,
        MODEL_2D_NB_CHANNELS
    )
    res_t2 = process_single_description(
        images, meta, t2_code,
        MAX_SERIES_DEPTH,
        MODEL_2D_HEIGHT,
        MODEL_2D_WIDTH,
        MODEL_2D_NB_CHANNELS
    )
    res_ax = process_single_description(
        images, meta, ax_code,
        MAX_SERIES_DEPTH,
        MODEL_2D_HEIGHT,
        MODEL_2D_WIDTH,
        MODEL_2D_NB_CHANNELS
    )

    # --- 3. Extract Global Study Context ---
    # Since all frames in the window belong to the same study (guaranteed by _get_group_key),
    # we take the first element as the representative study_id.
    study_id = meta['study_id'][0]

    # --- 4. Label Consolidation (TensorFlow-native approach) ---
    # We use tf.nest.map_structure to apply the reduction to every element 
    # in the labels dictionary without explicit Python loops.
    reduced_labels = tf.nest.map_structure(reduce_to_first_element, labels)

    # Return the structured data required by LumbarDicomTFRecordDataset._format_for_model :
    return (res_t1, res_t2, res_ax), study_id, reduced_labels

def reduce_to_first_element(v):
    """
    Reduces a tensor to its first element along the leading dimension if it is not a scalar.

    This helper is designed for study-level processing where labels or metadata 
    are redundantly replicated across all frames of a series. It ensures that 
    the downstream model receives a single representative value (rank N-1) 
    instead of a batch of identical values (rank N).

    Args:
        v (tf.Tensor): The input tensor to be reduced.

    Returns:
        tf.Tensor: The first element v[0] if rank > 0, otherwise the original scalar v.
    """

    # We check the rank dynamically using tf.rank (TensorFlow op) 
    # instead of v.shape.rank (Python attribute) to ensure compatibility 
    # with the tensorflow graph execution and Autograph

    return tf.cond(
        tf.rank(v) > 0,
        lambda: v[0], # Take first element if it's a batch/sequence
        lambda: v    # Keep as is if it's already a scalar
    )

def process_single_description(
    all_images: tf.Tensor, 
    all_meta: Dict[str, tf.Tensor], 
    target_desc_tensor: tf.Tensor,
    series_depth: int,
    height: int,
    width: int,
    nb_channels: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    """
    Routes the study images to the appropriate processing logic based on anatomical view.

    This function acts as a high-level router that:
    1. Filters the study-level collection for a specific description (e.g., Axial T2).
    2. Checks if any frames exist for this view.
    3. Dispatches to 'process_valid_series' for sorting and shape enforcement if 
       data is present, or to 'process_empty_series' to generate a zero-filled 
       placeholder if the view is missing.

    Args:
        all_images (tf.Tensor): Flattened batch of images for the study (N, H, W, C).
        all_meta (Dict[str, tf.Tensor]): Metadata containing 'description', 
                                         'series_id', and 'instance_number'.
        target_desc_tensor (tf.Tensor): The anatomical view code to filter for.
        series_depth (int): Expected depth of the output volume.
        height (int): Target image height.
        width (int): Target image width.
        nb_channels (int): Target number of channels.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: 
            - Volume: A (series_depth, H, W, C) tensor (actual data or zeros).
            - Metadata: A (series_depth, 5) tensor, including for each slice:
                        - The instance number (-1 if padding image)
                        - A padding flag (0 : "true" image, 1: padding image)
                        - The image scaling ratio from its original size to the model one
                        - The x_crop offset (normalized)
                        - The y_crop offset (normalized)
            - Best Series ID: Selected series identifier or -1 if empty.
            - Description Code: The target view code (int32).
    """

    # Convert to int64 for stable comparison.
    current_desc = tf.cast(all_meta['description'], tf.int64)
    target_code = tf.cast(target_desc_tensor, tf.int64)

    # Create the filtering mask
    desc_mask = tf.equal(current_desc, target_code)
    desc_mask = tf.cast(desc_mask, tf.bool)

    # Set shapes for TensorFlow Graph compatibility
    desc_mask.set_shape([None])
    all_images.set_shape([None, None, None, None])

    # Check if any image matches the description
    mask_has_data = tf.reduce_any(desc_mask)

    # Execute conditional logic
    return tf.cond(
        mask_has_data,
        true_fn=lambda: process_valid_series(
            all_images, all_meta, desc_mask, target_desc_tensor,
            series_depth, height, width, nb_channels
        ),
        false_fn=lambda: process_empty_series(
            all_images, target_desc_tensor,
            series_depth, height, width
        )
    )

def process_valid_series(
    all_images: tf.Tensor,
    all_meta: Dict[str, tf.Tensor],
    desc_mask: tf.Tensor,
    target_desc_tensor: tf.Tensor,
    series_depth: int,
    height: int,
    width: int,
    nb_channels: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    """
    Processes and normalizes a series that contains data for the target description.

    This function assumes that padding (to handle missing instances and reach 
    series_depth) has been performed during TFRecord creation. 
    It focuses on structural integrity:
    1. Filters images and metadata to isolate the target anatomical view.
    2. Selects the most robust series if multiple candidates exist.
    3. Re-orders frames by instance number to ensure correct anatomical sequence.
    4. Enforces a static 3D shape and converts grayscale to RGB if required.

    Args:
        all_images: Full batch of images for a study (N, H, W, C).
        all_meta: Metadata dictionary containing 'series_id' and 'instance_number'.
        desc_mask: Boolean mask identifying images matching the target description.
        target_desc_tensor: The description code being processed.
        series_depth: Fixed number of slices required for the 3D volume.
        height: Target image height.
        width: Target image width.
        nb_channels: Target number of channels (e.g., 3 for RGB).

    Returns:
       A tuple containing:
            - padded_vol (tf.Tensor): The normalized 3D volume (series_depth, H, W, 3).
            - slice_metadata (tf.Tensor): Per-slice flags [instance_number, is_padding, scaling_ratio, x_crop, y_crop] (series_depth, 5).
            - best_id (tf.Tensor): The ID of the selected series (int64).
            - target_desc (tf.Tensor): The description code (int32).
    """

    # --- 1. Filtering by Description ---
    d_images = tf.boolean_mask(all_images, desc_mask)
    d_series_ids = tf.boolean_mask(all_meta['series_id'], desc_mask)
    d_instances = tf.boolean_mask(all_meta['instance_number'], desc_mask)
    d_padding_flags = tf.boolean_mask(all_meta['is_padding'], desc_mask)
    d_scaling_ratios = tf.boolean_mask(all_meta['scaling_ratio'], desc_mask)
    d_x_crop = tf.boolean_mask(all_meta['x_crop'], desc_mask)
    d_y_crop = tf.boolean_mask(all_meta['y_crop'], desc_mask)

    # --- 2. Best Series Selection (Highest Frame Count) ---
    unique_ids, _, counts = tf.unique_with_counts(d_series_ids)
    best_id = unique_ids[tf.argmax(counts)]

    # --- 3. Isolate the chosen series
    series_mask = tf.equal(d_series_ids, best_id)
    series_mask = tf.cast(series_mask, tf.bool)
    #series_mask.set_shape([None])
    #d_images.set_shape([None, None, None, None])

    final_imgs = tf.boolean_mask(d_images, series_mask)
    final_instances = tf.boolean_mask(d_instances, series_mask)
    final_padding_flags = tf.boolean_mask(d_padding_flags, series_mask)
    final_scaling_ratios = tf.boolean_mask(d_scaling_ratios, series_mask)
    final_x_crop = tf.boolean_mask(d_x_crop, series_mask)
    final_y_crop = tf.boolean_mask(d_y_crop, series_mask)

    # --- 4. Spatial Sorting ---
    sort_idx = tf.argsort(final_instances)
    sorted_instances = tf.gather(final_instances, sort_idx)
    sorted_imgs = tf.gather(final_imgs, sort_idx)
    sorted_padding_flags = tf.gather(final_padding_flags, sort_idx)
    sorted_scaling_ratios = tf.gather(final_scaling_ratios, sort_idx)
    sorted_x_crop = tf.gather(final_x_crop, sort_idx)
    sorted_y_crop = tf.gather(final_y_crop, sort_idx)

    # 5. Final formatting
    # We use a simple slice to be 100% sure of the depth for the Graph
    final_vol = sorted_imgs[:series_depth, ...]

    # 6. Final shape enforcement for the model*
    # Convert single-channel MRI to 3-channel (pseudo-RGB) to match model input requirements.
    nb_channels = tf.shape(final_vol)[-1]
    final_vol = tf.cond(
        tf.equal(nb_channels, 1),
        lambda: tf.image.grayscale_to_rgb(final_vol),
        lambda: final_vol
    )

    # Stack the whole volume into a single tensor.
    vol_shape = tf.stack([
        tf.cast(series_depth, tf.int32),
        tf.cast(height, tf.int32),
        tf.cast(width, tf.int32),
        tf.constant(3, tf.int32)    # Forcing 3 as we use grayscale_to_rgb
    ])

    final_vol = tf.reshape(final_vol, vol_shape)
    final_vol = tf.cast(final_vol, tf.float32)

    # The whole metadata is stacked into a single tensor [series_depth, 5]
    # This creates a vector of 5 features for each slice in the volume
    # Hard-set shape to finalize the contract with the model
    slice_metadata = tf.stack(
        [
            tf.cast(sorted_instances[:series_depth], tf.float32),
            tf.cast(sorted_padding_flags[:series_depth], tf.float32),
            tf.cast(sorted_scaling_ratios[:series_depth], tf.float32),
            tf.cast(sorted_x_crop[:series_depth], tf.float32),
            tf.cast(sorted_y_crop[:series_depth], tf.float32)
        ], 
        axis=1
    )

    meta_shape = tf.stack([
        tf.cast(series_depth, tf.int32), 
        tf.constant(5, dtype=tf.int32)
    ])
    slice_metadata = tf.reshape(slice_metadata, meta_shape)

    return final_vol, slice_metadata, tf.cast(best_id, tf.int64), target_desc_tensor

def process_empty_series(
    all_images: tf.Tensor,
    target_desc_tensor: tf.Tensor,
    series_depth: int,
    height: int,
    width: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    """
    Generates a placeholder volume when no data matches the target description.

    This ensures that the TensorFlow Graph receives a consistent tensor shape 
    and data type even if the specific anatomical view is missing for a study.

    Args:
        all_images: Used only to infer the correct dtype for the zero tensor.
        target_desc_tensor: The description code that was not found.
        series_depth: Fixed number of slices for the output volume.
        height: Target image height.
        width: Target image width.

    Returns:
        A tuple containing:
            - empty_vol (tf.Tensor): A zero-filled tensor (series_depth, H, W, 3).
            - slice_metadata (tf.Tensor): Per-slice flags 
              [instance_number=1.0, is_padding=1.0, scaling_ratio=1.0, x_crop=0.0, y_crop=0.0] (series_depth, 5).
            - default_id (tf.Tensor): Sentinel value -1 (int64).
            - target_desc (tf.Tensor): The description code (int32).
    """

    # 1. Constants and Type Casting
    default_id = tf.constant(-1, dtype=tf.int64)
    s_depth = tf.cast(series_depth, tf.int32)
    h = tf.cast(height, tf.int32)
    w = tf.cast(width, tf.int32)
    channels = tf.constant(3, tf.int32)
    

    # 2. Build Image Volume Shape
    # Using tf.stack to create a dynamic shape tensor
    vol_shape = tf.stack([s_depth, h, w, channels])
    
    # Create and force shape via reshape (Graph safe)
    empty_vol = tf.zeros(vol_shape, dtype=tf.float32)
    empty_vol = tf.reshape(empty_vol, vol_shape)

    # 3. Build Slice Metadata
    # We use tf.reshape instead of tf.ensure_shape for the same reasons
    meta_shape = tf.stack([s_depth, tf.constant(5, tf.int32)])
    
    slice_metadata = tf.stack(
        [
            tf.fill([s_depth], -1.0),  # Dummy instance number
            tf.fill([s_depth], 1.0),   # Padding flag
            tf.fill([s_depth], 1.0),   # Scaling ratio
            tf.fill([s_depth], 0.0),   # x_crop
            tf.fill([s_depth], 0.0)    # y_crop
        ], 
        axis=1
    )
    
    # Finalize shape for the metadata tensor
    slice_metadata = tf.reshape(slice_metadata, meta_shape)

    return empty_vol, slice_metadata, default_id, target_desc_tensor

def format_for_model(
    study_volumes: Tuple[
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], 
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
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

    global MAX_SERIES_DEPTH, MODEL_2D_HEIGHT, MODEL_2D_WIDTH, MODEL_2D_NB_CHANNELS, MAX_RECORDS

    # Retrieve the config of the expected shapes
    target_shape = [MAX_SERIES_DEPTH, MODEL_2D_HEIGHT, MODEL_2D_WIDTH, MODEL_2D_NB_CHANNELS]
    meta_target_shape = [MAX_SERIES_DEPTH, 5]

    # Unpack processed volumes from the previous study-level processing step
    sag_t1, sag_t2, axial = study_volumes

    # --- 2. Build Inputs Dictionary ---
    # These keys MUST exactly match the names defined in ModelFactory.build_multi_series_model()
    # Use tf.ensure_shape to guarantee dimensions without breaking the graph flow
    features = {
        "study_id": tf.reshape(tf.cast(study_id_tf, tf.float32), [1]),
        "img_sag_t1": tf.ensure_shape(tf.cast(sag_t1[0], tf.float32), target_shape),
        "slice_metadata_t1": tf.ensure_shape(tf.cast(sag_t1[1], tf.float32), meta_target_shape),
        "series_sag_t1": tf.reshape(tf.cast(sag_t1[2], tf.float32), [1]),
        "desc_sag_t1": tf.reshape(tf.cast(sag_t1[3], tf.float32), [1]),
        "img_sag_t2": tf.ensure_shape(tf.cast(sag_t2[0], tf.float32), target_shape),
        "slice_metadata_t2": tf.ensure_shape(tf.cast(sag_t2[1], tf.float32), meta_target_shape),
        "series_sag_t2": tf.reshape(tf.cast(sag_t2[2], tf.float32), [1]),
        "desc_sag_t2": tf.reshape(tf.cast(sag_t2[3], tf.float32), [1]),
        "img_axial_t2": tf.ensure_shape(tf.cast(axial[0], tf.float32), target_shape),
        "slice_metadata_axial_t2": tf.ensure_shape(tf.cast(axial[1], tf.float32), meta_target_shape),
        "series_axial_t2": tf.reshape(tf.cast(axial[2], tf.float32), [1]),
        "desc_axial_t2": tf.reshape(tf.cast(axial[3], tf.float32), [1])
    }

    # --- 3. Build Targets Dictionary ---
    labels_dict = {}

    # Traceability: Pass the study_id back out to verify data integrity during inference
    # Expanded to (1,) or (batch, 1) to match the Lambda layer output shape
    #labels_dict["study_id_output"] = tf.reshape(tf.cast(study_id_tf, tf.float32), [1])

    # Diagnosis: Reshape and map the 25 level records
    # records shape: (MAX_RECORDS, 4) -> [condition_id, severity, x, y]
    records_raw = tf.reshape(labels["records"], (MAX_RECORDS, 4))

    # Sorting: use condition_id (col 0) as intreger for stable sorting
    condition_ids = tf.cast(records_raw[:,0], tf.int32)
    sort_indices = tf.argsort(condition_ids, direction="ASCENDING")

    # 2. Reorder the entire records tensor using these indices
    sorted_records = tf.gather(records_raw, sort_indices)

    # Classification target (Severity: 0, 1, or 2): explicit cast to int32
    # before one-hot encoding.
    severity_labels = tf.cast(sorted_records[:, 1], tf.int32)
    labels_dict["severity_output"] = tf.cast(tf.one_hot(severity_labels, depth=3), tf.float32)

    # Regression target (Coordinates: Normalized X, Y)
    # Remark: No need there to append [tf.newaxis], because records[:, 2:4] is already
    # a Rank-1 tensor (vector of size 2) and not a Rank-0 tensor (scalar)..
    labels_dict["location_output"] = tf.cast(sorted_records[:, 2:4], tf.float32)

    # IMPORTANT: Force float32 across the entire features dictionary.
    # This loop serves as a final safety check to prevent dtype mismatch.
    features = {k: tf.cast(v, tf.float32) for k, v in features.items()}

    # Do the same for labels to be absolutely safe
    labels_dict = {k: tf.cast(v, tf.float32) for k, v in labels_dict.items()}

    return features, labels_dict
