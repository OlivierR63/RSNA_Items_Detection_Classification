# coding: utf-8

# Standard library
from typing import Dict, Tuple, Any
import os
import sys

# Third party library
import tensorflow as tf


# Define the dataset mapping helper functions
def parse_tfrecord_single_element(
    feature_tf: tf.Tensor,
    current_epoch_tensor: tf.Tensor,  # Injected from the dataset map
    config: Dict[str, Any]
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:

    """
    Parses and pre-processes a single DICOM frame from a TFRecord.

    The function handles the full pipeline from raw bytes to model-ready tensors,
    including isometric resizing and coordinate system transformation.

    Operations performed:
    1. Feature Parsing: Extracts image bytes, metadata, and pathology records
       using a predefined feature description.
    2. Image Decoding & Reshaping: Decodes raw int16 pixels and reshapes them
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
        - feature_tf: A serialized tf.train.Example proto (as a string Tensor).
        - current_epoch_tensor: index on the current epoch
        - config: application setting

    Returns:
        A Tuple containing:
        - normalized_image_tf (tf.float16): Processed (H, W, 1) image frame.
        - metadata_dict (dict): Tensors for identifiers (study_id, series_id, etc.)
          and spatial context (scaling_ratio, x_crop, y_crop).
        - labels_dict (dict): Dictionary with a 'records' key containing
          a (MAX_RECORDS, 4) float32 tensor of [condition_level, severity, x_norm, y_norm].
    """

    logging_cfg = config['logging']
    level_cfg = logging_cfg['level']

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Starting function parse_tfrecord_single_element"),
        false_fn=lambda: tf.no_op()
    )

    models_cfg = config['models']
    backbone_2d_cfg = models_cfg['backbone_2d']
    img_shape_cfg = backbone_2d_cfg['img_shape']
    model_2d_height, model_2d_width, _ = img_shape_cfg

    data_specs_cfg = config['data_specs']
    max_records = data_specs_cfg['max_records_per_frame']

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
        'series_description': tf.io.FixedLenFeature([], tf.int64),
        'records': tf.io.FixedLenFeature([100], tf.float32),  # nb_max_records * 4
        'nb_records': tf.io.FixedLenFeature([], tf.int64)
    }

    # --- 1. Parse the scalar protocol buffer string into a dictionary of Tensors. -----
    parsed_features_tf: Dict[str, tf.Tensor] = tf.io.parse_single_example(
        feature_tf, feature_description
    )

    # Assign the results back to named Tensors
    study_id_t = parsed_features_tf['study_id']
    is_padding_t = parsed_features_tf['is_padding']
    file_format_t = parsed_features_tf['file_format']
    series_id_t = parsed_features_tf['series_id']
    series_min_t = parsed_features_tf['series_min']
    series_max_t = parsed_features_tf['series_max']
    instance_number_t = parsed_features_tf['instance_number']
    img_height_t = parsed_features_tf['img_height']
    img_width_t = parsed_features_tf['img_width']
    description_t = parsed_features_tf['series_description']
    nb_records_t = parsed_features_tf['nb_records']
    padded_records_t = parsed_features_tf['records']

    # Secure the shapes (essential)
    study_id_t.set_shape([])  # Scalar
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
    padded_records_t.set_shape([max_records * 4])

    # NOTE: Using tf.int16, the original Dicom type normalization
    # to float32 happens later.
    # Crucial stage: the image shape is unknown so far.
    # It must be set before applying the "resize" command.
    # A new dimension is added for the channel (grayscale = 1)
    height_t = tf.cast(img_height_t, tf.int32)
    width_t = tf.cast(img_width_t, tf.int32)

    # --- 2. Deserialize and Reshape the Image Tensor (Pure TF) ---
    image_raw_tf: tf.Tensor = tf.io.decode_raw(parsed_features_tf["image"], out_type=tf.int16)

    # IMPORTANT: decode_raw returns a 1D vector. We MUST reshape it to (H, W, 1)
    # before any image operation like pad or crop.
    # Before, we must ensure the data integrity.

    expected_size = height_t * width_t
    actual_size = tf.shape(image_raw_tf)[0]

    debug_msg = (
        f"Processing -> Study: {study_id_t}, "
        f"Series: {series_id_t}, "
        f"Instance: {instance_number_t}, "
        f"Format: ({img_height_t}, {img_width_t}), "
        f"Expected size: ({expected_size}), "
        f"Actual size: ({actual_size})"
    )

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print(debug_msg),
        false_fn=lambda: tf.no_op()
    )

    check_op = raise_size_error(expected_size, actual_size)
    with tf.control_dependencies([check_op]):
        image_tf = tf.reshape(image_raw_tf, [height_t, width_t, 1])

    # Static shape hint for the compiler
    image_tf.set_shape([None, None, 1])

    # Decide mode based on epoch number (Even = Crop, Odd = Resize)
    # This ensures the whole dataset switches mode at the same time
    use_crop_epoch = tf.equal(tf.math.mod(current_epoch_tensor, 2), 0)

    can_crop = tf.logical_or(height_t > model_2d_height, width_t > model_2d_width)

    # Create a stable seed for the random crop based on series_id
    # This ensures all images in the SAME series get the SAME crop offsets
    # We use stateless_random because it's deterministic given a seed
    seed = tf.stack([
        tf.cast(series_id_t, tf.int32),
        tf.cast(current_epoch_tensor, tf.int32)
    ])

    # Standardize to model input dimensions
    # Bilinear + Antialias is safe for BOTH 1x1 upsampling and DICOM downsampling.
    image_tf, scaling_ratio_t, offset_t = tf.cond(
        tf.equal(is_padding_t, 1),

        # If padding image, create right now the black final image
        true_fn=lambda: create_padding_image(config),

        # Otherly, follow the "normal" process
        false_fn=lambda: tf.cond(
            tf.logical_and(can_crop, use_crop_epoch),
            # seed is passed to ensure the crop is the same for the whole series
            true_fn=lambda: perform_deterministic_crop(image_tf, height_t, width_t, seed, config),
            false_fn=lambda: perform_resize(image_tf, height_t, width_t, config)
        )
    )

    # Reshape the padded records tensor to (MAX_RECORDS, 4)
    padded_records_t = tf.reshape(padded_records_t, [max_records, 4])

    # Extract categorical metadata : (condition_level, severity)
    # We keep columns 0 and 1
    categorical_data = padded_records_t[:, :2]

    # Extract x and y coordinates
    coords_raw_t = tf.cast(padded_records_t[:, 2:], tf.float32)

    # Update x and y coordinates after resizing image:
    # index 2 is x, index 3 is y.
    scaled_coords_t = coords_raw_t/scaling_ratio_t

    # Define cropping position
    x_crop_t = tf.maximum(tf.constant(0, tf.float32), -offset_t[0])
    y_crop_t = tf.maximum(tf.constant(0, tf.float32), -offset_t[1])

    # We create tensors for the division
    target_dims = tf.constant([model_2d_width, model_2d_height], dtype=tf.float32)

    # Map original coordinates into the processed image's coordinate system.
    # We subtract the crop origin (x_crop, y_crop) to 'shift' the label
    # so it matches the visible window sent to the backbone.
    # (In Resize mode, scaling and centering offsets are used instead).
    updated_coords_t = scaled_coords_t + offset_t

    # Normalize the coordinates [0, 1] with regard to the canvas model
    normalized_coords = updated_coords_t / target_dims

    # Rebuild final records (ensure same dtype for concat)
    final_records_t = tf.concat([
        tf.cast(categorical_data, tf.float32),
        tf.cast(normalized_coords, tf.float32)
    ], axis=1)

    # --- 3. Normalize the image pixel values -----
    # Cast to float16 to save RAM with no detrimental effect
    # on the intermediate calculation
    normalized_image_tf = tf.cast(
        normalize_image(
            image_tf=image_tf,
            series_min_t=series_min_t,
            series_max_t=series_max_t,
            config=config
        ),
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
        "series_description": description_t
    }

    labels_dict = {
        # The records are now a (self._MAX_RECORDS, 4) float32 tensor
        "records": final_records_t
    }

    return_object = (normalized_image_tf, metadata_dict, labels_dict)

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Function parse_tfrecord_single_element completed"),
        false_fn=lambda: tf.no_op(),
    )

    return return_object


def raise_size_error(expected_size, actual_size):
    """
    Creates a TensorFlow assertion to validate image data integrity.

    Args:
        expected_size: Calculated size based on metadata (height * width).
        actual_size: Actual number of pixels found in the raw binary string.
    """
    condition = tf.equal(expected_size, actual_size)

    return tf.cond(
        condition,
        lambda: tf.no_op(),
        lambda: stop_process(expected_size, actual_size)
    )


def stop_process(expected_size, actual_size):
    tf.py_function(python_stop, [expected_size, actual_size], [])
    return tf.no_op()


def python_stop(exp, act):
    # This message is displayed right before the process is killed
    tf.print("\n[FATAL] DATA CORRUPTION DETECTED")
    tf.print("Expected image size (height * width):", exp)
    tf.print("Actual image size found:  ", act)

    # Flush all the buffers and kill the process immediately
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
    return tf.no_op()


def create_padding_image(
    config: Dict[str, Any]
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Generates a blank (black) image and neutral transformation parameters.

    This is used when 'is_padding' is 1, ensuring the tensor structure
    remains consistent across the dataset pipeline.

    Args:
        - config: Application settings

    Returns:
        A Tuple containing:
        - padding (tf.float32): A (H, W, 1) "zeroed" (ie set to the minimal backbone 2d image value)
                                image tensor.
        - ratio (tf.float32): A neutral scaling ratio of 1.0.
        - offset (tf.float32): A zero translation vector [0.0, 0.0].
    """

    level_cfg = config['logging']['level']

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Starting function create_padding_image"),
        false_fn=lambda: tf.no_op()
    )

    min_val = config['models']['backbone_2d']['scaling']['min']
    model_2d_height, model_2d_width, _ = config["models"]["backbone_2d"]["img_shape"]
    padding = tf.ones(
        [model_2d_height, model_2d_width, 1],
        dtype=tf.float32
    ) * min_val

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Function create_padding_image completed"),
        false_fn=lambda: tf.no_op()
    )

    # Return neutral values to avoid affecting coordinate calculations
    return padding, tf.constant(1.0, dtype=tf.float32), tf.constant([0.0, 0.0], dtype=tf.float32)


def perform_deterministic_crop(
    raw_image_tf: tf.Tensor,
    height_t: tf.Tensor,
    width_t: tf.Tensor,
    seed: tf.Tensor,     # Expect a tensor of shape [2] (e.g.: [series_id, epoch])
    config: Dict[str, Any]
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Crops the image using a deterministic seed to ensure all slices in a series
    are cropped identically.

    This function handles images larger than the target (cropping) and images
    smaller than the target (padding) by first ensuring the image is at least
    MODEL_2D size using center padding. It then picks a deterministic top-left
    corner based on the provided seed and extracts the crop.

    Coordinate Transformation Logic:
    The scaling ratio is 1.0 (no resizing). The offset returned is a vector of
    negative integers [-x_start, -y_start]. When added to the original label
    coordinates, it effectively translates the points to the new local
    coordinate system of the cropped window.

    Args:
        - raw_image_tf: Original image tensor (H, W, 1) in int16 or float32.
        - height_t: Actual height of the raw_image_tf.
        - width_t: Actual width of the raw_image_tf.
        - seed: A shape [2] int32 Tensor (e.g., [series_id, epoch]) used to
            guarantee spatial consistency across a series.
        - config: Application settings

    Returns:
        A Tuple containing:
        - final_img (tf.float32): The cropped (and potentially padded) image
          of shape (model_2d_height, model_2d_width, 1).
        - scaling_ratio (tf.float32): Always 1.0 (isometric crop).
        - offset_vector (tf.float32): Vector [x_offset, y_offset] (float32)
          containing negative integers representing the crop origin.
    """
    level_cfg = config['logging']['level']

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Starting function perform_deterministic_crop"),
        false_fn=lambda: tf.no_op()
    )

    models_cfg = config.get('models', None)
    if models_cfg is None:
        error_msg = (
            "Fatal error in perform_deterministic_crop: "
            "the setting variable 'models' is required "
            "but was not found. Please check your YAML file structure."
        )
        raise ValueError(error_msg)

    backbone_2d_cfg = models_cfg.get('backbone_2d', None)
    if backbone_2d_cfg is None:
        error_msg = (
            "Fatal error in perform_deterministic_crop: "
            "the setting variable 'models -> backbone_2d' is required "
            "but was not found. Please check your YAML file structure."
        )
        raise ValueError(error_msg)

    img_shape_cfg = backbone_2d_cfg.get('img_shape', None)
    if img_shape_cfg is None:
        error_msg = (
            "Fatal error in perform_deterministic_crop: "
            "the setting variable 'models -> backbone_2d -> img_shape' is required "
            "but was not found. Please check your YAML file structure."
        )
        raise ValueError(error_msg)

    model_2d_height, model_2d_width, _ = img_shape_cfg

    # 1. Padding if the image is too small
    # This ensures the image is AT LEAST the size of the model canvas
    pad_h = tf.maximum(model_2d_height, height_t)
    pad_w = tf.maximum(model_2d_width, width_t)

    # We pad the bottom/right with zeros if needed
    padded_img = tf.image.pad_to_bounding_box(
        raw_image_tf, 0, 0, pad_h, pad_w
    )

    # 2. Calculate available slack in the (now padded) image
    max_y = tf.maximum(0, pad_h - model_2d_height)
    max_x = tf.maximum(0, pad_w - model_2d_width)

    # 3. Generate offsets deterministically using the seed
    # seed is [series_id, epoch]
    off_y = tf.random.stateless_uniform([], seed=seed, minval=0, maxval=max_y + 1, dtype=tf.int32)

    # We use a slightly different seed for width to avoid diagonal correlation
    seed_w = seed + [0, 1]
    off_x = tf.random.stateless_uniform([], seed=seed_w, minval=0, maxval=max_x + 1, dtype=tf.int32)

    # 4. Perform the crop
    final_img = tf.image.crop_to_bounding_box(
        padded_img, off_y, off_x, model_2d_height, model_2d_width
    )

    # 5. Return results
    scaling_ratio = tf.constant(1.0, dtype=tf.float32)

    # The labels still follow the same negative offset logic
    offset_vector = tf.stack([
        -tf.cast(off_x, tf.float32),
        -tf.cast(off_y, tf.float32)
    ])

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Function perform_deterministic_crop completed"),
        false_fn=lambda: tf.no_op()
    )

    return tf.cast(final_img, tf.float32), scaling_ratio, offset_vector


def perform_resize(
    raw_image_tf: tf.Tensor,
    height_t: tf.Tensor,
    width_t: tf.Tensor,
    config: Dict[str, Any]
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Performs isometric resizing and centering.
    """

    level_cfg = config['logging']['level']

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Starting function perform_resize"),
        false_fn=lambda: tf.no_op()
    )

    models_cfg = config['models']
    backbone_2d_cfg = models_cfg['backbone_2d']
    img_shape_cfg = backbone_2d_cfg['img_shape']

    model_2d_height, model_2d_width, _ = img_shape_cfg

    # Maintain [X, Y] logic for scaling calculation
    current_dims = tf.cast(tf.stack([width_t, height_t]), tf.float32)
    target_dims = tf.constant([model_2d_width, model_2d_height], dtype=tf.float32)
    ratios = current_dims / target_dims
    scaling_ratio = tf.reduce_max(ratios)

    # Calculate new dimensions
    new_h = tf.cast(tf.cast(height_t, tf.float32) / scaling_ratio, tf.int32)
    new_w = tf.cast(tf.cast(width_t, tf.float32) / scaling_ratio, tf.int32)

    # IMPORTANT: tf.image functions use [HEIGHT, WIDTH] order
    resized_img = tf.image.resize(raw_image_tf, [new_h, new_w], method='bilinear', antialias=True)
    final_img = tf.image.resize_with_pad(resized_img, model_2d_height, model_2d_width)

    # Offset calculation for coordinate mapping [X, Y]
    width_offset_t = tf.cast((model_2d_width - new_w) // 2, tf.float32)
    height_offset_t = tf.cast((model_2d_height - new_h) // 2, tf.float32)

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Function perform_resize completed"),
        false_fn=lambda: tf.no_op()
    )

    return final_img, scaling_ratio, tf.stack([width_offset_t, height_offset_t])


def normalize_image(
    image_tf: tf.Tensor,
    series_min_t: tf.Tensor,
    series_max_t: tf.Tensor,
    config: Dict[str, Any]
) -> tf.Tensor:

    """
    Normalizes image intensity while preserving 0 as the padding/neutral value.

    This ensures that background padding remains at absolute zero even after
    Min-Max scaling and range shifting.
    """

    level_cfg = config['logging']['level']

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Starting function normalize_image"),
        false_fn=lambda: tf.no_op()
    )

    model_cfg = config["models"]

    backbone_2d_cfg = model_cfg["backbone_2d"]

    scaling_dict = backbone_2d_cfg["scaling"]

    min_scaling_value, max_scaling_value = (
        scaling_dict.get("min", None),
        scaling_dict.get("max", None)
    )

    # 2. Standard Normalization
    s_min = tf.cast(series_min_t, tf.float32)
    s_max = tf.cast(series_max_t, tf.float32)

    denom = tf.maximum(s_max - s_min, 1e-8)
    normalized = (tf.cast(image_tf, tf.float32) - s_min) / denom

    # Rescale to target range
    rescaled_image = normalized * (max_scaling_value - min_scaling_value) + min_scaling_value

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Function normalize_image completed"),
        false_fn=lambda: tf.no_op()
    )

    return rescaled_image


def process_study_multi_series(
    images: tf.Tensor,
    meta: Dict[str, tf.Tensor],
    labels: Dict[str, tf.Tensor],
    config: Dict[str, any],
    is_training: bool = True
) -> Tuple[
            Tuple[
                    Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
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
    2. Series Resolution: Selecting randomly one series among those available
       if multiple candidates exist.
    3. Temporal/Spatial Ordering: Ensuring all volumes are anatomically sorted
       before being fed to the model.
    4. Metadata/Label Compression: Reducing redundant frame-level records into
       single study-level identifiers and diagnostics.

    Args:
        - images (tf.Tensor): Flattened batch of all images in the study (N, H, W, C).
        - meta (dict): Dictionary of metadata tensors (each of length N).
        - labels (dict): Dictionary of label tensors (study-wide diagnostics).
        - config (dict): Dictionary of project settings
        - is_training (bool): flag on the current mode : training (True) or validation (False)

    Returns:
        tuple: (study_data_triplet, study_id, reduced_labels)
               - study_data_triplet: A nested structure containing 4-tuples:
                 (Volume, Slice_Metadata, Series_ID, Description_Code)
                 for each of the three required anatomical planes.
               - study_id: The unified identifier for the study (scalar).
               - reduced_labels: Study-level diagnostic labels (compacted).
    """
    level_cfg = config['logging']['level']

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Starting function process_study_multi_series"),
        false_fn=lambda: tf.no_op()
    )

    # --- 1. Define Search Targets (Integer Codes) ---
    # Based on the internal mapping: 0: "Sagittal T1", 1: "Sagittal T2", 2: "Axial T2"
    # Using constants ensures Graph-compatibility during execution.
    t1_code = tf.constant(0, dtype=tf.int32)
    t2_code = tf.constant(1, dtype=tf.int32)
    ax_code = tf.constant(2, dtype=tf.int32)

    # --- 2. Process Individual Branches ---
    # We extract the 3-tuple (Padded_Images, Selected_Series_ID, Description_Code) for each plane.
    # This handles series selection and black-frame padding internally.
    series_depth = config['series_depth']

    model_cfg = config["models"]
    backbone_2d_cfg = model_cfg["backbone_2d"]
    img_shape = backbone_2d_cfg["img_shape"]

    model_2d_height, model_2d_width, model_2d_nb_channels = img_shape

    res_t1 = process_single_series_description(
        images,
        meta,
        t1_code,
        series_depth,
        model_2d_height,
        model_2d_width,
        model_2d_nb_channels,
        config,
        is_training
    )

    res_t2 = process_single_series_description(
        images,
        meta,
        t2_code,
        series_depth,
        model_2d_height,
        model_2d_width,
        model_2d_nb_channels,
        config,
        is_training
    )

    res_ax = process_single_series_description(
        images,
        meta,
        ax_code,
        series_depth,
        model_2d_height,
        model_2d_width,
        model_2d_nb_channels,
        config,
        is_training
    )

    # --- 3. Extract Global Study Context ---
    # Since all frames in the window belong to the same study (guaranteed by _get_group_key),
    # we take the first element as the representative study_id.
    study_id = meta['study_id'][0]

    # --- 4. Label Consolidation (TensorFlow-native approach) ---
    # We use tf.nest.map_structure to apply the reduction to every element
    # in the labels dictionary without explicit Python loops.
    reduced_labels = tf.nest.map_structure(reduce_to_first_element, labels)

    # Extraction complete: res_t1, res_t2, and res_ax now hold their own extracted
    # and stacked tensors. The original 'images' and 'meta' can be released.
    # The labels can be removed after extraction
    del meta
    del images
    del labels

    # Return the structured data required by LumbarDicomTFRecordDataset._format_for_model :
    rtrn_value = (res_t1, res_t2, res_ax), study_id, reduced_labels

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Function process_study_multi_series completed"),
        false_fn=lambda: tf.no_op()
    )

    return rtrn_value


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
        lambda: v[0],  # Take first element if it's a batch/sequence
        lambda: v    # Keep as is if it's already a scalar
    )


def process_single_series_description(
    all_images: tf.Tensor,
    all_meta: Dict[str, tf.Tensor],
    target_desc_tensor: tf.Tensor,
    series_depth: int,
    height: int,
    width: int,
    nb_channels: int,
    config: Dict[str, Any],
    is_training: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    """
    Routes the study images to the appropriate processing logic based on anatomical view
    (the "series description").

    This function acts as a high-level router that:
    1. Filters the study-level collection for a specific series description (e.g., Axial T2).
    2. Checks if any frames exist for this view.
    3. Dispatches to 'process_valid_series' for sorting and shape enforcement if
       data is present, or to 'process_empty_series' to generate a zero-filled
       placeholder if the view is missing.

    Args:
        - all_images (tf.Tensor): Flattened batch of images for the study (N, H, W, C).
        - all_meta (Dict[str, tf.Tensor]): Metadata containing 'description',
                                         'series_id', and 'instance_number'.
        - target_desc_tensor (tf.Tensor): The anatomical view code to filter for.
        - series_depth (int): Expected depth of the output volume.
        - height (int): Target image height.
        - width (int): Target image width.
        - nb_channels (int): Target number of channels.
        - config (dict): Dictionary of project settings.
        - is_training (bool): Flag for training mode (True) or validation (False).

    Returns:
        A tuple containing:
            - final_vol (tf.Tensor): 3D volume (series_depth, H, W, 3) [float32].
            - slice_metadata (tf.Tensor): Per-slice metadata (series_depth, 5) [float32].
              Columns: [instance_number, is_padding, scaling_ratio, x_crop, y_crop].
            - series_metadata (tf.Tensor): Series info vector (3,) [int32].
              Values: [sampling_flag, first_slice_index, target_desc_tensor].
              Note: target_desc_tensor is the encoded anatomical view.
    """

    level_cfg = config['logging']['level']

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Starting function process_single_series_description"),
        false_fn=lambda: tf.no_op()
    )

    # Convert to int32 for stable comparison.
    current_desc = tf.cast(all_meta['series_description'], tf.int32)
    target_code = tf.cast(target_desc_tensor, tf.int32)

    # Create the filtering mask
    desc_mask = tf.equal(current_desc, target_code)
    desc_mask = tf.cast(desc_mask, tf.bool)

    # Set shapes for TensorFlow Graph compatibility
    desc_mask.set_shape([None])
    all_images.set_shape([None, None, None, None])

    # Check if any image matches the description
    mask_has_data = tf.reduce_any(desc_mask)

    # Execute conditional logic
    rtrn_value = tf.cond(
        mask_has_data,
        true_fn=lambda: process_valid_series(
            all_images, all_meta, desc_mask, target_desc_tensor,
            series_depth, height, width, nb_channels, config, is_training
        ),
        false_fn=lambda: process_empty_series(
            target_desc_tensor, series_depth, height, width, config
        )
    )

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Function process_single_series_description completed"),
        false_fn=lambda: tf.no_op()
    )

    return rtrn_value


def process_valid_series(
    all_images: tf.Tensor,
    all_meta: Dict[str, tf.Tensor],
    desc_mask: tf.Tensor,
    target_desc_tensor: tf.Tensor,
    series_depth: int,
    height: int,
    width: int,
    nb_channels: int,
    config: Dict[str, Any],
    is_training: bool = True
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    """
    Processes and normalizes a series that contains data for the target description.

    This function assumes that padding (to handle missing instances and reach
    series_depth) has been performed during TFRecord creation.
    It focuses on structural integrity:
    1. Filters images and metadata to isolate the target anatomical view.
    2. Selects randomly one series among those available if multiple candidates exist.
    3. Re-orders frames by instance number to ensure correct anatomical sequence.
    4. Handles slice sampling (Global vs Local) during training.
    5. Enforces a static 3D shape and converts grayscale to RGB if required.

    Args:
        - all_images (tf.Tensor): Full batch of images for a study (N, H, W, C).
        - all_meta (Dict[str, tf.Tensor]): Metadata containing 'series_id', 'instance_number', etc.
        - desc_mask (tf.Tensor): Boolean mask identifying images matching the target description.
        - target_desc_tensor (tf.Tensor): The description code being processed.
        - series_depth (int): Fixed number of slices required for the 3D volume.
        - height (int): Target image height.
        - width (int): Target image width.
        - nb_channels (int): Target number of channels (e.g., 3 for RGB).
        - is_training (bool): Flag for training mode to enable/disable random sampling.

    Returns:
       A tuple containing:
            - final_vol (tf.Tensor): The normalized 3D volume (series_depth, H, W, 3) [float32].
            - slice_metadata (tf.Tensor): Per-slice metadata matrix (series_depth, 4) [float32].
              Columns: [is_padding, scaling_ratio, x_crop, y_crop].
            - series_metadata (tf.Tensor): Series-level info vector (3,) [int32].
              Values: [sampling_flag, first_slice_index, target_desc_tensor].
    """

    level_cfg = config['logging']['level']

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Starting function process_valid_series"),
        false_fn=lambda: tf.no_op()
    )

    # --- 1. Filtering by Description ---
    d_images = tf.boolean_mask(all_images, desc_mask)
    d_series_ids = tf.boolean_mask(all_meta['series_id'], desc_mask)
    d_instances = tf.boolean_mask(all_meta['instance_number'], desc_mask)
    d_padding_flags = tf.boolean_mask(all_meta['is_padding'], desc_mask)
    d_scaling_ratios = tf.boolean_mask(all_meta['scaling_ratio'], desc_mask)
    d_x_crop = tf.boolean_mask(all_meta['x_crop'], desc_mask)
    d_y_crop = tf.boolean_mask(all_meta['y_crop'], desc_mask)

    # --- 2. Best Series Selection (Highest Frame Count) ---
    # Identify the number of series present for this anatomical description
    unique_ids, _ = tf.unique(d_series_ids)

    # Get the number of unique series found
    nb_unique_series = tf.shape(unique_ids)[0]

    # Generate a random index between 0 and num_unique_series - 1
    random_idx = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=nb_unique_series,
        dtype=tf.int32
    )

    # Select the series ID at that random position
    selected_series_id = unique_ids[random_idx]

    # --- 3. Isolate the chosen series
    series_mask = tf.equal(d_series_ids, selected_series_id)
    series_mask = tf.cast(series_mask, tf.bool)

    final_imgs = tf.boolean_mask(d_images, series_mask)
    final_instances = tf.boolean_mask(d_instances, series_mask)
    final_padding_flags = tf.boolean_mask(d_padding_flags, series_mask)
    final_scaling_ratios = tf.boolean_mask(d_scaling_ratios, series_mask)
    final_x_crop = tf.boolean_mask(d_x_crop, series_mask)
    final_y_crop = tf.boolean_mask(d_y_crop, series_mask)

    # --- 4. Spatial Sorting ---
    sort_idx = tf.argsort(final_instances)
    sorted_imgs = tf.gather(final_imgs, sort_idx)
    sorted_padding_flags = tf.gather(final_padding_flags, sort_idx)
    sorted_scaling_ratios = tf.gather(final_scaling_ratios, sort_idx)
    sorted_x_crop = tf.gather(final_x_crop, sort_idx)
    sorted_y_crop = tf.gather(final_y_crop, sort_idx)

    actual_count = tf.shape(sorted_imgs)[0]
    f_actual_count = tf.cast(actual_count, tf.float32)

    # 5. Select image sampling mode
    slice_sampling_flag, indices = tf.cond(
        tf.logical_and(is_training, tf.greater(actual_count, series_depth)),
        lambda: get_indices_on_images(actual_count, series_depth, config),
        # Default: Uniform Global sampling (for Val or small series)
        lambda: (
            False,
            tf.cast(
                tf.round(
                    tf.linspace(0.0, f_actual_count - 1.0, series_depth)
                ),
                tf.int32
            )
        )
    )

    # 6. Final formatting
    # We use a simple slice to be 100% sure of the depth for the Graph
    final_vol = tf.gather(sorted_imgs, indices)

    # 7. Final shape enforcement for the model*
    # Convert single-channel MRI to 3-channel (pseudo-RGB) to match model input requirements.
    nb_channels = tf.shape(final_vol)[-1]
    final_vol = tf.cond(
        tf.equal(nb_channels, 1),
        lambda: tf.image.grayscale_to_rgb(final_vol),
        lambda: final_vol
    )

    # 8. Stack the whole volume into a single tensor.
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
            tf.cast(tf.gather(sorted_padding_flags, indices), tf.float32),   # Padding flag
            tf.cast(tf.gather(sorted_scaling_ratios, indices), tf.float32),  # Scaling ratio
            tf.cast(tf.gather(sorted_x_crop, indices), tf.float32),          # x offset
            tf.cast(tf.gather(sorted_y_crop, indices), tf.float32)           # y offset
        ],
        axis=1
    )

    meta_shape = tf.stack([
        tf.cast(series_depth, tf.int32),
        tf.constant(4, dtype=tf.int32)
    ])
    slice_metadata = tf.reshape(slice_metadata, meta_shape)

    series_metadata = tf.stack(
        [
            tf.cast(slice_sampling_flag, tf.int32),  # Flag on sampling performed on series slices
            tf.cast(indices[0], tf.int32),          # Index on the first sampled slice in the series
            tf.cast(target_desc_tensor, tf.int32)    # Encoded anatomical view (target_desc_tensor)
        ],
        axis=0
    )

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Function process_valid_series completed"),
        false_fn=lambda: tf.no_op()
    )

    return final_vol, slice_metadata, series_metadata


def get_indices_on_images(
    actual_count_tf: tf.Tensor,
    series_depth: int,
    config: Dict[str, Any]
) -> Tuple[bool, int]:
    """
    Determines which slice indices to extract from a series based on a sampling strategy.

    During training, this helper dynamically chooses between two sampling modes
    to provide the model with different anatomical perspectives:

    - Global View (False): Resamples the entire available series to fit series_depth.
      This captures the full anatomical extent but with a higher stride between slices.
    - Local View (True): Selects a contiguous window of slices (stride 1).
      This captures fine-grained spatial details for a specific sub-section.

    Args:
        actual_count (tf.Tensor): The total number of available slices in the series (int32).
        series_depth (int): The required number of slices for the output volume.

    Returns:
        A tuple containing:
            - sampling_flag: Boolean flag (bool).
              True if Local View was used, False if Global View was used.
            - indices (tf.Tensor): Integer vector of shape (series_depth,)
              containing the calculated slice indices [int32].
    """

    level_cfg = config['logging']['level']

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Starting function get_indices_on_images"),
        false_fn=lambda: tf.no_op()
    )

    # Decide between Global Resampling or Local Window
    # We use a random variable to choose the mode
    use_global_view = tf.random.uniform([]) > 0.5

    rtrn_value = tf.cond(
        use_global_view,
        # MODE 1: Global View (Full anatomy, high stride)
        lambda: (
            False,
            tf.cast(
                tf.round(tf.linspace(0.0, tf.cast(actual_count_tf - 1, tf.float32), series_depth)),
                tf.int32
            )
        ),

        # MODE 2: Local View (Consecutive slices, stride ~1.0)
        lambda: (
            True,
            (lambda start_idx: tf.range(start_idx, start_idx + series_depth))(
                tf.random.uniform([], 0, actual_count_tf - series_depth, dtype=tf.int32)
            )
        )
    )

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Function get_indices_on_images completed"),
        false_fn=lambda: tf.no_op()
    )

    return rtrn_value


def process_empty_series(
    target_desc_tensor: tf.Tensor,
    series_depth: int,
    img_height: int,
    img_width: int,
    config: Dict[str, Any],
    nb_channels: int = 3
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    """
    Generates a placeholder volume when no data matches the target description.

    This ensures that the TensorFlow Graph receives a consistent tensor shape
    and data type even if the specific anatomical view is missing for a study.
    It maintains the same output signature as process_valid_series to allow
    conditional branching within the dataset pipeline.

    Args:
        - target_desc_tensor (tf.Tensor): The description code that was not found.
        - series_depth (int): Fixed number of slices for the output volume.
        - height (int): Target image height.
        - width (int): Target image width.
        - config (dict): project settings dictionary
        - nb_channels: Target image channels number

    Returns:
       A tuple containing:
            - empty_vol (tf.Tensor): A zero-filled 3D volume (series_depth, H, W, 3) [float32].
            - slice_metadata (tf.Tensor): Placeholder slice metadata (series_depth, 4) [float32].
              Default values: [is_padding=1.0, scaling=1.0, x_crop=0.0, y_crop=0.0].
            - series_metadata (tf.Tensor): Sentinel info vector (3,) [int32].
              Values: [sampling_flag=0, first_slice_index=0, target_desc_tensor].
    """
    level_cfg = config["logging"]["level"]

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Starting function process_empty_series"),
        false_fn=lambda: tf.no_op()
    )

    # 1. Constants and Type Casting
    s_depth = tf.cast(series_depth, tf.int32)
    height = tf.cast(img_height, tf.int32)
    width = tf.cast(img_width, tf.int32)
    channels = tf.constant(nb_channels, tf.int32)

    # 2. Build Image Volume Shape
    # Using tf.stack to create a dynamic shape tensor
    vol_shape = tf.stack([s_depth, height, width, channels])

    model_cfg = config["models"]
    backbone_2d_cfg = model_cfg["backbone_2d"]
    scaling_cfg = backbone_2d_cfg["scaling"]
    black_pixel_value = scaling_cfg['min']

    # Create and force shape via reshape (Graph safe)
    empty_vol = tf.fill(vol_shape, black_pixel_value)
    empty_vol = tf.cast(empty_vol, tf.float32)
    empty_vol = tf.reshape(empty_vol, vol_shape)

    # 3. Build Slice Metadata
    # We use tf.reshape instead of tf.ensure_shape for the same reasons
    meta_shape = tf.stack([s_depth, tf.constant(4, tf.int32)])

    slice_metadata = tf.stack(
        [
            tf.fill([s_depth], 1.0),   # Padding flag
            tf.fill([s_depth], 1.0),   # Scaling ratio
            tf.fill([s_depth], 0.0),   # x_crop
            tf.fill([s_depth], 0.0)    # y_crop
        ],
        axis=1
    )

    series_metadata = tf.stack(
        [
            tf.cast(0, tf.int32),                   # Flag on sampling performed on series slices
            tf.cast(0, tf.int32),                   # Index on the first sampled slice in the series
            tf.cast(target_desc_tensor, tf.int32)   # Encoded anatomical view (target_desc_tensor)
        ],
        axis=0
    )

    # Finalize shape for the metadata tensor
    slice_metadata = tf.reshape(slice_metadata, meta_shape)

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Function process_empty_series completed"),
        false_fn=lambda: tf.no_op()
    )

    return empty_vol, slice_metadata, series_metadata


def format_for_model(
    study_volumes_tf: Tuple[
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ],
    study_id_tf: tf.Tensor,
    labels: Dict[str, tf.Tensor],
    config: Dict[str, Any]
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:

    """
    Final mapping stage that adapts study-level volumes and metadata
    to the specific multi-input/multi-output signature of the model.

    Args:
        - study_volumes_tf (tuple): Tensors for (Sagittal T1, Sagittal T2, Axial T2)
                                each with shape (self._series_depth, H, W, C).
        - study_id_tf (tf.Tensor): The unique identifier for the study.
        - labels (dict): Dictionary containing the 'records' tensor for diagnosis.
        - config (dict): Application settings

    Returns:
        tuple: (inputs_dict, targets_dict) ready for model.fit().
    """

    level_cfg = config["logging"]["level"]

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Starting function format_for_model"),
        false_fn=lambda: tf.no_op()
    )

    # Extract config values
    model_cfg = config["models"]
    backbone_2d_cfg = model_cfg["backbone_2d"]
    img_shape = backbone_2d_cfg["img_shape"]

    series_depth = config["series_depth"]

    data_specs_cfg = config["data_specs"]
    max_records = data_specs_cfg["max_records_per_frame"]

    # Target shapes for validation
    model_2d_height, model_2d_width, model_2d_nb_channels = img_shape
    target_shape = [series_depth, img_shape[0], img_shape[1], img_shape[2]]
    meta_target_shape = [series_depth, 4]
    series_target_shape = [3]

    # Unpack processed volumes from the previous study-level processing step
    sag_t1, sag_t2, axial = study_volumes_tf

    # --- 2. Build Inputs Dictionary ---
    # These keys MUST exactly match the names defined in ModelFactory.build_multi_series_model()
    # Note: tensors are assumed to be float32 from previous steps.
    # Use tf.ensure_shape to guarantee dimensions without breaking the graph flow
    features = {
        "study_id": tf.reshape(tf.cast(study_id_tf, tf.int64), [1]),

        # Sagittal T1
        "img_sag_t1": tf.ensure_shape(sag_t1[0], target_shape),
        "slice_metadata_t1": tf.ensure_shape(sag_t1[1], meta_target_shape),
        "series_metadata_t1": tf.reshape(sag_t1[2], series_target_shape),

        # Sagittal T2
        "img_sag_t2": tf.ensure_shape(sag_t2[0], target_shape),
        "slice_metadata_t2": tf.ensure_shape(sag_t2[1], meta_target_shape),
        "series_metadata_t2": tf.reshape(sag_t2[2], series_target_shape),

        # Axial T2
        "img_axial_t2": tf.ensure_shape(axial[0], target_shape),
        "slice_metadata_axial_t2": tf.ensure_shape(axial[1], meta_target_shape),
        "series_metadata_axial_t2": tf.reshape(axial[2], series_target_shape)
    }

    # Explicitly clear the unpacking tuples to free references
    del sag_t1, sag_t2, axial, study_volumes_tf

    # --- 3. Build Targets Dictionary ---

    # Traceability: Pass the study_id back out to verify data integrity during inference
    # Expanded to (1,) or (batch, 1) to match the Lambda layer output shape
    # labels_dict["study_id_output"] = tf.reshape(tf.cast(study_id_tf, tf.float32), [1])

    # Diagnosis: Reshape and map the 25 level records
    # records shape: (MAX_RECORDS, 4) -> [condition_level, severity, x, y]
    records_raw = tf.reshape(labels["records"], (max_records, 4))

    # Sorting: use condition_level (col 0) as integer for stable sorting
    condition_level_ids = tf.cast(records_raw[:, 0], tf.int32)
    sort_indices = tf.argsort(condition_level_ids, direction="ASCENDING")

    # 2. Reorder the entire records tensor using these indices
    sorted_records = tf.gather(records_raw, sort_indices)

    # Final labels formatting
    # Classification target (Severity: 0, 1, or 2): explicit cast to int32
    # before one-hot encoding.
    severity_labels = tf.cast(sorted_records[:, 1], tf.int32)
    labels_dict = {
        "severity_output": tf.cast(
            tf.one_hot(severity_labels, depth=3),
            tf.float32
        ),
        "location_output": tf.cast(
            sorted_records[:, 2:4],
            tf.float32
        )
    }

    # Clean up intermediate label tensors
    del records_raw, sorted_records, sort_indices

    # --- Debug & Assertions (Only if log level is DEBUG) ---
    if config["logging"]["level"] == "DEBUG":

        # We define a function that performs all type assertions
        # This is wrapped so it only exists in the Graph if DEBUG is True
        def perform_debug_assertions():
            # Assertions for Features (Inputs)
            # Study ID is the only int64
            tf.debugging.assert_type(
                features["study_id"],
                tf.int64,
                message="study_id must be int64"
            )

            # Images and Slice Metadata MUST be float32
            tf.debugging.assert_type(features["img_sag_t1"], tf.float32)
            tf.debugging.assert_type(features["img_sag_t2"], tf.float32)
            tf.debugging.assert_type(features["img_axial_t2"], tf.float32)
            tf.debugging.assert_type(features["slice_metadata_t1"], tf.float32)

            # Series metadata are categorical codes (int32)
            tf.debugging.assert_type(features["series_metadata_t1"], tf.int32)

            # Assertions for Labels (Targets)
            tf.debugging.assert_type(labels_dict["severity_output"], tf.float32)
            tf.debugging.assert_type(labels_dict["location_output"], tf.float32)

            # Optional: Use tf.print to log dtypes in the console during execution
            # This will show up in your terminal/logs while the dataset is running
            tf.print("DEBUG: Dataset Batch types verified (float32/int32/int64 check passed)")

            return tf.constant(True)  # tf.cond needs a return value

        # We trigger the assertions within the Graph
        _ = tf.cond(tf.constant(True), perform_debug_assertions, lambda: tf.constant(True))

    tf.cond(
        tf.equal(level_cfg, "DEBUG"),
        true_fn=lambda: tf.print("Function format_for_model completed"),
        false_fn=lambda: tf.no_op()
    )

    return features, labels_dict
