# coding utf-8

import tensorflow as tf
import numpy as np
import pytest
import os
import src.core.utils.dataset_utils as ds_utils

from unittest.mock import patch
from contextlib import ExitStack
from src.core.utils.dataset_utils import (
    parse_tfrecord_single_element,
    normalize_image,
    reduce_to_first_element,
    process_single_series_description,
    process_valid_series,
    process_empty_series,
    format_for_model
)

# Limit the system messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=All, 1=No Info, 2=No Info/Warn, 3=No Info/Warn/Error

# Constants for testing
TEST_WIDTH = 270
TEST_HEIGHT = 512
TEST_DEPTH = 10
TEST_CHANNELS = 1
MAX_RECORDS = 25


@pytest.fixture
def mock_tfrecord_example():
    """
    Helper to create a serialized tf.train.Example.
    """
    # Create a dummy 10x10 image as uint16
    img_array = np.ones((10, 10), dtype=np.uint16) * 500
    img_raw = img_array.tobytes()
    is_padding = False

    # Create dummy records (25 records * 4 values = 100 floats)
    # Record format: [condition_level, severity, x, y]
    # We set one record at [1, 2, 5, 5] (x=5, y=5 on a 10x10 image)
    records = np.zeros(100, dtype=np.float32)
    records[0:4] = [1.0, 2.0, 5.0, 5.0]

    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'is_padding': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(is_padding)])),
        'file_format': tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=[img_array.shape[0], img_array.shape[1]]
            )
        ),
        'study_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[123])),
        'series_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[456])),
        'series_min': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
        'series_max': tf.train.Feature(int64_list=tf.train.Int64List(value=[1000])),
        'instance_number': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
        'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[10])),
        'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[10])),
        'series_description': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
        'records': tf.train.Feature(float_list=tf.train.FloatList(value=records)),
        'nb_records': tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def test_parse_tfrecord_single_element(
    mock_config,
    mock_tfrecord_example
):
    """
    Test the full parsing logic including resizing,
    coordinate normalization, and dictionary structure.
    """

    with ExitStack() as stack:

        # 1. Setup
        # Mock normalize_image to return the input as float32
        normalize_image_path = 'src.core.utils.dataset_utils.normalize_image'
        mock_normalize = stack.enter_context(patch(normalize_image_path))
        mock_normalize.side_effect = lambda **kwargs: kwargs.get('image_tf')

        # 2. Execution
        current_epoch_tf = tf.constant(1, dtype=tf.int64)
        image, metadata, labels = parse_tfrecord_single_element(
            mock_tfrecord_example,
            current_epoch_tf,
            mock_config
        )

        # 3. Assertions: Image
        # Check resizing (270, 512) and type (float16)
        img_shape = mock_config.get('models').get('backbone_2d').get('img_shape')
        img_height = img_shape[0]
        img_width = img_shape[1]
        assert image.shape == (img_height, img_width, 1)
        assert image.dtype == tf.float16

        # 4. Assertions: Metadata
        assert metadata['study_id'].numpy() == 123
        assert metadata['series_id'].numpy() == 456
        assert metadata['instance_number'].numpy() == 1

        # 5. Assertions: Labels (Coordinate Normalization)
        # English comment: x=5 on width=10 should be 0.5 normalized
        records = labels['records'].numpy()
        assert records.shape == (25, 4)
        assert records[0, 0] == 1.0  # Condition_level
        assert records[0, 1] == 2.0  # Severity
        assert records[0, 2] == 0.5  # x_norm (5/10)
        assert records[0, 3] == 0.5  # y_norm (5/10)


def test_normalization_call_args(mock_tfrecord_example, mock_config):
    """
    Specifically verify that normalize_image is called with
    correct series_min and series_max values.
    """

    with patch('src.core.utils.dataset_utils.normalize_image') as mock_norm:
        mock_norm.return_value = tf.zeros((256, 256, 1), dtype=tf.float32)

        current_epoch_tf = tf.constant(1, dtype=tf.int64)
        parse_tfrecord_single_element(
            mock_tfrecord_example,
            current_epoch_tf,
            mock_config
        )

        # Verify that normalization bounds (0 and 1000) match
        # the mock_tfrecord_example definition.
        _, kwargs = mock_norm.call_args
        assert kwargs['series_min_t'].numpy() == 0
        assert kwargs['series_max_t'].numpy() == 1000


@pytest.fixture
def mock_scaling_bounds():
    """
    Patch global scaling values to target a range of [-1, 1].
    """
    with patch('src.core.utils.dataset_utils.MIN_SCALING_VALUE', -1.0), \
         patch('src.core.utils.dataset_utils.MAX_SCALING_VALUE', 1.0):
        yield


def test_normalize_image_standard_range(mock_config):
    """
    Test normalization with standard values.
    Input image has values [0, 500, 1000] with min=0 and max=1000.
    Expected output in range [-1, 1] should be [-1, 0, 1].
    """
    # 1. Setup
    image = tf.constant([0.0, 500.0, 1000.0], dtype=tf.float32)
    s_min = tf.constant(0, dtype=tf.int64)
    s_max = tf.constant(1000, dtype=tf.int64)

    # 2. Execution
    normalized = normalize_image(image, s_min, s_max, mock_config)

    # 3. Verification
    # 0.0 -> becomes -1.0
    # 500.0 -> becomes 0.0
    # 1000.0 -> becomes 1.0
    expected = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(normalized.numpy(), expected, atol=1e-5)


def test_normalize_image_division_by_zero(mock_config):
    """
    Verify that a zero-range series (max == min)
    does not cause a crash thanks to epsilon.
    """
    # Image values are all 500, min and max are also 500.
    image = tf.constant([500.0, 500.0], dtype=tf.float32)
    s_min = tf.constant(500, dtype=tf.int64)
    s_max = tf.constant(500, dtype=tf.int64)

    # Execution should not raise ZeroDivisionError
    normalized = normalize_image(image, s_min, s_max, mock_config)

    # Verification: (500-500)/epsilon = 0.
    # 0 * (1 - (-1)) + (-1) = -1.0
    # Verify that each pixel in the image is equal to -1.0
    assert np.all(normalized.numpy() == -1.0)


def test_normalize_image_out_of_bounds(mock_config):
    """
    Check behavior when image pixels are outside
    the provided series min/max.
    """
    image = tf.constant([-100.0, 1100.0], dtype=tf.float32)
    s_min = tf.constant(0, dtype=tf.int64)
    s_max = tf.constant(1000, dtype=tf.int64)

    normalized = normalize_image(image, s_min, s_max, mock_config)

    # -100 becomes -1.2 (outside target range)
    # 1100 becomes 1.2
    assert normalized.numpy()[0] < -1.0
    assert normalized.numpy()[1] > 1.0


@pytest.fixture
def mock_study_env():
    with ExitStack() as stack:
        # Patching module-level constants using patch.object
        stack.enter_context(patch.object(ds_utils, 'MODEL_2D_WIDTH', TEST_WIDTH))
        stack.enter_context(patch.object(ds_utils, 'MODEL_2D_HEIGHT', TEST_HEIGHT))
        stack.enter_context(patch.object(ds_utils, 'MAX_SERIES_DEPTH', TEST_DEPTH))
        stack.enter_context(patch.object(ds_utils, 'MODEL_2D_NB_CHANNELS', TEST_CHANNELS))

        mock_proc_single_desc = stack.enter_context(
            patch.object(
                ds_utils,
                'process_single_series_description'
            )
        )

        mock_reduce = stack.enter_context(
            patch.object(
                ds_utils,
                'reduce_to_first_element'
            )
        )

        yield mock_proc_single_desc, mock_reduce


class TestProcessStudyMultiSeries:

    def test_nominal_case(self, mock_config):
        """
        Tests the standard orchestration with all three series present.
        """
        with ExitStack() as stack:
            mock_proc_single_desc = stack.enter_context(
                patch.object(
                    ds_utils,
                    'process_single_series_description'
                )
            )

            mock_reduce = stack.enter_context(
                patch.object(
                    ds_utils,
                    'reduce_to_first_element'
                )
            )

            # Using constants for tensor shapes to ensure consistency
            series_depth = mock_config['series_depth']
            img_shape = mock_config['models']['backbone_2d']['img_shape']
            img_height = img_shape[0]
            img_width = img_shape[1]
            img_channels = 1
            standard_shape = (series_depth, img_height, img_width, img_channels)

            mock_proc_single_desc.side_effect = [
                (tf.ones(standard_shape), tf.constant(101), tf.constant(0)),
                (tf.ones(standard_shape), tf.constant(102), tf.constant(1)),
                (tf.ones(standard_shape), tf.constant(103), tf.constant(2))
            ]
            mock_reduce.side_effect = lambda x: x[0]

            meta = {'study_id': tf.constant([999, 999], dtype=tf.int64)}

            # Batch size of 2 for input simulation
            images = tf.zeros((2, img_height, img_width, img_channels))
            labels = {'label_a': tf.constant([1, 1])}

            triplet, study_id, reduced = ds_utils.process_study_multi_series(
                images,
                meta,
                labels,
                mock_config
            )

            assert len(triplet) == 3
            assert study_id.numpy() == 999
            assert mock_proc_single_desc.call_count == 3

    def test_missing_series_resilience(self, mock_config):
        """
        Tests if the function handles a missing series (e.g., Axial T2 missing).
        """

        with ExitStack() as stack:
            mock_proc_single_desc = stack.enter_context(
                patch.object(
                    ds_utils,
                    'process_single_series_description'
                )
            )

            # Using constants for tensor shapes to ensure consistency
            series_depth = mock_config['series_depth']
            img_shape = mock_config['models']['backbone_2d']['img_shape']
            img_height = img_shape[0]
            img_width = img_shape[1]
            img_channels = 1
            standard_shape = (series_depth, img_height, img_width, img_channels)

            mock_proc_single_desc.side_effect = [
                (tf.ones(standard_shape), tf.constant(101), tf.constant(0)),
                (tf.ones(standard_shape), tf.constant(102), tf.constant(1)),
                (tf.zeros(standard_shape), tf.constant(-1), tf.constant(2))  # Missing 'Axial T2'
            ]

            # Input number of frames can differ from series_depth
            num_input_frames = 1
            input_images = tf.zeros((num_input_frames, img_height, img_width, img_channels))
            triplet, _, _ = ds_utils.process_study_multi_series(
                input_images,
                {'study_id': tf.constant([1], dtype=tf.int64)},
                {},
                mock_config
            )

            # Verify output depth matches img_depth even if input was different
            for idx in range(3):
                assert triplet[0][0].shape[0] == series_depth, f"Depth mismatch at index {idx}"

            error_msg = "'Axial T2' series should be empty (padded)"
            assert tf.reduce_sum(triplet[2][0]).numpy() == 0, error_msg
            assert triplet[0][1] == 101, "Valid series ID should be preserved"
            assert triplet[2][1].numpy() == -1, "Missing series ID should be -1"

    def test_tf_graph_compatibility(self, mock_config):
        """
        Verifies the function can be traced by tf.function (no Python-side leaks).
        """
        with ExitStack() as stack:
            mock_proc_single_desc = stack.enter_context(
                patch.object(
                    ds_utils,
                    'process_single_series_description'
                )
            )

            mock_reduce = stack.enter_context(
                patch.object(
                    ds_utils,
                    'reduce_to_first_element'
                )
            )

            # Using constants for tensor shapes to ensure consistency
            series_depth = mock_config['series_depth']
            img_shape = mock_config['models']['backbone_2d']['img_shape']
            img_height = img_shape[0]
            img_width = img_shape[1]
            img_channels = 1

            # Define expected shapes for the graph to validate
            standard_shape = (series_depth, img_height, img_width, img_channels)

            mock_proc_single_desc.return_value = (
                tf.zeros(standard_shape),
                tf.constant(0),
                tf.constant(0)
            )
            mock_reduce.side_effect = lambda x: x[0]

            # Wrap the target function in a static graph
            @tf.function
            def wrapped_call(imgs, metadata, labels):
                return ds_utils.process_study_multi_series(
                    imgs,
                    metadata,
                    labels,
                    mock_config
                )

            # Prepare minimal valid inputs
            meta = {'study_id': tf.constant([1], dtype=tf.int64)}
            images = tf.zeros((1, img_height, img_width, img_channels))
            labels = {'target': tf.constant([1])}

            # Execution
            triplet, study_id, reduced_labels = wrapped_call(images, meta, labels)

            # Explicit assertions to confirm graph output integrity
            assert isinstance(study_id, tf.Tensor), "Output study_id must be a Tensor"
            assert study_id.shape == (), "Study ID should be a scalar tensor"

            # Validate that the triplet maintains its shape through the graph
            assert len(triplet) == 3
            error_msg = f"Expected shape {standard_shape}, got {triplet[0][0].shape}"
            assert triplet[0][0].shape == standard_shape, error_msg

            # Ensure labels were reduced correctly within the graph
            assert isinstance(reduced_labels, dict)
            assert 'target' in reduced_labels


class TestReduceToFirstElement:

    def test_reduce_vector_to_scalar(self):
        """
        Tests that a 1D tensor is reduced to its first element.
        """
        # Input rank 1 -> Output rank 0
        v = tf.constant([42, 43, 44])
        result = reduce_to_first_element(v)

        assert result.numpy() == 42
        assert result.shape == ()

    def test_reduce_matrix_to_vector(self):
        """
        Tests that a 2D tensor is reduced to its first row.
        """
        # Input rank 2 -> Output rank 1 (v[0])
        v = tf.constant([[1, 2], [3, 4]])
        result = reduce_to_first_element(v)

        # Should return the first row [1, 2]
        tf.debugging.assert_equal(result, tf.constant([1, 2]))

        assert result.shape == (2,)

    def test_keep_scalar_as_is(self):
        """
        Tests that a 0D tensor (scalar) is returned unchanged.
        """
        # Input rank 0 -> Output rank 0
        v = tf.constant(7)
        result = reduce_to_first_element(v)

        assert result.numpy() == 7
        assert result.shape == ()

    def test_reduce_in_tf_graph(self):
        """
        Verifies the function works correctly inside a compiled tf.function.
        """
        # This ensures tf.cond and tf.rank are behaving in Graph mode
        @tf.function
        def wrapped_reduce(tensor):
            return reduce_to_first_element(tensor)

        v = tf.constant([10, 20])
        result = wrapped_reduce(v)

        assert result.numpy() == 10


class TestProcessSingleDescription:

    @pytest.fixture
    def sample_data(self, mock_config):
        # Setup basic tensors for testing filtering logic
        nb_images = 4

        img_shape = mock_config['models']['backbone_2d']['img_shape']
        img_height = img_shape[0]
        img_width = img_shape[1]
        img_channels = 1

        series_depth = mock_config['series_depth']

        images = tf.zeros((nb_images, img_height, img_width, img_channels))
        meta = {
            'series_description': tf.constant([0, 1, 0, 2], dtype=tf.int32),
            'series_id': tf.constant([10, 11, 10, 12], dtype=tf.int64),
            'instance_number': tf.constant([1, 1, 2, 1], dtype=tf.int32),
            'is_padding': tf.constant([0, 0, 1, 0], dtype=tf.int32),
            'scaling_ratio': tf.constant([0.8, 0.8, 0.9, 0.8], dtype=tf.float32),
            'x_crop': tf.constant([70, 90, 0, 45], dtype=tf.float32),
            'y_crop': tf.constant([65, 120, 55, 0], dtype=tf.float32),
        }
        return images, meta, series_depth, img_height, img_width, img_channels

    def test_routing_to_valid_series(self, sample_data, mock_config):
        """
        Tests that the function routes to process_valid_series when a match is found.
        """

        images, meta, series_depth, img_height, img_width, img_channels = sample_data
        target_code = tf.constant(1, dtype=tf.int32)  # Matches second element

        # Mocking sub-functions to verify routing
        with ExitStack() as stack:
            # Enter multiple contexts without backslashes
            mock_valid = stack.enter_context(patch.object(ds_utils, 'process_valid_series'))
            mock_empty = stack.enter_context(patch.object(ds_utils, 'process_empty_series'))

            mock_valid.return_value = (
                tf.zeros(
                    (
                        series_depth,
                        img_height,
                        img_width,
                        img_channels
                     )
                ),
                tf.constant(11, dtype=tf.int64),
                target_code
            )

            result = process_single_series_description(
                images,
                meta,
                target_code,
                series_depth,
                img_height,
                img_width,
                img_channels,
                mock_config
            )

            # IMPORTANT - Force execution by evaluating one of the tensors
            # This triggers tf.cond to actually call the mocked function
            _ = [t.numpy() for t in result] if isinstance(result, tuple) else result.numpy()

            # English comment: Assert that the 'True' branch was taken
            mock_valid.assert_called_once()
            mock_empty.assert_not_called()

    def test_routing_to_empty_series(self, sample_data, mock_config):
        """
        Tests that the function routes to process_empty_series when no match is found.
        """

        images, meta, series_depth, img_height, img_width, img_channels = sample_data
        target_code = tf.constant(99, dtype=tf.int32)  # No match in description [0, 1, 0, 2]

        # Mocking sub-functions to verify routing
        with ExitStack() as stack:

            # Patch the functions inside the module where process_single_series_description lives
            # Replace 'src.core.utils.dataset_utils' with the actual import path if different

            # Enter multiple contexts without backslashes
            mock_valid = stack.enter_context(patch.object(ds_utils, 'process_valid_series'))
            mock_empty = stack.enter_context(patch.object(ds_utils, 'process_empty_series'))

            mock_empty.return_value = (
                tf.zeros((series_depth, img_height, img_width, img_channels)),
                tf.constant(-1, dtype=tf.int64),
                target_code
            )

            result = process_single_series_description(
                images,
                meta,
                target_code,
                series_depth,
                img_height,
                img_width,
                img_channels,
                mock_config
            )

            # IMPORTANT - Force execution by evaluating one of the tensors
            # This triggers tf.cond to actually call the mocked function
            _ = [t.numpy() for t in result] if isinstance(result, tuple) else result.numpy()

            # Assert that the 'False' branch was taken
            mock_empty.assert_called_once()
            mock_valid.assert_not_called()

    def test_type_resilience_casting(self, sample_data, mock_config):
        """
        Verifies that the function correctly handles mixed integer types (int32 vs int64).
        """

        images, meta, series_depth, img_height, img_width, img_channels = sample_data

        # Target is int32, but meta['description'] will be cast to int64
        target_code = tf.constant(2, dtype=tf.int32)

        # Mocking sub-functions to verify routing
        with ExitStack() as stack:
            # Enter multiple contexts without backslashes
            mock_valid = stack.enter_context(patch.object(ds_utils, 'process_valid_series'))
            stack.enter_context(patch.object(ds_utils, 'process_empty_series'))

            mock_valid.return_value = (
                tf.zeros((series_depth, img_height, img_width, img_channels)),
                tf.constant(12, dtype=tf.int64),
                target_code
            )

            result = process_single_series_description(
                images,
                meta,
                target_code,
                series_depth,
                img_height,
                img_width,
                img_channels,
                mock_config
            )

            # IMPORTANT - Force execution by evaluating one of the tensors
            # This triggers tf.cond to actually call the mocked function
            _ = [t.numpy() for t in result] if isinstance(result, tuple) else result.numpy()

            # If casting fails, tf.equal would raise an error before this
            assert mock_valid.called

    def test_graph_compatibility_smoke(self, sample_data, mock_config):
        """
        Checks if the conditional logic is traceable by tf.function.
        """
        images, meta, series_depth, img_height, img_width, img_channels = sample_data

        target_code = tf.constant(0, dtype=tf.int32)

        @tf.function
        def graph_fn(imgs, mt, tc):
            return process_single_series_description(
                images,
                meta,
                target_code,
                series_depth,
                img_height,
                img_width,
                img_channels,
                mock_config
            )

        # We don't mock sub-functions here to ensure the full graph is valid
        # Assuming process_valid/empty_series are also graph-compatible
        try:
            graph_fn(images, meta, target_code)
        except Exception as e:
            pytest.fail(f"tf.function tracing failed: {e}")


class TestProcessValidSeries:

    @pytest.fixture
    def mock_sample_data(self, mock_config):
        # Create 15 mock images (N=15)
        nb_images = 15
        nb_channels = 1
        img_shape = mock_config['models']['backbone_2d']['img_shape']
        img_height = img_shape[0]
        img_width = img_shape[1]

        all_images = tf.random.uniform(
            shape=(nb_images, img_height, img_width, nb_channels),
            minval=0,
            maxval=256,
            seed=42,
            dtype=tf.int32
        )

        all_meta = {
            # 15 images total, we'll mask some later
            'series_description': tf.constant(
                [idx % 3 for idx in range(nb_images)],
                dtype=tf.int32
            ),

            'series_id': tf.constant([5]*2 + [10]*4 + [15]*5 + [20]*4, dtype=tf.int64),

            'instance_number': tf.constant(
                [idx for idx in reversed(range(nb_images))],
                dtype=tf.int32
            ),

            'is_padding': tf.constant(
                [0 if idx % 3 == 0 else 1 for idx in range(nb_images)],
                dtype=tf.int32
            ),

            'scaling_ratio': tf.random.uniform(
                shape=(nb_images,),
                seed=42,
                dtype=tf.float32
            ),

            'x_crop': tf.random.uniform(
                shape=(nb_images,),
                minval=0,
                maxval=256,
                seed=42,
                dtype=tf.int32
            ),

            'y_crop': tf.random.uniform(
                shape=(nb_images,),
                minval=0,
                maxval=256,
                seed=42,
                dtype=tf.int32
            ),
        }

        # Mask to keep all the images of the series
        desc_mask = tf.ones((15,), dtype=tf.bool)
        target_desc = tf.constant(1, dtype=tf.int32)

        return (
            all_images, all_meta, nb_images, img_height,
            img_width, nb_channels, desc_mask, target_desc
        )

    def test_anatomical_sorting(self, mock_sample_data):
        """
        Verifies that frames are sorted by instance_number.
        """
        (
            imgs, meta, _, img_height,
            img_width, nb_channels, _, target_desc
        ) = mock_sample_data

        # Update the mask to keep only all the images of the series 20
        # Therefore, the series depth will be set to 4 images
        mask = tf.equal(meta['series_id'], 20)
        nb_images = 4

        target_channels = 1

        # For series 20, instance_numbers are [9, 8, ..., 0] (reversed order)
        # After sorting, the first frame in padded_vol should be the one that was at index 14
        padded_vol, _, _ = process_valid_series(
            all_images=imgs,
            all_meta=meta,
            desc_mask=mask,
            target_desc_tensor=target_desc,
            series_depth=nb_images,
            height=img_height,
            width=img_width,
            nb_channels=target_channels
        )

        # The last image of the stack (idx 14) had instance_number 0
        # After sorting, instance 0 should be at the beginning of the volume.
        expected_first_frame = imgs[14]
        tf.debugging.assert_equal(
            tf.cast(padded_vol[0], tf.int32),
            expected_first_frame
        )

    def test_grayscale_to_rgb_conversion(self, mock_sample_data):
        """
        Validates channel expansion from 1 to 3.
        """
        (
            imgs, meta, nb_images, img_height,
            img_width, nb_channels, mask, target_desc
        ) = mock_sample_data

        padded_vol, _, _ = process_valid_series(
            all_images=imgs,
            all_meta=meta,
            desc_mask=mask,
            target_desc_tensor=target_desc,
            series_depth=nb_images,
            height=img_height,
            width=img_width,
            nb_channels=3
        )

        # English comment: Shape must be (Depth, Height, Width, 3)
        assert padded_vol.shape == (nb_images, img_height, img_width, 3)


class TestProcessEmptySeries:

    @pytest.fixture
    def mock_sample_data(self, mock_config):
        # Create 15 mock images (N=15)
        nb_images = 15
        nb_channels = 1
        img_shape = mock_config['models']['backbone_2d']['img_shape']
        img_height = img_shape[0]
        img_width = img_shape[1]

        all_images = tf.random.uniform(
            shape=(nb_images, img_height, img_width, nb_channels),
            minval=0,
            maxval=256,
            seed=42,
            dtype=tf.int32
        )

        all_meta = {
            # 15 images total, we'll mask some later
            'series_description': tf.constant(
                [idx % 3 for idx in range(nb_images)],
                dtype=tf.int32
            ),

            'series_id': tf.constant([5]*2 + [10]*4 + [15]*5 + [20]*4, dtype=tf.int64),

            'instance_number': tf.constant(
                [idx for idx in reversed(range(nb_images))],
                dtype=tf.int32
            ),

            'is_padding': tf.constant(
                [0 if idx % 3 == 0 else 1 for idx in range(nb_images)],
                dtype=tf.int32
            ),

            'scaling_ratio': tf.random.uniform(
                shape=(nb_images,),
                seed=42,
                dtype=tf.float32
            ),

            'x_crop': tf.random.uniform(
                shape=(nb_images,),
                minval=0,
                maxval=256,
                seed=42,
                dtype=tf.int32
            ),

            'y_crop': tf.random.uniform(
                shape=(nb_images,),
                minval=0,
                maxval=256,
                seed=42,
                dtype=tf.int32
            ),
        }

        # Mask to keep all the images of the series
        desc_mask = tf.ones((15,), dtype=tf.bool)
        target_desc = tf.constant(1, dtype=tf.int32)

        return (
            all_images, all_meta, nb_images, img_height,
            img_width, nb_channels, desc_mask, target_desc
        )

    def test_output_shapes_and_normalization_values(self, mock_sample_data, mock_config):
        """
        Verify that the function returns the correct tensor shapes and
        uses -1.0 as the filler value (Black for MobileNetV2 normalization).
        """
        # 1. Extract dimensions from the mock fixture
        (
            _, _, _, img_height,
            img_width, _, _, _
        ) = mock_sample_data

        # Test parameters
        target_series_desc = tf.constant(5, dtype=tf.int32)
        series_depth = 10

        # 2. Call the function under test
        empty_vol, slice_meta, series_meta = process_empty_series(
            target_desc_tensor=target_series_desc,
            series_depth=series_depth,
            img_height=img_height,
            img_width=img_width,
            config=mock_config
        )

        # 3. Assertions for Shapes
        # Standard Python assert is fine for shape tuples
        assert empty_vol.shape == (series_depth, img_height, img_width, 3)
        assert slice_meta.shape == (series_depth, 4)
        assert series_meta.shape == (3,)

        # 4. Assertions for Values (Normalization Check)
        # The 'black' pixels value after normalization depends on the chosen 2D backbone model.
        # Example: or MobileNetV2, 'black' pixels must be -1.0 after normalization.
        # This value is set in the config parameters.
        # We calculate the expected sum of all pixels in the volume.
        black_pixel_value = mock_config.get('models').get('backbone_2d').get('scaling').get('min')
        num_pixels = tf.cast(series_depth * img_height * img_width * 3, tf.float32)
        expected_total_sum = num_pixels * black_pixel_value

        # Use tf.debugging to handle tensor comparisons accurately
        tf.debugging.assert_near(
            tf.reduce_sum(empty_vol),
            expected_total_sum,
            atol=1e-3,
            message="The empty volume must be filled with -1.0 (MobileNetV2 Black)."
        )

        # 5. Assertions for Metadata (Sentinel Values)
        # Column 0 of slice_meta is 'is_padding', which should be 1.0 (True)
        padding_flags = slice_meta[:, 0]
        tf.debugging.assert_equal(
            padding_flags,
            tf.ones([series_depth], dtype=tf.float32),
            message="All slices in an empty series must have is_padding=1.0"
        )

        # Check if the target description is correctly propagated
        tf.debugging.assert_equal(
            series_meta[2],
            target_series_desc,
            message="The series metadata must contain the requested target_desc."
        )

        # Check if sampling_flag is 0 (No real sampling performed)
        tf.debugging.assert_equal(
            series_meta[0],
            0,
            message="Sampling flag must be 0 for empty series."
        )

    def test_rgb_channel_configuration(self, mock_sample_data, mock_config):
        """
        Ensures the function handles the request for 3 channels correctly.
        """
        (
            imgs, meta, nb_images, img_height,
            img_width, nb_channels, mask, target_desc
        ) = mock_sample_data

        channels = 3

        empty_vol, _, _ = process_empty_series(
            target_desc_tensor=tf.constant(5),
            series_depth=nb_images,
            img_height=img_height,
            img_width=img_width,
            nb_channels=channels,
            config=mock_config
        )

        # Check the last dimension specifically
        assert empty_vol.shape[-1] == 3
        assert empty_vol.shape == (nb_images, img_height, img_width, 3)

    def test_target_desc_passthrough(
        self,
        mock_sample_data,
        mock_config
    ):
        """
        Verify that the target anatomical description code is correctly
        propagated to the series metadata output without any modification.
        """
        # 1. Extract dimensions from the mock fixture
        (
            _, _, _, img_height,
            img_width, nb_channels, _, _
        ) = mock_sample_data

        # Define various description codes to ensure robust passthrough
        # including edge cases like zero or negative values.
        codes_to_test = [0, 99, -1]
        series_depth = 10

        for code in codes_to_test:
            # 2. Setup input tensor for the specific anatomical view
            target_tensor = tf.constant(code, dtype=tf.int32)

            # 3. Call the function under test
            # We focus on the third returned element: series_metadata
            _, _, series_meta = process_empty_series(
                target_desc_tensor=target_tensor,
                series_depth=series_depth,
                img_height=img_height,
                img_width=img_width,
                nb_channels=nb_channels,
                config=mock_config
            )

            # 4. Assertions
            # In process_empty_series, target_desc_tensor is stored at index 2
            # of the series_metadata vector.
            returned_code = series_meta[2]

            tf.debugging.assert_equal(
                returned_code,
                target_tensor,
                message=f"Failed to passthrough description code: {code}"
            )


class TestFormatForModel:
    """
    Unit tests for the format_for_model function to ensure proper
    mapping of inputs and targets for multi-series spine models.
    """

    def test_format_output_structure_and_sorting(self, mock_config):
        """
        Verify that the function returns correct shapes, sorts records by
        condition_level_id, and applies correct One-Hot encoding.
        """
        # 1. Setup dimensions from config
        series_depth = mock_config["series_depth"]
        img_shape = mock_config["models"]["backbone_2d"]["img_shape"]
        h, w, c = img_shape
        max_records = mock_config["data_specs"]["max_records_per_frame"]

        # Helper to create volumes filled with -1.0 (MobileNetV2 Black)
        def create_mock_volume():
            img_vol = tf.fill((series_depth, h, w, c), -1.0)
            slice_meta = tf.zeros((series_depth, 4), dtype=tf.float32)
            series_meta = tf.constant([1, 0, 15], dtype=tf.int32)
            return (img_vol, slice_meta, series_meta)

        study_volumes = (create_mock_volume(), create_mock_volume(), create_mock_volume())
        study_id = tf.constant(98765, dtype=tf.int64)

        # 2. Setup unsorted labels to test sorting logic
        # records format: [condition_level_id, severity, x, y]
        labels = {
            "records": tf.constant([
                [5.0, 2.0, 0.5, 0.5],  # ID 5, Severity 2 (One-hot [0,0,1])
                [2.0, 0.0, 0.1, 0.1],  # ID 2, Severity 0 (One-hot [1,0,0])
            ] + [[0.0]*4] * (max_records - 2), dtype=tf.float32)
        }

        # 3. Execute formatting
        features, targets = format_for_model(
            study_volumes_tf=study_volumes,
            study_id_tf=study_id,
            labels=labels,
            config=mock_config
        )

        # 4. Assertions: Feature Dictionary
        # Check image shapes
        assert features["img_sag_t1"].shape == (series_depth, h, w, c)
        # Check study_id shape [1]
        assert features["study_id"].shape == (1,)
        # Verify MobileNetV2 Black normalization (-1.0)
        tf.debugging.assert_equal(features["img_sag_t1"][0, 0, 0, 0], -1.0)

        # 5. Assertions: Target Dictionary (Sorting & One-Hot)
        # IDs were [5, 2]. After ascending sort, index 0 should be ID 2.
        # Condition ID 2 had severity 0.0 -> One-hot [1.0, 0.0, 0.0]
        expected_first_severity = tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)

        tf.debugging.assert_equal(
            targets["severity_output"][0],
            expected_first_severity,
            message="Records were not correctly sorted by condition_level_id."
        )

        # Ensure all targets are cast to float32
        assert targets["severity_output"].dtype == tf.float32
        assert targets["location_output"].dtype == tf.float32
