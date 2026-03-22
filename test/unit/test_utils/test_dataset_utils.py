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
        mock_normalize.side_effect = lambda image, mini, maxi: image

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
        args, kwargs = mock_norm.call_args
        assert args[1].numpy() == 0
        assert args[2].numpy() == 1000


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
                'process_single_description'
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
                    'process_single_description'
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
                    'process_single_description'
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
                    'process_single_description'
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

    def test_routing_to_valid_series(self, sample_data):
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
                img_channels
            )

            # IMPORTANT - Force execution by evaluating one of the tensors
            # This triggers tf.cond to actually call the mocked function
            _ = [t.numpy() for t in result] if isinstance(result, tuple) else result.numpy()

            # English comment: Assert that the 'True' branch was taken
            mock_valid.assert_called_once()
            mock_empty.assert_not_called()

    def test_routing_to_empty_series(self, sample_data):
        """
        Tests that the function routes to process_empty_series when no match is found.
        """

        images, meta, series_depth, img_height, img_width, img_channels = sample_data
        target_code = tf.constant(99, dtype=tf.int32)  # No match in description [0, 1, 0, 2]

        # Mocking sub-functions to verify routing
        with ExitStack() as stack:

            # Patch the functions inside the module where process_single_description lives
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
                img_channels
            )

            # IMPORTANT - Force execution by evaluating one of the tensors
            # This triggers tf.cond to actually call the mocked function
            _ = [t.numpy() for t in result] if isinstance(result, tuple) else result.numpy()

            # Assert that the 'False' branch was taken
            mock_empty.assert_called_once()
            mock_valid.assert_not_called()

    def test_type_resilience_casting(self, sample_data):
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
                img_channels
            )

            # IMPORTANT - Force execution by evaluating one of the tensors
            # This triggers tf.cond to actually call the mocked function
            _ = [t.numpy() for t in result] if isinstance(result, tuple) else result.numpy()

            # If casting fails, tf.equal would raise an error before this
            assert mock_valid.called

    def test_graph_compatibility_smoke(self, sample_data):
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
                img_channels
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

        # Mask to keep all 15 images
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
            imgs, meta, nb_images, img_height,
            img_width, nb_channels, mask, target_desc
        ) = mock_sample_data

        target_channels = 1

        # For series 20, instance_numbers are [9, 8, ..., 0] (reversed order)
        # After sorting, the first frame in padded_vol should be the one that was at index 14
        padded_vol, _, _ = process_valid_series(
            imgs,
            meta,
            mask,
            target_desc,
            nb_images,
            img_height,
            img_width,
            target_channels
        )

        # The last image of the stack (idx 14) had instance_number 0
        # After sorting, instance 0 should be at the beginning of the volume.
        expected_first_frame = imgs[14]
        tf.debugging.assert_equal(
            tf.cast(padded_vol[0], tf.int32),
            expected_first_frame
        )

    def test_symmetric_padding(self, mock_sample_data):
        """
        Tests if padding is applied equally before and after
        when nb frames per series < max_depth
        """
        (
            imgs, meta, _, img_height,
            img_width, nb_channels, mask, target_desc
        ) = mock_sample_data

        num_frames = 4
        max_depth = 10  # 6 frames to pad -> 3 before, 3 after

        imgs = tf.random.uniform((num_frames, img_height, img_width, 1))
        meta = {
            'series_id': tf.constant(
                [1]*num_frames, dtype=tf.int64
            ),

            'instance_number': tf.constant(
                list(range(num_frames)), dtype=tf.int32
            ),

            'is_padding': tf.constant(
                [0 if idx % 3 == 0 else 1 for idx in range(num_frames)],
                dtype=tf.int32
            ),

            'scaling_ratio': tf.random.uniform(
                shape=(num_frames,),
                seed=42,
                dtype=tf.float32
            ),

            'x_crop': tf.random.uniform(
                shape=(num_frames,),
                minval=0,
                maxval=256,
                seed=42,
                dtype=tf.int32
            ),

            'y_crop': tf.random.uniform(
                shape=(num_frames,),
                minval=0,
                maxval=256,
                seed=42,
                dtype=tf.int32
            ),
        }
        mask = tf.ones((num_frames,), dtype=tf.bool)

        padded_vol, _, _ = process_valid_series(
            all_images=imgs,
            all_meta=meta,
            desc_mask=mask,
            target_desc_tensor=tf.constant(1),
            series_depth=max_depth,
            height=img_height,
            width=img_width,
            nb_channels=nb_channels,
            is_training=True
        )

        # Check that frames 0, 1, 2 are zero (padding before)
        assert tf.reduce_sum(padded_vol[0:3]) == 0

        # Check that frames 3 to 6 are the "real" frames
        for idx in range(3, 7):
            assert tf.reduce_sum(padded_vol[idx]) > 0

        # Check that frames 7, 8, 9 are zero (padding after)
        assert tf.reduce_sum(padded_vol[7:10]) == 0

    def test_grayscale_to_rgb_conversion(self, mock_sample_data):
        """
        Validates channel expansion from 1 to 3.
        """
        (
            imgs, meta, _, img_height,
            img_width, nb_channels, mask, target_desc
        ) = mock_sample_data

        padded_vol, _, _ = process_valid_series(
            imgs,
            meta,
            mask,
            target_desc,
            TEST_DEPTH,
            TEST_HEIGHT,
            TEST_WIDTH,
            3
        )

        # English comment: Shape must be (Depth, Height, Width, 3)
        assert padded_vol.shape == (TEST_DEPTH, TEST_HEIGHT, TEST_WIDTH, 3)


class TestProcessEmptySeries:

    def test_output_shapes_and_values(self):
        """
        Checks if the function returns the correct shapes and sentinel values.
        """
        # Create dummy input with specific dtype
        all_images = tf.zeros((1, 512, 512, 1), dtype=tf.float32)
        target_code = tf.constant(5, dtype=tf.int32)
        channels = 1

        empty_vol, default_id, target_desc = process_empty_series(
            all_images,
            target_code,
            TEST_DEPTH,
            TEST_HEIGHT,
            TEST_WIDTH,
            channels
        )

        # Assertions for shapes
        assert empty_vol.shape == (TEST_DEPTH, TEST_HEIGHT, TEST_WIDTH, channels)

        # Assertions for values
        tf.debugging.assert_equal(tf.reduce_sum(empty_vol), 0.0)  # Should be all zeros
        assert default_id == -1
        assert target_desc == 5
        assert default_id.dtype == tf.int64

    def test_dtype_propagation(self):
        """
        Verifies that the output volume matches the input images' dtype.
        """

        # Test with float16 to ensure the function adapts
        all_images_f16 = tf.zeros((1, 64, 64, 1), dtype=tf.float16)

        empty_vol, _, _ = process_empty_series(
            all_images_f16,
            tf.constant(1),
            TEST_DEPTH,
            TEST_HEIGHT,
            TEST_WIDTH,
            1
        )

        assert empty_vol.dtype == tf.float16

    def test_rgb_channel_configuration(self):
        """
        Ensures the function handles the request for 3 channels correctly.
        """

        all_images = tf.zeros((1, 64, 64, 1), dtype=tf.float32)
        channels = 3

        empty_vol, _, _ = process_empty_series(
            all_images,
            tf.constant(1),
            TEST_DEPTH,
            TEST_HEIGHT,
            TEST_WIDTH,
            channels
        )

        # Check the last dimension specifically
        assert empty_vol.shape[-1] == 3
        assert empty_vol.shape == (TEST_DEPTH, TEST_HEIGHT, TEST_WIDTH, 3)

    def test_target_desc_passthrough(self):
        """
        Confirms that the target description code is returned unchanged.
        """
        all_images = tf.zeros((1, 8, 8, 1))
        codes_to_test = [0, 99, -1]

        for code in codes_to_test:
            target_tensor = tf.constant(code, dtype=tf.int32)
            _, _, returned_desc = process_empty_series(
                all_images,
                target_tensor,
                TEST_DEPTH,
                TEST_HEIGHT,
                TEST_WIDTH,
                1
            )
            assert returned_desc == code


class TestFormatForModel:

    @pytest.fixture
    def mock_study_data(self):
        """
        Generates a complete set of mock study data.
        """
        # Function to create a volume triplet (images, series_id, desc_id)
        def create_vol_triplet(desc_id):
            img = tf.zeros((TEST_DEPTH, TEST_HEIGHT, TEST_WIDTH, 3))
            series_id = tf.constant(100 + desc_id, dtype=tf.int64)
            description_id = tf.constant(desc_id, dtype=tf.int32)
            return (img, series_id, description_id)

        study_volumes = (
            create_vol_triplet(1),  # Sag T1
            create_vol_triplet(2),  # Sag T2
            create_vol_triplet(3)  # Axial T2
        )

        study_id = tf.constant(12345, dtype=tf.int64)

        records_np = np.zeros((MAX_RECORDS, 4), dtype=np.float32)

        # Generate a unique random permutation of IDs from 0 to 24
        random_ids = np.random.permutation(MAX_RECORDS)

        # Assign the shuffled IDs to the first column
        records_np[:, 0] = random_ids

        # Locate the row where the first element (ID) is 1
        # and update its values to [1, 0, 0.1, 0.1]
        row_idx_01 = np.where(records_np[:, 0] == 1.0)[0][0]
        records_np[row_idx_01] = [1.0, 0.0, 0.1, 0.1]

        # Locate the row where the first element (ID) is 10
        # and update its values to [10, 2, 0.5, 0.5]
        row_idx_10 = np.where(records_np[:, 0] == 10.0)[0][0]
        records_np[row_idx_10] = [10.0, 2.0, 0.5, 0.5]

        labels = {"records": tf.constant(records_np)}

        return study_volumes, study_id, labels

    def test_input_dictionary_keys_and_shapes(self, mock_study_data):
        """
        Tests the formatting logic by patching global dimensions to match
        the fixture's shape [10, 512, 270, 3].
        """
        volumes, study_id, labels = mock_study_data

        # Use ExitStack to manage multiple global variable patches
        with ExitStack() as stack:

            # Patch globals to match the fixture dimensions:
            #  [TEST_DEPTH, TEST_HEIGHT, TEST_WIDTH, 3]
            # This prevents the tf.ensure_shape InvalidArgumentError
            stack.enter_context(patch.object(ds_utils, 'MAX_SERIES_DEPTH', TEST_DEPTH))
            stack.enter_context(patch.object(ds_utils, 'MODEL_2D_HEIGHT', TEST_HEIGHT))
            stack.enter_context(patch.object(ds_utils, 'MODEL_2D_WIDTH', TEST_WIDTH))
            stack.enter_context(patch.object(ds_utils, 'MODEL_2D_NB_CHANNELS', 3))
            stack.enter_context(patch.object(ds_utils, 'MAX_RECORDS', 25))

            # Execute the formatting with the patched environment
            # Note: format_for_model returns (records, features, labels_dict)
            records_out, features, labels_dict = format_for_model(
                volumes,
                study_id,
                labels
            )

            # --- Assertions ---
            # Verify that 'img_sag_t1' exists and has the patched shape
            assert "img_sag_t1" in features
            assert features["img_sag_t1"].shape == (10, 512, 270, 3)

            # Verify that metadata are correctly cast to float32 and reshaped to (1,)
            assert features["series_sag_t1"].dtype == tf.float32
            assert features["series_sag_t1"].shape == (1,)

            # Verify the labels sorting and one-hot encoding
            assert "severity_output" in labels_dict
            assert labels_dict["severity_output"].shape == (25, 3)

    def test_labels_sorting_logic(self, mock_study_data):
        """
        Ensures that records are sorted by condition_id (index 0).
        """
        volumes, study_id, labels = mock_study_data

        with ExitStack() as stack:

            # Patch globals to match the fixture dimensions:
            #  [TEST_DEPTH, TEST_HEIGHT, TEST_WIDTH, 3]
            # This prevents the tf.ensure_shape InvalidArgumentError
            stack.enter_context(patch.object(ds_utils, 'MAX_SERIES_DEPTH', TEST_DEPTH))
            stack.enter_context(patch.object(ds_utils, 'MODEL_2D_HEIGHT', TEST_HEIGHT))
            stack.enter_context(patch.object(ds_utils, 'MODEL_2D_WIDTH', TEST_WIDTH))
            stack.enter_context(patch.object(ds_utils, 'MODEL_2D_NB_CHANNELS', 3))
            stack.enter_context(patch.object(ds_utils, 'MAX_RECORDS', 25))

            # The function returns (records, features, labels_dict)
            records_out, _, _ = format_for_model(volumes, study_id, labels)

            # In our mock, condition_level_id 1 and condition 10 were randomly
            # distributed along the column.
            # After sorting, condition_level_id 1 must be at index 1; condition_level_id_10
            # must be at index 10.
            for idx in range(MAX_RECORDS):
                # English comment: Convert the tensor row to numpy for reliable comparison
                actual_row = records_out[idx, :].numpy()

                if idx == 1:
                    # English comment: Use np.array().astype() or specify dtype in constructor
                    expected = np.array([1, 0, 0.1, 0.1], dtype=np.float32)
                    np.testing.assert_array_almost_equal(actual_row, expected)

                elif idx == 10:
                    expected = np.array([10, 2, 0.5, 0.5], dtype=np.float32)
                    np.testing.assert_array_almost_equal(actual_row, expected)

                else:
                    # English comment: Compare single element by casting idx to float32
                    assert actual_row[0] == np.float32(idx)

            # Verify strictly ascending order
            condition_level_ids = records_out[:, 0].numpy()
            assert np.all(np.diff(condition_level_ids) >= 0)

    def test_severity_one_hot_encoding(self, mock_study_data):
        """
        Checks if severity labels are correctly converted to one-hot (depth=3).
        """
        volumes, study_id, labels = mock_study_data

        with ExitStack() as stack:

            # Patch globals to match the fixture dimensions:
            #  [TEST_DEPTH, TEST_HEIGHT, TEST_WIDTH, 3]
            # This prevents the tf.ensure_shape InvalidArgumentError
            stack.enter_context(patch.object(ds_utils, 'MAX_SERIES_DEPTH', TEST_DEPTH))
            stack.enter_context(patch.object(ds_utils, 'MODEL_2D_HEIGHT', TEST_HEIGHT))
            stack.enter_context(patch.object(ds_utils, 'MODEL_2D_WIDTH', TEST_WIDTH))
            stack.enter_context(patch.object(ds_utils, 'MODEL_2D_NB_CHANNELS', 3))
            stack.enter_context(patch.object(ds_utils, 'MAX_RECORDS', 25))

            _, _, labels_dict = format_for_model(volumes, study_id, labels)

            severity_output = labels_dict["severity_output"]

            # Expected shape (MAX_RECORDS, 3)
            assert severity_output.shape == (MAX_RECORDS, 3)

            # Our condition 1 (now at index 0) had severity 0 -> [1, 0, 0]
            tf.debugging.assert_equal(severity_output[1], [1.0, 0.0, 0.0])

            # Our condition 10 (now at index 1) had severity 2 -> [0, 0, 1]
            tf.debugging.assert_equal(severity_output[10], [0.0, 0.0, 1.0])

    def test_location_output_shape(self, mock_study_data):
        """
        Verifies coordinates output (X, Y).
        """
        volumes, study_id, labels = mock_study_data

        with ExitStack() as stack:

            # Patch globals to match the fixture dimensions:
            #  [TEST_DEPTH, TEST_HEIGHT, TEST_WIDTH, 3]
            # This prevents the tf.ensure_shape InvalidArgumentError
            stack.enter_context(patch.object(ds_utils, 'MAX_SERIES_DEPTH', TEST_DEPTH))
            stack.enter_context(patch.object(ds_utils, 'MODEL_2D_HEIGHT', TEST_HEIGHT))
            stack.enter_context(patch.object(ds_utils, 'MODEL_2D_WIDTH', TEST_WIDTH))
            stack.enter_context(patch.object(ds_utils, 'MODEL_2D_NB_CHANNELS', 3))
            stack.enter_context(patch.object(ds_utils, 'MAX_RECORDS', 25))

            _, _, labels_dict = format_for_model(volumes, study_id, labels)

            location_output = labels_dict["location_output"]

            # Should be (25, 2) for X and Y
            assert location_output.shape == (MAX_RECORDS, 2)

            # Verify specific values for condition_level 1 (moved to index 0)
            tf.debugging.assert_near(location_output[1], [0.1, 0.1])
