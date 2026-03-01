# coding utf-8

import tensorflow as tf
import numpy as np
import json
import pytest
import os
import src.core.utils.dataset_utils as ds_utils

from unittest.mock import patch, MagicMock
from contextlib import ExitStack
from typing import Dict, Tuple
from pathlib import Path
from src.config.config_loader import ConfigLoader
from src.core.utils.dataset_utils import (
    fast_parse,
    calculate_max_series_depth,
    parse_tfrecord_single_element,
    normalize_image,
    process_study_multi_series,
    reduce_to_first_element,
    process_single_description,
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


def test_fast_parse():
    """
    Test the fast_parse function to ensure it correctly extracts and joins 
    study_id and series_id from a serialized TFRecord example.
    """
    # 1. Setup mock data
    study_id = 12345
    series_id = 67890
    expected_result = b"12345_67890"  # TensorFlow strings are byte strings

    # 2. Create a serialized tf.train.Example
    # Construct the example with the expected feature keys
    feature = {
        'study_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[study_id])),
        'series_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[series_id]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized_example = example_proto.SerializeToString()

    # 3. Call the function under test
    # Execute parsing on the serialized byte string
    result = fast_parse(serialized_example)

    # 4. Assertions
    # Verify the output matches the joined string format
    assert isinstance(result, tf.Tensor)
    assert result.numpy() == expected_result


@pytest.mark.parametrize("study_id, series_id, expected", [
    (111, 222, b"111_222"),
    (0, 0, b"0_0"),
    (999999, 1, b"999999_1"),
])
def test_fast_parse_parameterized(study_id, series_id, expected):
    """
    Ensure fast_parse handles various ID values correctly using parameterization.
    """
    feature = {
        'study_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[study_id])),
        'series_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[series_id]))
    }
    serialized = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
    
    result = fast_parse(serialized)
    assert result.numpy() == expected



class TestCalculateMaxSeriesDepth:

    @pytest.fixture(autouse=True)
    def setup_env(self, tmp_path):
        """
        Setup temporary environment for each test.
        """
        self.tmp_dir = tmp_path / "tfrecords"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.tmp_dir / "depth_metadata_cache.json"
        
        # English comment: Patch the global variable at the module level
        with patch("src.core.utils.dataset_utils.TFRECORD_DIR", str(self.tmp_dir)):
            yield

    def test_return_zero_if_no_files(self):
        """
        Covers 'if not tfrecord_files: return 0'
        """
        assert calculate_max_series_depth() == 0

    def test_cache_hit_success(self):
        """
        Covers valid cache branch.
        """
        # Create a dummy file to have file_count = 1
        (self.tmp_dir / "test.tfrecord").write_bytes(b"")
        
        # Create a relevant cache file
        cache_data = {'file_count': 1, 'max_depth': 99}
        self.cache_file.write_text(json.dumps(cache_data))
        
        assert calculate_max_series_depth() == 99

    def test_cache_invalidation_on_file_count_mismatch(self):
        """
        Covers cache invalidation if file count changes.
        """
        # We generate 2 dummy record files now
        (self.tmp_dir / "f1.tfrecord").write_bytes(b"")
        (self.tmp_dir / "f2.tfrecord").write_bytes(b"")
        
        # Cache recorded only 1 file
        cache_data = {'file_count': 1, 'max_depth': 99}
        self.cache_file.write_text(json.dumps(cache_data))
        
        # We mock the TF pipeline to return a new value (e.g., 5)
        with patch('tensorflow.data.TFRecordDataset') as mock_ds:
            mock_ds.return_value.map.return_value.as_numpy_iterator.return_value = [b"A"]*5
            assert calculate_max_series_depth() == 5

    def test_corrupted_cache_fallback_to_calculation(self):
        """
        Covers the 'except Exception: pass' block.
        """
        (self.tmp_dir / "f1.tfrecord").write_bytes(b"")
        self.cache_file.write_text("NOT_JSON") # Corrupt file
        
        with patch('tensorflow.data.TFRecordDataset') as mock_ds:
            mock_ds.return_value.map.return_value.as_numpy_iterator.return_value = [b"B"]*10
            assert calculate_max_series_depth() == 10

    def test_full_calculation_and_saving_cache(self):
        """
        Covers the aggregation logic and JSON saving.
        """
        (self.tmp_dir / "data.tfrecord").write_bytes(b"")
        
        # Mocking 2 series: 'S1' with 3 slices, 'S2' with 7 slices
        mock_data = [b"S1"]*3 + [b"S2"]*7
        
        with patch('tensorflow.data.TFRecordDataset') as mock_ds:
            mock_ds.return_value.map.return_value.as_numpy_iterator.return_value = mock_data
            
            result = calculate_max_series_depth()
            
            assert result == 7
            # Verify cache was saved
            with open(self.cache_file, 'r') as f:
                saved = json.load(f)
                assert saved['max_depth'] == 7
                assert saved['file_count'] == 1

    
@pytest.fixture()
def mock_globals():
    """
    Setup global variables required by the parsing function.
    """
    with patch('src.core.utils.dataset_utils.MODEL_2D_HEIGHT', 256), \
         patch('src.core.utils.dataset_utils.MODEL_2D_WIDTH', 256), \
         patch('src.core.utils.dataset_utils.MAX_RECORDS', 25):
        yield

def create_mock_tfrecord_example():
    """
    Helper to create a serialized tf.train.Example.
    """
    # Create a dummy 10x10 image as uint16
    img_array = np.ones((10, 10), dtype=np.uint16) * 500
    img_raw = img_array.tobytes()

    # Create dummy records (25 records * 4 values = 100 floats)
    # Record format: [condition_level, severity, x, y]
    # We set one record at [1, 2, 5, 5] (x=5, y=5 on a 10x10 image)
    records = np.zeros(100, dtype=np.float32)
    records[0:4] = [1.0, 2.0, 5.0, 5.0] 

    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'study_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[123])),
        'series_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[456])),
        'series_min': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
        'series_max': tf.train.Feature(int64_list=tf.train.Int64List(value=[1000])),
        'instance_number': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
        'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[10])),
        'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[10])),
        'description': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
        'records': tf.train.Feature(float_list=tf.train.FloatList(value=records)),
        'nb_records': tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


@pytest.fixture
def mock_env_dims():
    # ExitStack allows managing multiple context managers cleanly
    with ExitStack() as stack:
        # Enter each patch context and add it to the stack
        stack.enter_context(patch('src.core.utils.dataset_utils.MODEL_2D_WIDTH', TEST_WIDTH))
        stack.enter_context(patch('src.core.utils.dataset_utils.MODEL_2D_HEIGHT', TEST_HEIGHT))
        mocked_normalize = stack.enter_context(patch('src.core.utils.dataset_utils.normalize_image'))
        
        # All patches are active during the yield
        yield mocked_normalize

    # All patches are automatically released here when exiting the 'with' block

def test_parse_tfrecord_single_element(mock_env_dims):
    """
    Test the full parsing logic including resizing, 
    coordinate normalization, and dictionary structure.
    """
    mock_normalize = mock_env_dims

    # 1. Setup
    serialized_example = create_mock_tfrecord_example()
    
    # Mock normalize_image to return the input as float32
    mock_normalize.side_effect = lambda image, mini, maxi: image 

    # 2. Execution
    image, metadata, labels = parse_tfrecord_single_element(tf.constant(serialized_example))

    # 3. Assertions: Image
    # Check resizing (270, 512) and type (float16)
    assert image.shape == (TEST_HEIGHT, TEST_WIDTH, 1)
    assert image.dtype == tf.float16

    # 4. Assertions: Metadata
    assert metadata['study_id'].numpy() == 123
    assert metadata['series_id'].numpy() == 456
    assert metadata['instance_number'].numpy() == 1

    # 5. Assertions: Labels (Coordinate Normalization)
    # English comment: x=5 on width=10 should be 0.5 normalized
    records = labels['records'].numpy()
    assert records.shape == (25, 4)
    assert records[0, 0] == 1.0 # Condition
    assert records[0, 1] == 2.0 # Severity
    assert records[0, 2] == 0.5 # x_norm (5/10)
    assert records[0, 3] == 0.5 # y_norm (5/10)


def test_normalization_call_args():
    """
    Specifically verify that normalize_image is called with 
    correct series_min and series_max values.
    """
    serialized_example = create_mock_tfrecord_example()
    
    with patch('src.core.utils.dataset_utils.normalize_image') as mock_norm:
        mock_norm.return_value = tf.zeros((256, 256, 1), dtype=tf.float32)
        
        parse_tfrecord_single_element(tf.constant(serialized_example))
        
        # English comment: Check if series_min=0 and series_max=1000 were passed
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

def test_normalize_image_standard_range(mock_scaling_bounds):
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
    normalized = normalize_image(image, s_min, s_max)

    # 3. Verification
    # 0.0 -> becomes -1.0
    # 500.0 -> becomes 0.0
    # 1000.0 -> becomes 1.0
    expected = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(normalized.numpy(), expected, atol=1e-5)


def test_normalize_image_division_by_zero(mock_scaling_bounds):
    """
    Verify that a zero-range series (max == min) 
    does not cause a crash thanks to epsilon.
    """
    # Image values are all 500, min and max are also 500.
    image = tf.constant([500.0, 500.0], dtype=tf.float32)
    s_min = tf.constant(500, dtype=tf.int64)
    s_max = tf.constant(500, dtype=tf.int64)

    # Execution should not raise ZeroDivisionError
    normalized = normalize_image(image, s_min, s_max)
    
    # Verification: (500-500)/epsilon = 0. 
    # 0 * (1 - (-1)) + (-1) = -1.0
    # Verify that each pixel in the image is equal to -1.0 
    assert np.all(normalized.numpy() == -1.0)


def test_normalize_image_out_of_bounds(mock_scaling_bounds):
    """
    Check behavior when image pixels are outside 
    the provided series min/max.
    """
    image = tf.constant([-100.0, 1100.0], dtype=tf.float32)
    s_min = tf.constant(0, dtype=tf.int64)
    s_max = tf.constant(1000, dtype=tf.int64)

    normalized = normalize_image(image, s_min, s_max)

    # -100 becomes -1.2 (outside target range)
    # 1100 becomes 1.2
    assert normalized.numpy()[0] < -1.0
    assert normalized.numpy()[1] > 1.0


@pytest.fixture
def mock_study_env():
    with ExitStack() as stack:
        # English comment: Patching module-level constants using patch.object
        stack.enter_context(patch.object(ds_utils, 'MODEL_2D_WIDTH', TEST_WIDTH))
        stack.enter_context(patch.object(ds_utils, 'MODEL_2D_HEIGHT', TEST_HEIGHT))
        stack.enter_context(patch.object(ds_utils, 'MAX_SERIES_DEPTH', TEST_DEPTH))
        stack.enter_context(patch.object(ds_utils, 'MODEL_2D_NB_CHANNELS', TEST_CHANNELS))
        
        mock_proc_single_desc = stack.enter_context(patch.object(ds_utils, 'process_single_description'))
        mock_reduce = stack.enter_context(patch.object(ds_utils, 'reduce_to_first_element'))
        
        yield mock_proc_single_desc, mock_reduce


class TestProcessStudyMultiSeries:

    def test_nominal_case(self, mock_study_env):
        """
        Tests the standard orchestration with all three series present.
        """
        mock_proc_single_desc, mock_reduce = mock_study_env
        
        # English comment: Using constants for tensor shapes to ensure consistency
        standard_shape = (TEST_DEPTH, TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)
        
        mock_proc_single_desc.side_effect = [
            (tf.ones(standard_shape), tf.constant(101), tf.constant(0)),
            (tf.ones(standard_shape), tf.constant(102), tf.constant(1)),
            (tf.ones(standard_shape), tf.constant(103), tf.constant(2))
        ]
        mock_reduce.side_effect = lambda x: x[0]

        meta = {'study_id': tf.constant([999, 999], dtype=tf.int64)}

        # English comment: Batch size of 2 for input simulation
        images = tf.zeros((2, TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS))
        labels = {'label_a': tf.constant([1, 1])}

        triplet, study_id, reduced = ds_utils.process_study_multi_series(images, meta, labels)

        assert len(triplet) == 3
        assert study_id.numpy() == 999
        assert mock_proc_single_desc.call_count == 3

    def test_missing_series_resilience(self, mock_study_env):
        """
        Tests if the function handles a missing series (e.g., Axial T2 missing).
        """
        mock_proc_single_desc, _ = mock_study_env

        standard_shape = (TEST_DEPTH, TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)
        
        mock_proc_single_desc.side_effect = [
            (tf.ones(standard_shape), tf.constant(101), tf.constant(0)),
            (tf.ones(standard_shape), tf.constant(102), tf.constant(1)),
            (tf.zeros(standard_shape), tf.constant(-1), tf.constant(2)) # Missing 'Axial T2'
        ]

        # Input number of frames can differ from TEST_DEPTH
        num_input_frames = 1 
        input_images = tf.zeros((num_input_frames, TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS))
        triplet, _, _ = ds_utils.process_study_multi_series(
            input_images, 
            {'study_id': tf.constant([1], dtype=tf.int64)}, 
            {}
        )

        # Verify output depth matches TEST_DEPTH even if input was different
        for idx in range(3):
            assert triplet[0][0].shape[0] == TEST_DEPTH, f"Depth mismatch at index {idx}"
        
        assert tf.reduce_sum(triplet[2][0]).numpy() == 0, "'Axial T2' series should be empty (padded)"
        assert triplet[0][1] == 101, "Valid series ID should be preserved"
        assert triplet[2][1].numpy() == -1, "Missing series ID should be -1"

    def test_tf_graph_compatibility(self, mock_study_env):
        """
        Verifies the function can be traced by tf.function (no Python-side leaks).
        """

        mock_proc_single_desc, mock_reduce = mock_study_env
        
        # Define expected shapes for the graph to validate
        standard_shape = (TEST_DEPTH, TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)
        mock_proc_single_desc.return_value = (tf.zeros(standard_shape), tf.constant(0), tf.constant(0))
        mock_reduce.side_effect = lambda x: x[0]

        # Wrap the target function in a static graph
        @tf.function
        def wrapped_call(imgs, m, l):
            return ds_utils.process_study_multi_series(imgs, m, l)

        # Prepare minimal valid inputs
        meta = {'study_id': tf.constant([1], dtype=tf.int64)}
        images = tf.zeros((1, TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS))
        labels = {'target': tf.constant([1])}

        # Execution
        triplet, study_id, reduced_labels = wrapped_call(images, meta, labels)

        # Explicit assertions to confirm graph output integrity
        assert isinstance(study_id, tf.Tensor), "Output study_id must be a Tensor"
        assert study_id.shape == (), "Study ID should be a scalar tensor"
        
        # Validate that the triplet maintains its shape through the graph
        assert len(triplet) == 3
        assert triplet[0][0].shape == standard_shape, f"Expected shape {standard_shape}, got {triplet[0][0].shape}"
        
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
    def sample_data(self):
        # English comment: Setup basic tensors for testing filtering logic
        nb_images = 4
        images = tf.zeros((nb_images, TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS))
        meta = {
            'description': tf.constant([0, 1, 0, 2], dtype=tf.int32),
            'series_id': tf.constant([10, 11, 10, 12], dtype=tf.int64),
            'instance_number': tf.constant([1, 1, 2, 1], dtype=tf.int32)
        }
        return images, meta

    def test_routing_to_valid_series(self, sample_data):
        """
        Tests that the function routes to process_valid_series when a match is found.
        """

        images, meta = sample_data
        target_code = tf.constant(1, dtype=tf.int32) # Matches second element

        # Mocking sub-functions to verify routing
        with ExitStack() as stack:
            # Enter multiple contexts without backslashes
            mock_valid = stack.enter_context(patch.object(ds_utils, 'process_valid_series'))
            mock_empty = stack.enter_context(patch.object(ds_utils, 'process_empty_series'))
            
            mock_valid.return_value = (
                tf.zeros(
                    (
                        TEST_DEPTH,
                        TEST_HEIGHT,
                        TEST_WIDTH,
                        TEST_CHANNELS
                     )
                ), 
                tf.constant(11, dtype=tf.int64), 
                target_code
            )

            result = process_single_description(
                images,
                meta,
                target_code,
                TEST_DEPTH,
                TEST_HEIGHT,
                TEST_WIDTH,
                TEST_CHANNELS
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

        images, meta = sample_data
        target_code = tf.constant(99, dtype=tf.int32) # No match in description [0, 1, 0, 2]

        # Mocking sub-functions to verify routing
        with ExitStack() as stack:

            # Patch the functions inside the module where process_single_description lives
            # Replace 'src.core.utils.dataset_utils' with the actual import path if different

            # Enter multiple contexts without backslashes
            mock_valid = stack.enter_context(patch.object(ds_utils, 'process_valid_series'))
            mock_empty = stack.enter_context(patch.object(ds_utils, 'process_empty_series'))
            
            mock_empty.return_value = (
                tf.zeros((TEST_DEPTH, TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)), 
                tf.constant(-1, dtype=tf.int64), 
                target_code
            )

            result = process_single_description(
                images,
                meta,
                target_code,
                TEST_DEPTH,
                TEST_HEIGHT,
                TEST_WIDTH,
                TEST_CHANNELS
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

        images, meta = sample_data

        # Target is int32, but meta['description'] will be cast to int64
        target_code = tf.constant(2, dtype=tf.int32)

        # Mocking sub-functions to verify routing
        with ExitStack() as stack:
            # Enter multiple contexts without backslashes
            mock_valid = stack.enter_context(patch.object(ds_utils, 'process_valid_series'))
            mock_empty = stack.enter_context(patch.object(ds_utils, 'process_empty_series'))
            
            mock_valid.return_value = (
                tf.zeros((TEST_DEPTH, TEST_HEIGHT, TEST_WIDTH, TEST_CHANNELS)), 
                tf.constant(12, dtype=tf.int64), 
                target_code
            )

            result = process_single_description(
                images,
                meta,
                target_code,
                TEST_DEPTH,
                TEST_HEIGHT,
                TEST_WIDTH,
                TEST_CHANNELS
            )
            
            # IMPORTANT - Force execution by evaluating one of the tensors
            # This triggers tf.cond to actually call the mocked function
            _ = [t.numpy() for t in result] if isinstance(result, tuple) else result.numpy()

            # English comment: If casting fails, tf.equal would raise an error before this
            assert mock_valid.called

    def test_graph_compatibility_smoke(self, sample_data):
        """
        Checks if the conditional logic is traceable by tf.function.
        """

        images, meta = sample_data
        target_code = tf.constant(0, dtype=tf.int32)

        @tf.function
        def graph_fn(imgs, mt, tc):
            return process_single_description(
                images,
                meta,
                target_code,
                TEST_DEPTH,
                TEST_HEIGHT,
                TEST_WIDTH,
                TEST_CHANNELS
            )

        # We don't mock sub-functions here to ensure the full graph is valid
        # Assuming process_valid/empty_series are also graph-compatible
        try:
            graph_fn(images, meta, target_code)
        except Exception as e:
            pytest.fail(f"tf.function tracing failed: {e}")


class TestProcessValidSeries:

    @pytest.fixture
    def mock_inputs(self):
        # Create 15 mock images (N=15)
        nb_images = 15
        nb_channels = 1
        all_images = tf.random.uniform((nb_images, TEST_HEIGHT, TEST_WIDTH, nb_channels))
        all_meta = {
            # 15 images total, we'll mask some later
            'series_id': tf.constant([10]*5 + [20]*10, dtype=tf.int64),
            'instance_number': tf.constant(list(range(5)) + list(reversed(range(10))), dtype=tf.int32)
        }
        # Mask to keep all 15 images
        desc_mask = tf.ones((15,), dtype=tf.bool)
        target_desc = tf.constant(1, dtype=tf.int32)
        
        return all_images, all_meta, desc_mask, target_desc

    def test_selects_series_with_most_frames(self, mock_inputs):
        """
        Checks if the function correctly picks the series with the highest count.
        """
        imgs, meta, mask, target_desc = mock_inputs
        target_channels = 1
        
        # Series 20 has 10 frames, Series 10 has only 5.
        padded_vol, best_id, _ = process_valid_series(
            imgs,
            meta,
            mask,
            target_desc,
            TEST_DEPTH,
            TEST_HEIGHT,
            TEST_WIDTH,
            target_channels
        )
        
        assert best_id == 20
        assert padded_vol.shape[0] == TEST_DEPTH

    def test_anatomical_sorting(self, mock_inputs):
        """
        Verifies that frames are sorted by instance_number.
        """
        imgs, meta, mask, target_desc = mock_inputs
        target_channels = 1
        
        # For series 20, instance_numbers are [9, 8, ..., 0] (reversed order)
        # After sorting, the first frame in padded_vol should be the one that was at index 14
        padded_vol, _, _ = process_valid_series(
            imgs,
            meta,
            mask,
            target_desc,
            TEST_DEPTH,
            TEST_HEIGHT,
            TEST_WIDTH,
            target_channels
        )
        
        # The last image of the stack (idx 14) had instance_number 0
        # After sorting, instance 0 should be at the beginning of the volume.
        expected_first_frame = imgs[14]
        tf.debugging.assert_near(padded_vol[0], expected_first_frame)

    def test_symmetric_padding(self):
        """
        Tests if padding is applied equally before and after when frames < max_depth.
        """
        num_frames = 4
        max_depth = 10 # 6 frames to pad -> 3 before, 3 after
        
        imgs = tf.random.uniform((num_frames, TEST_HEIGHT, TEST_WIDTH, 1))
        meta = {
            'series_id': tf.constant([1]*num_frames, dtype=tf.int64),
            'instance_number': tf.constant(list(range(num_frames)), dtype=tf.int32)
        }
        mask = tf.ones((num_frames,), dtype=tf.bool)
        
        padded_vol, _, _ = process_valid_series(
            imgs,
            meta,
            mask,
            tf.constant(1),
            max_depth,
            TEST_HEIGHT,
            TEST_WIDTH,
            1
        )
        
        # Check that frames 0, 1, 2 are zero (padding before)
        assert tf.reduce_sum(padded_vol[0:3]) == 0

        # Check that frames 3 to 6 are the "real" frames
        for idx in range(3,7):
            assert tf.reduce_sum(padded_vol[idx]) > 0

        # Check that frames 7, 8, 9 are zero (padding after)
        assert tf.reduce_sum(padded_vol[7:10]) == 0

    def test_grayscale_to_rgb_conversion(self, mock_inputs):
        """
        Validates channel expansion from 1 to 3.
        """
        imgs, meta, mask, target_desc = mock_inputs
        
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
        tf.debugging.assert_equal(tf.reduce_sum(empty_vol), 0.0) # Should be all zeros
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
            create_vol_triplet(1), # Sag T1
            create_vol_triplet(2), # Sag T2
            create_vol_triplet(3)  # Axial T2
        )
        
        study_id = tf.constant(12345, dtype=tf.int64)

        records_np = np.zeros((MAX_RECORDS, 4), dtype = np.float32)
        
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
        
            # In our mock, condition_level_id 1 and condition 10 were randomly distributed along the column.
            # After sorting, condition_level_id 1 must be at index 1; condition_level_id_10 must be at index 10.
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

