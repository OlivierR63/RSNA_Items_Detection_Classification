# coding: utf-8

import pytest
import tensorflow as tf
from pathlib import Path
from unittest.mock import MagicMock
from src.core.models.model_factory import ModelFactory


class TestModelFactory:
    """
    Comprehensive test suite for ModelFactory.
    Verifies architecture integrity, weight sharing, and output shapes.
    """

    @pytest.fixture
    def factory(self, mock_config, mock_logger):
        """
        Standard factory instance for general tests.
        """
        return ModelFactory(
            config=mock_config,
            logger=mock_logger,
            nb_output_records=25,
            series_depth=10
        )

    # --- 1. Initialization Tests ---

    def test_init_raises_value_error_on_zero_depth(self, mock_config, mock_logger):
        """
        Ensures the factory catches invalid series_depth early.
        """
        with pytest.raises(ValueError, match="argument 'series_depth' was not properly filled"):
            ModelFactory(
                config=mock_config,
                logger=mock_logger,
                nb_output_records=25,
                series_depth=0
            )

    def test_init_path_resolution(self, factory, mock_config, mock_logger):
        """
        Verifies that the TFRecord directory path is correctly converted to Path object.
        """
        assert isinstance(factory._tfrecord_dir, Path)

    # --- 2. Architecture & Weight Sharing Tests ---

    def test_shared_backbone_identity(self, factory, setup_csv_files):
        """
        Critical: Verifies that only one TimeDistributed backbone instance is created
        and shared across all branches to ensure weight sharing.
        """
        model = factory.build_multi_series_model()

        # In Keras functional API, a shared layer appears only once in model.layers
        td_layers = [
            lay for lay in model.layers
            if isinstance(lay, tf.keras.layers.TimeDistributed)
        ]

        assert_msg = "Backbone should be shared (found multiple TimeDistributed layers)"
        assert len(td_layers) == 1, assert_msg
        assert td_layers[0].name == "shared_2d_backbone"

    def test_aggregator_instantiation(self, mock_config, mock_logger):
        """
        Verifies that the aggregator class is instantiated with correct parameters.
        """

        # 1. Setup the mock aggregator instance
        mock_agg_instance = MagicMock()

        def side_effect_action(x):
            return tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=[1, 2, 3])[:, :64])(x)

        mock_agg_instance.side_effect = side_effect_action

        # 2. Setup the mock class to return our instance
        mock_agg_class = MagicMock(return_value=mock_agg_instance)
        factory = ModelFactory(
            config=mock_config,
            logger=mock_logger,
            nb_output_records=25,
            series_depth=10,
            aggregator3d_class=mock_agg_class
        )

        factory.build_multi_series_model()

        # 3. Verifications
        # Should be called 3 times (Sag T1, Sag T2, Axial T2)
        assert mock_agg_instance.call_count == 3
        _, kwargs = mock_agg_class.call_args
        assert kwargs['series_depth'] == 10
        assert kwargs['logger'] == mock_logger

    # --- 4. Output Shape & Consistency Tests ---
    @pytest.mark.parametrize("nb_records", [1, 25])
    def test_output_shapes(self, mock_config, mock_logger, nb_records):
        """
        Verifies that severity and location heads match the configured number of records.
        """
        factory = ModelFactory(
            config=mock_config,
            logger=mock_logger,
            nb_output_records=nb_records,
            series_depth=5
        )
        model = factory.build_multi_series_model()

        # Check Severity Output: (Batch, nb_records, 3 classes)
        assert model.output_shape['severity_output'] == (None, nb_records, 3)

        # Check Location Output: (Batch, nb_records, 2 coordinates)
        assert model.output_shape['location_output'] == (None, nb_records, 2)

        # Check Study ID Passthrough
        assert model.output_shape['study_id_output'] == (None, 1)

    # --- 4. Functional Insight (Dry Run) ---

    def test_model_forward_pass(self, factory):
        """
        Performs a dummy forward pass to ensure no concatenation or reshape errors
        occur during tensor flow.
        """
        model = factory.build_multi_series_model()

        # Generate dummy data for all inputs
        depth = factory._series_depth
        height, width, channels = factory._config['models']['backbone_2d']['img_shape']

        batch = 1
        dummy_inputs = {
            "study_id": tf.zeros((batch, 1)),
            "img_sag_t1": tf.zeros((batch, depth, height, width, channels)),
            "series_metadata_t1": tf.zeros((batch, 3), dtype='int32'),
            "slice_metadata_t1": tf.zeros((batch, depth, 4)),
            "img_sag_t2": tf.zeros((batch, depth, height, width, channels)),
            "series_metadata_t2": tf.zeros((batch, 3), dtype='int32'),
            "slice_metadata_t2": tf.zeros((batch, depth, 4)),
            "img_axial_t2": tf.zeros((batch, depth, height, width, channels)),
            "series_metadata_axial_t2": tf.zeros((batch, 3), dtype='int32'),
            "slice_metadata_axial_t2": tf.zeros((batch, depth, 4))
        }

        # Predict (Forward pass)
        try:
            outputs = model(dummy_inputs, training=False)
        except Exception as e:
            pytest.fail(f"Model forward pass failed with error: {e}")

        assert "severity_output" in outputs
        assert "location_output" in outputs

        # Ensure activations are correct (Softmax for severity -> sum to 1)
        severity_sums = tf.reduce_sum(outputs['severity_output'], axis=-1)
        tf.debugging.assert_near(severity_sums, tf.ones_like(severity_sums))

    # --- 5. Traceability Tests ---

    def test_study_id_traceability(self, factory):
        """
        Verifies that the study_id_output is exactly the same as the input study_id.
        """
        model = factory.build_multi_series_model()

        test_ids = tf.constant([[101], [202]], dtype=tf.int64)

        # Create a dict with all inputs to satisfy the model
        # We use the mapped names and shapes
        inputs = {}

        for name, tensor in model.input.items():
            if name == "study_id":
                # Inject our specific test Ids
                inputs["study_id"] = test_ids

            else:
                # tensor.shape gives us the symbolic shape (None, 16, 224, 224, 3)
                shape = list(tensor.shape)
                shape[0] = 2  # Set batch size to 2
                inputs[name] = tf.zeros(shape)

        # Forward pass
        outputs = model(inputs, training=False)

        # 3. Check if outputs is also a dict (consistent with input)
        if isinstance(outputs, dict):
            output_results = outputs
        else:
            # Fallback if outputs is a list
            output_results = dict(zip(model.output_names, outputs))

        # Final Verification
        tf.debugging.assert_equal(output_results['study_id_output'], test_ids)
