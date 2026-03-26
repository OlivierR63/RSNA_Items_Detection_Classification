# coding utf-8

import tensorflow as tf
import pytest
import logging
from src.projects.lumbar_spine.RSNA_lumbar_losses_and_metric import (
    compute_rsna_loss_core,
    rsna_weighted_log_loss,
    RSNAKaggleMetric
)


class TestRSNALossesAndMetrics:
    """
    Unit tests for RSNA 2024 specific loss functions and Keras metrics.
    """

    @pytest.fixture
    def mock_logger(self):
        return logging.getLogger("TestLogger")

    def test_compute_rsna_loss_core_perfect_prediction(self):
        """
        Ensures that a perfect prediction results in a near-zero loss.
        """
        # One-hot encoded: Normal (index 0)
        y_true = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.9999, 0.0, 0.0]], dtype=tf.float32)

        loss = compute_rsna_loss_core(y_true, y_pred)

        # Loss should be very small
        tf.debugging.assert_near(loss, [0.0], atol=1e-3)

    def test_rsna_weighted_log_loss_penalty_logic(self):
        """
        Validates that 'Severe' errors are penalized 4x more than 'Normal' errors.
        """
        # Scenario 1: Predicting Normal (1.0, 0, 0) when it's Severe (0, 0, 1.0)
        # Prediction: 90% Normal, 10% Severe
        y_true_severe = tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.0, 0.1]], dtype=tf.float32)

        # The loss for the Severe class (index 2) is multiplied by 4.0
        loss_val = rsna_weighted_log_loss(y_true_severe, y_pred)

        # Expected calculation: -1.0 * log(0.1) * 4.0
        expected_loss = -tf.math.log(0.1) * 4.0
        tf.debugging.assert_near(loss_val, expected_loss, atol=1e-5)

    def test_rsna_kaggle_metric_accumulation(self, mock_logger):
        """
        Tests the Keras metric state management (update_state, result, reset).
        """
        metric = RSNAKaggleMetric(logger=mock_logger)

        # Batch 1
        y_true_1 = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
        y_pred_1 = tf.constant([[0.5, 0.2, 0.3]], dtype=tf.float32)
        metric.update_state(y_true_1, y_pred_1)

        # Batch 2
        y_true_2 = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
        y_pred_2 = tf.constant([[0.8, 0.1, 0.1]], dtype=tf.float32)
        metric.update_state(y_true_2, y_pred_2)

        # Check result (should be mean of the two batches)
        loss1 = compute_rsna_loss_core(y_true_1, y_pred_1)
        loss2 = compute_rsna_loss_core(y_true_2, y_pred_2)
        expected_result = (tf.reduce_mean(loss1) + tf.reduce_mean(loss2)) / 2.0

        tf.debugging.assert_near(metric.result(), expected_result, atol=1e-5)

        # Reset state
        metric.reset_state()
        tf.debugging.assert_equal(metric.total_loss, 0.0)
        tf.debugging.assert_equal(metric.count, 0.0)
