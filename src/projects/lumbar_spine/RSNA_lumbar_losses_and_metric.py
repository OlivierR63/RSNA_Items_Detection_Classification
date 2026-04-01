# coding: utf-8

import tensorflow as tf
import logging


def compute_rsna_loss_core(y_true, y_pred):
    """
    Core logic for RSNA weighted log loss.
    Assumes y_pred is already a probability distribution (after softmax).
    """
    class_weights = tf.constant([1.0, 2.0, 4.0], dtype=tf.float32)

    # If y_true is not one-hot (e.g., shape is [batch, 25]), convert it:
    if len(y_true.shape) != len(y_pred.shape):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)

    # Ensure y_true is float32 for mathematical operations
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Standard epsilon for stability
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    # Compute weighted log loss
    # Shape of y_true * log(y_pred) is (batch, 25, 3)
    individual_losses = -y_true * tf.math.log(y_pred)

    # Apply weights across the last dimension (the 3 classes)
    weighted_losses = tf.reduce_sum(individual_losses * class_weights, axis=-1)

    # Return the sum of weights for all 25 conditions
    return weighted_losses


# --- 1. The Loss Functions (for Gradients) ---
def rsna_weighted_log_loss(y_true, y_pred):
    """
    Calculates the Weighted Hierarchical Log Loss for RSNA 2024.

    This loss function applies specific weights to each class (Normal: 1,
    Moderate: 2, Severe: 4) and handles the multi-level (25 conditions)
    output structure of the model.

    Args:
        y_true (tf.Tensor): Ground truth labels.
            Expected shape: (batch_size, 25, 1) or (batch_size, 25).
        y_pred (tf.Tensor): Model predictions (probabilities).
            Expected shape: (batch_size, 25, 3).

    Returns:
        tf.Tensor: A scalar tensor representing the mean weighted log loss
            for the current batch.
    """

    # Ensure y_true is float32 for mathematical operations
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Use the core logic and return mean for the batch
    weighted_loss = compute_rsna_loss_core(y_true, y_pred)

    return tf.reduce_mean(weighted_loss, axis=-1)


# --- 2. The Metric Class (for Monitoring) ---
class RSNAKaggleMetric(tf.keras.metrics.Metric):
    """
    Keras metric to track the official RSNA 2024 competition score.

    This metric accumulates the weighted log loss across all batches in an epoch
    to provide a representative global score, matching the Kaggle leaderboard.

    Attributes:
        total_loss (tf.Variable): Accumulated loss across batches.
        count (tf.Variable): Number of batches processed.
    """

    def __init__(
        self,
        logger: logging.Logger,
        name: str = 'rsna_main_score',
        **kwargs
    ):
        """
        Initializes the metric's state variables.
        """
        self._logger = logger

        self._logger.debug(f"Starting initializing RSNAKaggleMetric ; **kwargs = {kwargs}")

        try:
            super(RSNAKaggleMetric, self).__init__(name=name, **kwargs)

            # Force explicit tf.float32 type for accumulators
            self.total_loss = self.add_weight(
                name='total_loss',
                initializer=tf.zeros_initializer(),
                dtype=tf.float32
            )

            self.count = self.add_weight(
                name='count',
                initializer=tf.zeros_initializer(),
                dtype=tf.float32
            )
            self._logger.debug("RSNAKaggleMetric initialized")

        except Exception as e:
            error_msg = f"RSNAKaggleMetric initialization failed: {e}"
            self._logger.error(error_msg, exc_info=True)
            raise

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the metric state at each batch.

        Args:
            y_true (tf.Tensor): Ground truth labels from the dataset.
            y_pred (tf.Tensor): Predictions from the model.
            sample_weight (optional): Not used here but required by Keras API.
        """
        self._logger.debug("Starting method RSNAKaggleMetric.update_state")

        # Ensure y_true is float32 for mathematical operations
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Reuse the exact same core logic
        weighted_loss = compute_rsna_loss_core(y_true, y_pred)
        batch_mean = tf.reduce_mean(weighted_loss)

        self.total_loss.assign_add(tf.cast(batch_mean, tf.float32))
        self.count.assign_add(1.0)

        self._logger.debug("Method RSNAKaggleMetric.update_state completed successfully")

    def result(self):
        """
        Computes and returns the final metric value (epoch average).

        Returns:
            tf.Tensor: The average RSNA score across all processed batches.
        """
        self._logger.debug("Starting method RSNAKaggleMetric.result")

        division = tf.math.divide_no_nan(
            tf.cast(self.total_loss, tf.float32),
            tf.cast(self.count, tf.float32)
        )

        self._logger.debug("Method RSNAKaggleMetric.result completed successfully")

        return division

    def reset_state(self):
        """
        Resets the state variables at the end of each epoch.
        """
        self._logger.debug("Starting method RSNAKaggleMetric.reset_state")
        self.total_loss.assign(0.0)
        self.count.assign(0.0)
        self._logger.debug("Method RSNAKaggleMetric.reset_state completed successfully.")
