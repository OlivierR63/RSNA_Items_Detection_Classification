# coding: utf-8

import tensorflow as tf
import tf_keras
import logging
from src.core.utils.logger import get_current_logger
from src.config.config_loader import ConfigLoader
from src.core.utils.dataframe_class_count import DataFrameClassCount
from src.projects.lumbar_spine.csv_metadata_handler import CSVMetadataHandler


def get_class_weights():
    logger = get_current_logger()
    critical_msg = "Wrong class weight setting in the YAML configuration file: %s"
    try:
        class_weights_cfg = ConfigLoader().get_value('compilation').get('class_weights')

        severity_dic = {}
        for label, value in class_weights_cfg.items():
            label = label.lower().replace(" ", "")
            severity_dic[label] = value

        # Load severity label mapper
        raw_mapper = CSVMetadataHandler().get_raw_mapper()
        severity_mapper = raw_mapper['severity'].reverse_mapping

        weights_list = []
        for key in severity_mapper.keys():
            idx = int(key)
            weights_list.append(severity_dic[severity_mapper[idx]])

        class_weights = tf.constant(
            weights_list,
            dtype=tf.float32
        )
        return class_weights

    except Exception as e:
        logger.critical(
            critical_msg,
            e,
            exc_info=True,
            extra={'status': 'failed'}
        )
        tf.print(critical_msg, e)
        raise


def compute_rsna_loss_core(y_true, y_pred, class_weights, balancing_weights):
    """
   Executes the core mathematical logic for the RSNA 2024 weighted log loss.

    This function computes the weighted cross-entropy for a multi-task classification
    problem involving 25 distinct spinal conditions. It applies two levels of
    weighting: competition-defined class weights (Normal, Moderate, Severe)
    and dataset-specific balancing weights.

    Mathematical Operations:
        1. Format Alignment: Converts integer labels to one-hot encoding if needed.
        2. Numerical Stability: Clips predictions to prevent log(0) resulting in NaN.
        3. Weighted Log Loss: Computes -y_true * log(y_pred) * class_weights * balancing_weights.
        4. Spatial Reduction: Sums the weighted components across the class dimension.

    Args:
        y_true (tf.Tensor): Ground truth labels.
            Can be provided as class indices (Shape: [batch, 25])
            or one-hot encoded probabilities (Shape: [batch, 25, 3]).
        y_pred (tf.Tensor): Model predictions as probabilities (after softmax).
            Shape: [batch, 25, 3].
        balancing_weights (tf.Tensor): Dataset-specific balancing weights.
            Shape: [25, 3] or broadcastable to [batch, 25, 3].
        class_weights (tf.Tensor): RSNA-defined weights. Shape: (3,)

    Returns:
        tf.Tensor: A tensor of weighted losses for each sample and each condition.
            Shape: [batch, 25].

    Note:
        The final reduction (e.g., tf.reduce_mean) is intentionally omitted here
        to allow this core function to be used by both the Loss (which needs a
        scalar) and potentially by custom metrics or debug tools that might
        require per-condition loss analysis.
    """

    # If y_true is not one-hot (e.g., shape is [batch, 25]), convert it:
    # Also ensure y_true is float32 for mathematical operations
    if len(y_true.shape) != len(y_pred.shape):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=3, dtype=tf.float32)

    y_pred = tf.cast(y_pred, tf.float32)

    # Standard epsilon for numerical stability
    epsilon = tf_keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    # Compute weighted log loss
    # Vectorized calculation
    # Shape of y_true * log(y_pred) is (batch, 25, 3)
    individual_losses = -y_true * tf.math.log(y_pred)

    # Apply weights across the last dimension (the 3 classes)
    # ==> Resulting shape: (batch, 25)
    weighted_losses = tf.reduce_sum(
        individual_losses * class_weights * balancing_weights,
        axis=-1
    )

    # Return the weighted losses per condition
    # for each sample in the batch
    return weighted_losses


def apply_label_smoothing(y_true, smoothing=0.1):
    # Number of classes
    num_classes = tf.cast(tf.shape(y_true)[-1], y_true.dtype)

    # Formula : y_smoothed = y_true * (1 - alpha) + (alpha / num_classes)
    return y_true * (1.0 - smoothing) + (smoothing / num_classes)


class RSNALossAndMetricProvider:
    def __init__(self, logger):
        try:
            self._config = ConfigLoader().get()
            self._logger = logger
            self._balancing_weights = None
            self._class_weights = get_class_weights()
        except Exception:
            raise

    def get_loss(self):
        """
        Dynamically builds and returns the RSNA weighted log loss function.

        This method acts as a factory that injects dataset-specific balancing
        weights into the loss calculation via a closure. This allows the
        returned function to maintain the standard Keras loss signature
        (y_true, y_pred) while accessing external weighting metadata.

        Returns:
            function: A callable 'rsna_weighted_log_loss' configured with
                the provider's balancing weights.
        """
        class_weights = self._class_weights
        balancing_weights = self._get_balancing_weights()

        def rsna_weighted_log_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
            """
            Keras-compatible weighted hierarchical log loss.

            Calculates the mean loss across 25 spinal conditions, applying
            both class-level weights (Normal/Moderate/Severe) and
            sample-level balancing weights.

            Args:
                y_true (tf.Tensor): Ground truth labels.
                    Shape: (batch_size, 25) or (batch_size, 25, 3).
                y_pred (tf.Tensor): Model predictions as probabilities.
                    Shape: (batch_size, 25, 3).

            Returns:
                tf.Tensor: Scalar float32 tensor representing the mean
                    weighted log loss for the batch.
            """

            # Use the core logic and return mean for the whole batch
            weighted_loss = compute_rsna_loss_core(
                y_true,
                y_pred,
                class_weights,
                balancing_weights
            )

            return tf.reduce_mean(weighted_loss)

        return rsna_weighted_log_loss

    def get_metrics(self):
        """
        Return the metric instance using the same weights
        """
        return RSNAKaggleMetric(
            class_weights=self._class_weights,
            balancing_weights=self._get_balancing_weights(),
            logger=self._logger
        )

    def _get_balancing_weights(self):
        """
        Retrieves the dataset-specific balancing weights from the DataFrameClassCount singleton.

        Returns:
            tf.Tensor: A 1D float32 tensor containing the balancing weights for each class.
        """
        try:
            # Access the singleton instance of DataFrameClassCount
            df_class_count = DataFrameClassCount()

            if not df_class_count.is_balancing_weights_valid():
                df_class_count.set_balancing_weights()
                balancing_weights = df_class_count.get_balancing_weights()

            else:
                balancing_weights = df_class_count.get_balancing_weights()

            return balancing_weights

        except Exception as e:
            self._logger.critical(
                "Failed to retrieve balancing weights: %s",
                e,
                exc_info=True,
                extra={'status': 'failed'}
            )
            tf.print("Failed to retrieve balancing weights:", e)
            raise


# --- 2. The Metric Class (for Monitoring) ---
class RSNAKaggleMetric(tf_keras.metrics.Metric):
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
        class_weights: tf.Tensor,
        balancing_weights: tf.Tensor,
        logger: logging.Logger = None,
        name: str = 'rsna_main_score',
        **kwargs
    ):
        """
        Initializes the metric's state variables.
        """
        self._logger = logger or get_current_logger()
        critical_msg = "RSNAKaggleMetric initialization failed: %s"

        try:
            super().__init__(name=name, **kwargs)

            self._config = ConfigLoader().get()
            self._class_weights = class_weights
            self._balancing_weights = balancing_weights

            if balancing_weights is None or tf.convert_to_tensor(balancing_weights).shape[0] == 0:
                error_msg = "Attribute self._balancing_weights must be provided and cannot be None."
                raise ValueError(f"Error in RSNAKaggleMetric.__init__(): {error_msg}")

            self._debug_mode = False
            if self._config and self._config.get('logging', {}).get('level') == "DEBUG":
                self._debug_mode = True

            if self._debug_mode:
                tf.print(f"Starting initializing RSNAKaggleMetric ; **kwargs = {kwargs}")

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
            if self._debug_mode:
                tf.print("RSNAKaggleMetric initialized")

        except Exception as e:
            # Normally, the logger object is not compliant with the graph.
            # The logger is accepted only in the Exception block because
            # if we are here, it means that the graph is being aborted.
            self._logger.critical(
                critical_msg,
                e,
                exc_info=True,
                extra={"status": "failed"}
            )
            tf.print(critical_msg, e)
            raise

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the metric state at each batch.

        Args:
            y_true (tf.Tensor): Ground truth labels from the dataset.
            y_pred (tf.Tensor): Predictions from the model.
            sample_weight (optional): Not used here but required by Keras API.
        """
        if self._debug_mode:
            tf.print("Starting method RSNAKaggleMetric.update_state")

        # Ensure y_true is float32 for mathematical operations
        y_true = tf.cast(y_true, tf.float32)
        y_smoothed = apply_label_smoothing(y_true)
        y_pred = tf.cast(y_pred, tf.float32)

        # Use the exact same core logic
        weighted_loss = compute_rsna_loss_core(
            y_smoothed,
            y_pred,
            self._class_weights,
            self._balancing_weights
        )
        batch_mean = tf.reduce_mean(weighted_loss)

        self.total_loss.assign_add(tf.cast(batch_mean, tf.float32))
        self.count.assign_add(1.0)

        if self._debug_mode:
            tf.print("Method RSNAKaggleMetric.update_state completed successfully")

    def result(self):
        """
        Computes and returns the final metric value (epoch average).

        Returns:
            tf.Tensor: The average RSNA score across all processed batches.
        """
        if self._debug_mode:
            tf.print("Starting method RSNAKaggleMetric.result")

        division = tf.math.divide_no_nan(
            tf.cast(self.total_loss, tf.float32),
            tf.cast(self.count, tf.float32)
        )

        if self._debug_mode:
            tf.print("Method RSNAKaggleMetric.result completed successfully")

        return division

    def reset_state(self):
        """
        Resets the state variables at the end of each epoch.
        """
        if self._debug_mode:
            tf.print("Starting method RSNAKaggleMetric.reset_state")

        self.total_loss.assign(0.0)
        self.count.assign(0.0)

        if self._debug_mode:
            tf.print("Method RSNAKaggleMetric.reset_state completed successfully.")

    def get_config(self):
        """
        Returns the configuration of the metric for serialization.

        This method allows Keras to save the metric's state to a file.
        It ensures that only JSON-serializable types are included in the
        configuration dictionary to prevent errors during the saving process.
        """
        # Get the base configuration (name and dtype) from the parent class
        config = super().get_config()

        # Filter the internal _config dictionary to ensure all values are serializable.
        # We keep only basic Python types (str, int, float, bool, dict, list).
        # This specifically excludes non-serializable objects like Loggers or TF Variables.
        serializable_config = {
            k: v for k, v in self._config.items()
            if isinstance(v, (str, int, float, bool, dict, list))
        }

        # Inject the cleaned dictionary into the Keras configuration
        config.update({
            "config": serializable_config,
        })

        return config

    @classmethod
    def from_config(cls, config):
        """
        Instantiates a metric from its configuration dictionary.

        This method is the inverse of get_config. It uses the dictionary
        extracted from the saved file to call the class constructor (__init__)
        and recreate the object exactly as it was.
        """
        # Use the dictionary unpacking operator (**) to pass the config keys
        # as keyword arguments to the class constructor.
        return cls(**config)
