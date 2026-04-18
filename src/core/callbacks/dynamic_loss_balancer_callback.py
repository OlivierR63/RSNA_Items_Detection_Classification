# coding: utf-8

# import tensorflow as tf
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback


class DynamicLossBalancerCallback(Callback):
    """
    Keras Callback to dynamically adjust multi-task loss weights.

    This callback ensures that different output losses (e.g., severity and location)
    maintain a similar numerical magnitude, preventing one task from dominating
    the gradient updates.

    Attributes:
        _loss_weight_var (tf.Variable): The TensorFlow variable used as a loss weight
                                       during model.compile().
        _logger (logging.Logger): Optional logger for tracking updates.
        _momentum (float): Smoothing factor (0 to 1) to prevent abrupt weight changes.
        _min_weight (float): Minimum allowable value for the weight.
        _max_weight (float): Maximum allowable value for the weight.
    """

    def __init__(
        self,
        weight_variable: tf.Variable,
        logger=None,
        momentum: float = 0.5,
        min_weight: float = 0.5,
        max_weight: float = 15.0
    ) -> None:
        super().__init__()
        self._loss_weight_var = weight_variable
        self._logger = logger
        self._momentum = momentum
        self._min_weight = min_weight
        self._max_weight = max_weight

    def on_epoch_end(self, epoch, logs=None):
        """
        Calculates and updates the loss weight at the end of each epoch.
        """
        logs = logs or {}

        # 1. Retrieve the raw losses from the logs dictionary
        # Note: Ensure these names match the output layers in your model
        sev_loss = logs.get('severity_output_loss')
        loc_loss = logs.get('location_output_loss')

        if sev_loss is not None and loc_loss is not None and loc_loss > 1e-6:
            # 2. Calculate the target weight to balance magnitudes
            # Aim: weight_variable * loc_loss ≈ sev_loss
            target_weight = sev_loss / (loc_loss + K.epsilon())

            # 3. Apply momentum-based smoothing to ensure training stability
            # This prevents the weight from oscillating wildly between epochs
            current_weight = float(K.get_value(self._loss_weight_var))
            new_loc_weight = (
                (self._momentum * current_weight) +
                ((1 - self._momentum) * target_weight)
            )

            # 4. Clip the weight within a safe range to avoid extreme bias
            new_loc_weight = np.clip(new_loc_weight, self._min_weight, self._max_weight)

            # 5. Update the TensorFlow variable directly
            # This reflects in the model's loss calculation for the next epoch
            K.set_value(self._loss_weight_var, new_loc_weight)

            # 6. Logging the balancing operation
            if self._logger:
                self._logger.info(
                    f" [LossBalancer] End of Epoch {epoch + 1} Stats: "
                    f"Severity Loss = {sev_loss:.4f}, Location Loss = {loc_loss:.4f}"
                )
                self._logger.info(
                    f" [LossBalancer] Location Weight Update for Epoch {epoch + 2}: "
                    f"{current_weight:.2f} -> {new_loc_weight:.2f}"
                )
        else:
            # Warning if logs are missing or loss is zero
            if self._logger:
                self._logger.warning(
                    " [LossBalancer] Skipping weight update: "
                    "Missing or invalid loss values in logs."
                )
