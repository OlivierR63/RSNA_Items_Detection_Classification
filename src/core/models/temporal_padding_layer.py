# coding: utf-8

import tensorflow as tf
from keras import layers


class TemporalPaddingLayer(layers.Layer):
    """
    Custom Keras layer that ensures a fixed temporal depth for 3D convolutional architectures.

    This layer acts as a structural guardrail:
    1. It dynamically pads sequences that are shorter than the target depth.
    2. It truncates sequences that exceed the target depth.
    3. It forces a static shape definition, which is required by Conv3D layers to
       initialize their weights during the model compilation phase.
    """

    def __init__(self, target_depth, height, width, channels, **kwargs):
        """
        Initializes the padding layer with expected dimensions.

        Args:
            target_depth (int): The exact number of frames/slices required (Time dimension).
            height (int): The spatial height of the feature maps.
            width (int): The spatial width of the feature maps.
            channels (int): The number of filters/channels from the 2D backbone.
            **kwargs: Standard Keras layer keyword arguments.
        """
        super(TemporalPaddingLayer, self).__init__(**kwargs)
        self.target_depth = target_depth
        self.height = height
        self.width = width
        self.channels = channels

    def call(self, x):
        """
        Processes the input tensor to match the target temporal depth.

        Args:
            x (tf.Tensor): Input tensor from a TimeDistributed backbone.
                           Expected shape: [Batch, Time, (H, W), Channels]

        Returns:
            tf.Tensor: Padded or truncated tensor with a fixed static shape.
        """
        # Retrieve dynamic shape from the execution graph
        input_shape = tf.shape(x)
        current_depth = input_shape[1]

        # Determine the number of zero-frames to add
        padding_needed = tf.maximum(0, self.target_depth - current_depth)

        # Create a dynamic padding mask based on the tensor's rank.
        # This ensures compatibility whether MobileNetV2 outputs 5D (spatial)
        # or 3D (pooled) tensors.
        rank = tf.rank(x)
        paddings = tf.zeros([rank, 2], dtype=tf.int32)

        # Update the temporal dimension (index 1) in the padding matrix.
        # Format for tf.pad: [[before_0, after_0], [before_1, after_1], ...]
        update_indices = [[1, 1]]
        update_values = [padding_needed]
        paddings = tf.tensor_scatter_nd_update(paddings, update_indices, update_values)

        # Apply constant zero padding
        x_padded = tf.pad(x, paddings, mode='CONSTANT', constant_values=0)

        # Truncate the sequence if it is longer than target_depth (Defensive programming)
        x_trimmed = x_padded[:, :self.target_depth, ...]

        # Hard-set the static shape for Keras's weight initialization and graph optimization.
        # This is critical for subsequent layers like Conv3D or GlobalAveragePooling3D.
        static_shape = [None, self.target_depth, self.height, self.width, self.channels]
        x_trimmed.set_shape(static_shape)

        return x_trimmed

    def compute_output_shape(self, input_shape):
        """
        Returns the deterministic output shape for Keras shape inference.
        """
        return (input_shape[0], self.target_depth, self.height, self.width, self.channels)

    def get_config(self):
        """
        Ensures the layer is serializable for model saving/loading (.h5 or SavedModel).
        """
        config = super().get_config()
        config.update({
            "target_depth": self.target_depth,
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
        })
        return config
