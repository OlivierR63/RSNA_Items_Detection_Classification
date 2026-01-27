# coding: utf-8

# coding: utf-8
import tensorflow as tf

def rsna_weighted_log_loss(y_true, y_pred):
    """
    Calculates the weighted cross-entropy loss based on RSNA 2024 criteria.
    Weights: Normal = 1.0, Moderate = 2.0, Severe = 4.0.
    """
    # Weights for each class (Normal, Moderate, Severe)
    class_weights = tf.constant([1.0, 2.0, 4.0], dtype=tf.float32)
    
    # Ensure y_true consists of integers and convert to one-hot encoding
    # We flatten y_true to handle potential (batch_size, 1) shapes
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_true_one_hot = tf.one_hot(y_true, depth=3)
    
    # Clip predictions to prevent numerical instability (log(0) or log(1))
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate cross-entropy: -sum(y_true * log(y_pred))
    loss = -y_true_one_hot * tf.math.log(y_pred)
    
    # Apply class weights and reduce across the class dimension
    weighted_loss = tf.reduce_sum(loss * class_weights, axis=-1)
    
    return weighted_loss

def zero_loss(y_true, y_pred):
    """
    Returns a zero loss with the same shape as predictions.
    Used for outputs that should not contribute to the gradient (e.g., ID pass-through).
    """
    return tf.reduce_mean(tf.zeros_like(y_pred), axis=-1)