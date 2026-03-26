# coding: utf-8

import pytest
import tensorflow as tf
from src.core.models.backbone_2d import Backbone2D


@pytest.mark.parametrize("model_type", ["MobileNetV2", "ResNet50"])
@pytest.mark.parametrize("freeze", [True, False])
def test_backbone_build_success(mock_config, mock_logger, model_type, freeze):
    """
    Test successful instantiation of different backbone types.
    """
    # Override fixture config
    mock_config["models"]["backbone_2d"]["type"] = model_type
    mock_config["models"]["backbone_2d"]["freeze"] = freeze

    backbone = Backbone2D(mock_config, mock_logger)
    model = backbone.get_model()

    assert isinstance(model, tf.keras.Model)

    # Instead of checking model.trainable, we might verify if the model
    # embeds trainable weights.
    trainable_weights = len(model.trainable_weights)

    if freeze:
        assert trainable_weights == 0, f"Expected 0 trainable weights, got {trainable_weights}"
    else:
        assert trainable_weights > 0


@pytest.mark.parametrize("missing_key", [
    ("models",),
    ("models", "backbone_2d"),
    ("models", "backbone_2d", "img_shape"),
    ("models", "backbone_2d", "type")
])
def test_backbone_missing_config_keys(mock_config, mock_logger, missing_key, caplog):
    """
    Test that missing required keys raise the appropriate RuntimeError/ValueError.
    """
    # Deep delete the key from the config dictionary
    cfg = mock_config
    for key in missing_key[:-1]:
        cfg = cfg[key]
    del cfg[missing_key[-1]]

    with pytest.raises(RuntimeError):
        Backbone2D(mock_config, mock_logger)

    assert any("Fatal error" in record.message for record in caplog.records)


def test_backbone_invalid_shape(mock_config, mock_logger, caplog):
    """
    Test that an image shape without 3 dimensions raises an error.
    """
    mock_config["models"]["backbone_2d"]["img_shape"] = [224, 224]  # Missing Channels

    with pytest.raises(RuntimeError) as excinfo:
        Backbone2D(mock_config, mock_logger)

    assert "img_shape must have 3 dimensions" in str(excinfo.value)
    assert any("img_shape must have 3 dimensions" in record.message for record in caplog.records)


def test_backbone_unsupported_type(mock_config, mock_logger):
    """
    Test that a non-implemented model type raises an error.
    """
    mock_config["models"]["backbone_2d"]["type"] = "InceptionV3"

    with pytest.raises(RuntimeError, match="Unsupported backbone"):
        Backbone2D(mock_config, mock_logger)


def test_backbone_getters(mock_config, mock_logger):
    """
    Verify that getters return expected shapes and instances.
    """
    backbone = Backbone2D(mock_config, mock_logger)

    shape = backbone.get_output_shape()
    # For MobileNetV2 with 224x224, standard output is (None, 7, 7, 1280)
    assert isinstance(shape, tuple)
    assert shape[1:3] == (7, 7)
