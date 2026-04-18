# coding: utf-8

import pytest
import psutil
from unittest.mock import MagicMock, patch
from src.core.callbacks.system_resource_monitor_callback import SystemResourceMonitorCallbacks


class TestSystemResourceMonitorCallbacks:
    """
    Unit tests for SystemResourceMonitorCallbacks.
    Focuses on memory threshold triggers and cleanup operations.
    """

    @pytest.fixture
    def mock_model(self):
        """
        Creates a mock Keras model with a stop_training attribute.
        """
        model = MagicMock()
        model.stop_training = False
        return model

    def test_init_sets_default_threshold(self):
        """
        Ensures the callback initializes with the correct memory threshold.
        """
        callback = SystemResourceMonitorCallbacks(memory_threshold_percent=85.0)
        assert callback.memory_threshold == 85.0
        assert isinstance(callback.process, psutil.Process)

    @patch("psutil.virtual_memory")
    @patch("psutil.Process.memory_info")
    def test_on_train_batch_end_under_threshold(self, mock_mem_info, mock_virt_mem, mock_model):
        """
        Tests that training continues when memory usage is below the threshold.
        """
        # Setup: 70% usage (below default 90%)
        mock_virt_mem.return_value.percent = 70.0
        # Mock RSS memory (1 GB in bytes)
        mock_mem_info.return_value.rss = 1024 ** 3

        callback = SystemResourceMonitorCallbacks(memory_threshold_percent=90.0)
        callback.model = mock_model

        callback.on_train_batch_end(batch=0)

        # Validation: stop_training should remain False
        assert mock_model.stop_training is False

    @patch("psutil.virtual_memory")
    @patch("psutil.Process.memory_info")
    def test_on_train_batch_end_exceeds_threshold(self, mock_mem_info, mock_virt_mem, mock_model):
        """
        Tests that training stops immediately when memory exceeds the threshold.
        """
        # Setup: 95% usage (above 90% threshold)
        mock_virt_mem.return_value.percent = 95.0
        mock_mem_info.return_value.rss = 2 * (1024 ** 3)

        callback = SystemResourceMonitorCallbacks(memory_threshold_percent=90.0)
        callback.model = mock_model

        callback.on_train_batch_end(batch=10)

        # Validation: stop_training must be set to True
        assert mock_model.stop_training is True

    @patch("tensorflow.keras.backend.clear_session")
    @patch("gc.collect")
    def test_on_epoch_end_triggers_cleanup(self, mock_gc, mock_clear_session):
        """
        Verifies that Keras session is cleared and GC is collected at epoch end.
        """
        callback = SystemResourceMonitorCallbacks()

        # Execute epoch end logic
        callback.on_epoch_end(epoch=0)

        # Validation: Both cleanup functions must be called
        mock_clear_session.assert_called_once()
        mock_gc.assert_called_once()

    @patch("psutil.virtual_memory")
    def test_batch_integer_conversion(self, mock_virt_mem, mock_model, capsys):
        """
        Ensures the batch number is correctly handled in the print output.
        """
        mock_virt_mem.return_value.percent = 50.0
        callback = SystemResourceMonitorCallbacks()
        callback.model = mock_model

        # Test with batch 0 (should print as Batch 001)
        callback.on_train_batch_end(batch=0)

        captured = capsys.readouterr()
        assert "Batch 001" in captured.out
