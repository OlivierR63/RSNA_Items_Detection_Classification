# coding: utf-8

import pytest
from unittest.mock import MagicMock, patch
from src.core.callbacks.cpu_temperature_monitor_callback import CPUTemperatureMonitorCallback


class TestCPUTemperatureMonitorCallback:
    """
    Unit tests for CPUTemperatureMonitorCallback.
    Covers Linux/Windows logic and training interruption triggers.
    """

    @pytest.fixture
    def mock_model(self):
        """
        Provides a mock Keras model.
        """
        model = MagicMock()
        model.stop_training = False
        return model

    # --- 1. Testing _get_cpu_temp logic across OS ---

    @patch("platform.system")
    @patch("psutil.sensors_temperatures", create=True)
    def test_get_cpu_temp_linux_coretemp(self, mock_psutil, mock_platform):
        """
        Verifies Linux temperature retrieval via 'coretemp'.
        """
        mock_platform.return_value = "Linux"
        # Mocking psutil output structure
        mock_entry = MagicMock()
        mock_entry.current = 55.0
        mock_psutil.return_value = {'coretemp': [mock_entry]}

        callback = CPUTemperatureMonitorCallback()
        assert callback._get_cpu_temp() == 55.0

    @patch("platform.system")
    @patch("psutil.sensors_temperatures", create=True)
    def test_get_cpu_temp_linux_thermal(self, mock_psutil, mock_platform):
        """
        Verifies Linux temperature retrieval via 'cpu_thermal' (Raspberry Pi style).
        """
        mock_platform.return_value = "Linux"
        mock_entry = MagicMock()
        mock_entry.current = 42.0
        mock_psutil.return_value = {'cpu_thermal': [mock_entry]}

        callback = CPUTemperatureMonitorCallback()
        assert callback._get_cpu_temp() == 42.0

    @patch("platform.system")
    def test_get_cpu_temp_windows_success(self, mock_platform):
        """
        Verifies Windows temperature retrieval and Kelvin to Celsius conversion.
        """
        mock_platform.return_value = "Windows"

        # Mocking the WMI object and its return value
        with patch("wmi.WMI") as mock_wmi_class:
            mock_wmi_instance = mock_wmi_class.return_value
            mock_temp_info = MagicMock()
            # 3031.5 tenths of Kelvin = 303.15 K = 30.0 C
            mock_temp_info.CurrentTemperature = 3031.5
            mock_wmi_instance.MSAcpi_ThermalZoneTemperature.return_value = [mock_temp_info]

            callback = CPUTemperatureMonitorCallback()
            assert pytest.approx(callback._get_cpu_temp(), 0.1) == 30.0

    @patch("platform.system")
    def test_get_cpu_temp_unsupported_os(self, mock_platform):
        """
        Ensures None is returned on unsupported Operating Systems (e.g., Darwin/MacOS).
        """
        mock_platform.return_value = "Darwin"
        callback = CPUTemperatureMonitorCallback()
        assert callback._get_cpu_temp() is None

    # --- 2. Testing Callback Triggers ---

    @patch.object(CPUTemperatureMonitorCallback, "_get_cpu_temp")
    def test_on_train_batch_end_normal_temp(self, mock_get_temp, mock_model):
        """
        Ensures training is NOT stopped when temperature is within limits.
        """
        mock_get_temp.return_value = 50.0
        callback = CPUTemperatureMonitorCallback(temp_threshold=80.0)
        callback.model = mock_model

        callback.on_train_batch_end(batch=0)
        assert mock_model.stop_training is False

    @patch.object(CPUTemperatureMonitorCallback, "_get_cpu_temp")
    def test_on_train_batch_end_overheat(self, mock_get_temp, mock_model, capsys):
        """
        Ensures training is stopped and CRITICAL message is printed when overheating.
        """
        mock_get_temp.return_value = 90.0
        callback = CPUTemperatureMonitorCallback(temp_threshold=85.0)
        callback.model = mock_model

        callback.on_train_batch_end(batch=20)

        assert mock_model.stop_training is True
        captured = capsys.readouterr()
        assert "[CRITICAL]" in captured.out

    @patch.object(CPUTemperatureMonitorCallback, "_get_cpu_temp")
    def test_on_train_batch_end_none_handling(self, mock_get_temp, mock_model, capsys):
        """
        Ensures a warning is printed only once if sensors are unavailable.
        """
        mock_get_temp.return_value = None
        callback = CPUTemperatureMonitorCallback()
        callback.model = mock_model

        # Call for batch 0: should print warning
        callback.on_train_batch_end(batch=0)
        out_batch_0 = capsys.readouterr().out
        assert "[Warning]" in out_batch_0

        # Call for batch 1: should NOT print warning again
        callback.on_train_batch_end(batch=1)
        out_batch_1 = capsys.readouterr().out
        assert "[Warning]" not in out_batch_1

    @patch.object(CPUTemperatureMonitorCallback, "_get_cpu_temp")
    def test_logging_interval(self, mock_get_temp, mock_model, capsys):
        """
        Verifies that temperature is logged only every 10 batches.
        """
        mock_get_temp.return_value = 40.0
        callback = CPUTemperatureMonitorCallback()
        callback.model = mock_model

        # Batch 0 (0 % 10 == 0) -> Logged
        callback.on_train_batch_end(batch=0)
        assert "[CPU Temp]" in capsys.readouterr().out

        # Batch 1 -> Not logged
        callback.on_train_batch_end(batch=1)
        assert "[CPU Temp]" not in capsys.readouterr().out

        # Batch 10 -> Logged
        callback.on_train_batch_end(batch=10)
        assert "[CPU Temp]" in capsys.readouterr().out
