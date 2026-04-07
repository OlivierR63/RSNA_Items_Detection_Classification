# coding: utf-8

import platform
import pytest
from unittest.mock import MagicMock, patch
from contextlib import ExitStack
from src.core.callbacks.cpu_temperature_monitor_callback import CPUTemperatureMonitorCallback


class TestCPUTemperatureMonitorCallback:

    # --- REAL OS TESTS (Functionality) ---

    @pytest.mark.skipif(platform.system() != "Windows", reason="WMI requires Windows")
    def test_get_cpu_temp_windows_real_success(self):
        """
        Tests the actual WMI logic on Windows.
        """

        with ExitStack() as stack:
            mock_wmi_class = stack.enter_context(patch("wmi.WMI"))
            mock_wmi_instance = mock_wmi_class.return_value
            mock_temp_info = MagicMock()
            mock_temp_info.CurrentTemperature = 3031.5  # 30.0 °C
            mock_wmi_instance.MSAcpi_ThermalZoneTemperature.return_value = [mock_temp_info]

            callback = CPUTemperatureMonitorCallback()
            assert pytest.approx(callback._get_cpu_temp(), 0.1) == 30.0

    @pytest.mark.skipif(platform.system() != "Linux", reason="Sensors require Linux")
    def test_get_cpu_temp_linux_real_success(self):
        """
        Tests the actual psutil logic on Linux.
        """
        with ExitStack() as stack:
            mock_psutil = stack.enter_context(patch("psutil.sensors_temperatures"))
            mock_psutil.return_value = {'coretemp': [MagicMock(current=55.0)]}

            callback = CPUTemperatureMonitorCallback()
            assert callback._get_cpu_temp() == 55.0

    # --- BRANCH COVERAGE TESTS (Logic) ---

    def test_get_cpu_temp_linux_branch_logic(self):
        """
        Forces entry into Linux branch to cover psutil code on any OS.
        """
        with ExitStack() as stack:
            # Add create=True to mock attribute that don't exist on Windows
            mock_psutil = stack.enter_context(
                patch("psutil.sensors_temperatures", create=True)
            )

            # We simulate both common Linux sensor keys for 100% branch coverage
            mock_psutil.return_value = {
                'cpu_thermal': [MagicMock(current=45.0)]
            }

            callback = CPUTemperatureMonitorCallback()
            callback.os_name = "Linux"  # Manual override for coverage
            assert callback._get_cpu_temp() == 45.0

    def test_get_cpu_temp_unsupported_os(self):
        """
        Covers the final 'return None' (Line 48).
        """
        callback = CPUTemperatureMonitorCallback()
        callback.os_name = "UnknownOS"
        assert callback._get_cpu_temp() is None

    def test_get_cpu_temp_windows_exception_handling(self):
        """
        Covers the 'except Exception' block in Windows logic.
        """
        with ExitStack() as stack:
            # We mock 'wmi' at the module level where it was imported
            mock_wmi = stack.enter_context(
                patch("src.core.callbacks.cpu_temperature_monitor_callback.wmi")
            )

            callback = CPUTemperatureMonitorCallback()
            callback.os_name = "Windows"

            # Forces the WMI() call to raise an exception.
            # This triggers the 'except Exception' block immediately
            mock_wmi.WMI.side_effect = Exception("WMI service not available")

            # The return value must be None
            assert callback._get_cpu_temp() is None

    # --- CALLBACK LOGIC TESTS ---

    def test_on_train_batch_end_monitoring(self):
        """
        Covers temperature logging and emergency stop logic.
        """
        with ExitStack() as stack:
            callback = CPUTemperatureMonitorCallback(temp_threshold=80.0)
            callback.model = MagicMock()
            callback.model.stop_training = False

            mock_print = stack.enter_context(patch("builtins.print"))

            # Case 1: Normal temp, log every 10 batches (Batch 0)
            stack.enter_context(patch.object(callback, '_get_cpu_temp', return_value=50.0))
            callback.on_train_batch_end(batch=0)
            mock_print.assert_called_with(" - [CPU Temp] 50.0°C")
            assert callback.model.stop_training is False

            # Case 2: Critical temp, trigger stop
            stack.enter_context(patch.object(callback, '_get_cpu_temp', return_value=90.0))
            callback.on_train_batch_end(batch=10)
            assert callback.model.stop_training is True

    def test_on_train_batch_end_warning_when_none(self):
        """
        Covers the warning print when temperature cannot be read.
        """
        with ExitStack() as stack:
            callback = CPUTemperatureMonitorCallback()
            stack.enter_context(patch.object(callback, '_get_cpu_temp', return_value=None))
            mock_print = stack.enter_context(patch("builtins.print"))

            callback.on_train_batch_end(batch=0)

            # Verify the "Unable to read CPU temperature" message
            args, _ = mock_print.call_args
            assert "Unable to read CPU temperature" in args[0]
