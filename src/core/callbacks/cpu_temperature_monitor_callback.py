# coding: utf-8

import tensorflow as tf
import psutil
import platform

# This library is imported only into Windows environment
if platform.system() == "Windows":
    import wmi
else:  # pragma: no cover (Not counted as missing on Windows coverage)
    wmi = None


class CPUTemperatureMonitorCallback(tf.keras.callbacks.Callback):
    """
    Callback to monitor CPU temperature during training to prevent thermal throttling.
    """
    def __init__(self, temp_threshold=85.0):
        super().__init__()
        self.temp_threshold = temp_threshold
        self.os_name = platform.system()

    def _get_cpu_temp(self):
        """
        Internal helper to fetch CPU temperature based on the OS.
        """
        if self.os_name == "Linux":
            # Returns a dict of hardware temperatures
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
            elif 'cpu_thermal' in temps:  # Common on Raspberry Pi
                return temps['cpu_thermal'][0].current

        elif self.os_name == "Windows":
            # Requires WMI and often specific hardware drivers
            try:
                w = wmi.WMI(namespace="root\\wmi")
                temperature_info = w.MSAcpi_ThermalZoneTemperature()[0]
                print(f"temperature_info = {temperature_info}")

                # Temperature is in tenths of Kelvin, convert to Celsius
                return (temperature_info.CurrentTemperature / 10.0) - 273.15

            except Exception:
                return None

        return None

    def on_train_batch_end(self, batch, logs=None):
        """
        Monitor temperature at the end of each batch.
        """
        current_temp = self._get_cpu_temp()

        if current_temp is not None:
            # Log every 10 batches to avoid console spam
            if batch % 10 == 0:
                print(f" - [CPU Temp] {current_temp:.1f}°C")

            # Emergency stop if CPU is too hot
            if current_temp > self.temp_threshold:
                print(f"\n[CRITICAL] CPU Temperature ({current_temp:.1f}°C) "
                      f"exceeded threshold ({self.temp_threshold}°C)!")
                self.model.stop_training = True
        else:
            # Only print warning once if temperature cannot be read
            if batch == 0:
                print("\n[Warning] Unable to read CPU temperature on this system.")
