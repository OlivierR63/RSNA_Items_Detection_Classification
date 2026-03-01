# coding utf-8

import psutil
import os
import tensorflow as tf

class SystemResourceMonitorCallbacks(tf.keras.callbacks.Callback):
    """
    Monitor RAM and CPU usage at the end of each batch to prevent OOM (Out of Memory)
    crashes, especially important when training 3D models on CPU.
    """
    def __init__(self, memory_threshold_percent=90.0):
        super().__init__()
        self.memory_threshold = memory_threshold_percent
        self.process = psutil.Process(os.getpid())

    def on_train_batch_end(self, batch, logs=None):
        # Force conversion to python int to avoid EagerTensor conflicts
        batch_int = int(batch)

        # 1. Get system-wide memory usage
        mem = psutil.virtual_memory()
        
        # 2. Get current process memory usage (RSS)
        process_mem_gb = self.process.memory_info().rss / (1024 ** 3)
        
        # 3. Log the status
        #if batch_int % 5 == 0: # Log every 5 batches to avoid cluttering
        print(f"\t\t[System Monitor] Batch {batch_int:03d}: "
              f"System RAM: {mem.percent}% | "
              f"Process RAM: {process_mem_gb:.2f} GB")

        # 4. Emergency stop
        if float(mem.percent) > self.memory_threshold:
            print(f"\n[CRITICAL] Memory usage ({mem.percent}%) exceeded threshold! "
                  "Stopping training to prevent system crash.")
            self.model.stop_training = True

# Usage:
# monitor_callback = SystemResourceMonitorCallback(memory_threshold_percent=92.0)