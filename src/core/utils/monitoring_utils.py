# coding: utf-8

import logging
import psutil


def log_memory_usage(
    *,
    process: psutil.Process,
    stage_name: str = "",
    logger: logging.Logger
) -> None:
    """
    Logs the current RAM usage of the Python process and the overall system.

    Specifically monitors the Resident Set Size (RSS) to detect potential
    memory leaks during heavy 3D medical imaging data loading and training.

    Args:
        -process (psutil.Process): The process object to monitor.
        - stage_name (str): A label describing the current execution step.
        - logger (logging.Logger): The logger instance to use for output.
    """
    try:
        # Get the current process ID and its memory information
        mem_info = process.memory_info()

        # Convert Resident Set Size (RSS) to Megabytes (MB)
        # RSS represents the portion of memory occupied by a process that is held in RAM
        mem_mb = mem_info.rss / (1024 * 1024)

        # Retrieve the total percentage of system RAM currently in use
        total_ram_percent = psutil.virtual_memory().percent

        # Output the memory snapshot to the console
        debug_msg = (
            f">>> [RAM] {stage_name} | Process: {mem_mb:.2f} MB | "
            f"System: {total_ram_percent}%"
        )

        logger.debug(debug_msg)

    except Exception as e:
        logger.error(f"Memory monitoring failed: {e}")
