import sys
import tensorflow as tf
from datetime import datetime
from pathlib import Path

class SystemStreamTee(object):
    """
    Duplicates stdout or stderr to a file while keeping the 
    original output in the terminal.
    """
    def __init__(self, file_string):
        self._terminal = sys.stdout

        # Ensure the directory exists before attempting to open the file
        file_path = Path(file_string)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Mode 'a' to append to the file (creates it if it does not exist)
        self._log_file = open(file_path.resolve(), "a", encoding="utf-8")

        # Add a session header
        self._print_header()

    def _print_header(self):
        """
        Prints technical context at the start of the log file.
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Detect available hardware
        cpus = tf.config.list_physical_devices('CPU')
        gpus = tf.config.list_physical_devices('GPU')
        
        header = [
            "\n\n" + "="*80,
            f"    NEW SESSION STARTED: {now}",
            f"        Hardware detected: CPUS: {len(cpus)} | GPUS: {len(gpus)}",
            f"        Mirroring to: {self._log_file.name}",
            "="*80 + "\n"
        ]
        
        for line in header:
            self.write(line + "\n")

    def write(self, message):
        # Write to the terminal
        self._terminal.write(message)

        # Write to the file
        self._log_file.write(message)

        # Force writing to disk to avoid losing data during a crash/thrashing
        self._log_file.flush()

    def flush(self):
        # Necessary for compatibility with some environments (like Jupyter or IDLE)
        self._terminal.flush()
        self._log_file.flush()

    def close(self):
        """Properly close the log file."""
        if self._log_file:
            self._log_file.close()