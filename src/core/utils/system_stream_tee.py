# coding: utf-8

import sys
import tensorflow as tf
from datetime import datetime
from pathlib import Path


class _StreamProxy(object):
    """
    Internal proxy helper to route a specific system stream to the log file.
    """

    def __init__(self, original_stream, log_file):
        self._original_stream = original_stream
        self._log_file = log_file

    def write(self, message):
        # Write to the original terminal stream (stdout or stderr)
        self._original_stream.write(message)
        # Write to the shared log file
        self._log_file.write(message)
        self._log_file.flush()

    def flush(self):
        self._original_stream.flush()
        self._log_file.flush()


class SystemStreamTee(object):
    """
    Intercepts and duplicates both stdout and stderr into a single log file.
    """

    def __init__(self, file_path: Path):
        """
        Initializes the dual logger and replaces global system streams.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Open the single shared log file in append mode
        self._log_file = open(file_path.resolve(), "a", encoding="utf-8")

        # Print a single clean header for the session
        self._print_header()

        # Replace standard streams with dedicated proxies
        sys.stdout = _StreamProxy(sys.stdout, self._log_file)
        sys.stderr = _StreamProxy(sys.stderr, self._log_file)

    def _print_header(self):
        """
        Prints technical context and hardware detection once per session.
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cpus = tf.config.list_physical_devices('CPU')
        gpus = tf.config.list_physical_devices('GPU')

        header = [
            "\n\n" + "="*80,
            f"    NEW GLOBAL SESSION STARTED: {now}",
            f"        Hardware detected: CPUS: {len(cpus)} | GPUS: {len(gpus)}",
            f"        Combined Mirroring: STDOUT & STDERR -> {self._log_file.name}",
            "="*80 + "\n"
        ]

        for line in header:
            # Safely write the header directly to the file and original stdout
            sys.stdout.write(line + "\n")
            self._log_file.write(line + "\n")

        self._log_file.flush()
