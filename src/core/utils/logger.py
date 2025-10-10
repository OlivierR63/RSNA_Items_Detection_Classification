# coding: utf-8

import logging
from logging.handlers import RotatingFileHandler, JSONFormatter
from pathlib import Path
from datetime import datetime
import os
import sys
import contextlib
from typing import Optional, Callable, Any, Generator
import json
from functools import wraps

# Global variable to hold the current logger instance
_CURRENT_LOGGER: Optional[logging.Logger] = None

@contextlib.contextmanager
def setup_logger(process_name: str, config_dir: str = "config", use_json: bool = False) -> Generator[logging.Logger, None, None] :
    """Configures a logger linked to the current process ID.
    Supports both text and JSON formatting.

    Args:
        process_name: Name of the process (e.g., "train").
        config_dir: Configuration directory (default: "config").
        use_json: If True, uses JSON formatting for logs (default: False).

    Yields:
        logging.Logger: Configured logger (also stored in _CURRENT_LOGGER).
    """

    # 1. Get the current process ID
    pid = os.getpid()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 2. Create the logs directory if it doesn't exist
    log_dir = Path(config_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 3. Log file name: {process_name}_{pid}_{timestamp}.log
    log_file = log_dir / f"{process_name}_{pid}_{timestamp}.log"

    # 4. Configure the logger
    logger = logging.getLogger(f"{process_name}_{pid}")
    logger.setLevel(logging.INFO)

    # 5. Clear any existing handlers to avoid duplicate logs
    if _CURRENT_LOGGER is not None:
        for handler in _CURRENT_LOGGER.handlers:
            handler.close()
        _CURRENT_LOGGER.handlers.clear()

    # 5. Formatter
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 6. File handler (automatically closed at the end of the block)
    file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 7. Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Store the logger in the global variable
    _CURRENT_LOGGER = logger

    try:
        yield logger  # Pass the logger to the 'with' block
    finally:
        # 8. Close handlers at the end of the process (even in case of error)
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
        _CURRENT_LOGGER = None # Reset the global variable


def get_current_logger() -> logging.Logger:
    """Returns the current logger instance.

    Returns:
        logging.Logger: The current logger instance.

    Raises:
        RuntimeError: If no logger is currently set up.
    """
    if _CURRENT_LOGGER is None:
        raise RuntimeError("No logger is currently set up. Call setup_logger() first.")
    return _CURRENT_LOGGER


def log_method(logger_key: str = "logger") -> Callable:
    """Decorator to automatically inject the current logger into a method.

    Args:
        logger_key: The name of the parameter to inject the logger into (default: "logger").

    Returns:
        A decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the current logger
            logger = get_current_logger()

            # Inject the logger into the keyword arguments
            kwargs[logger_key] = logger

            # Call the original function
            return func(*args, **kwargs)
        return wrapper
    return decorator


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_data)