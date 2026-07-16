# coding: utf-8

import logging
from pathlib import Path
from datetime import datetime
import os
import sys
import contextlib
from typing import Callable, Generator
import json
from functools import wraps
from inspect import signature

# Global variable to hold the current logger instance
_CURRENT_LOGGER: logging.Logger | None = None


@contextlib.contextmanager
def setup_logger(
    process_name: str,
    log_dir: str = "logs"
) -> Generator[logging.Logger, None, None]:

    """
    Configures a logger linked to the current process ID.
    Supports both text and JSON formatting.

    Args:
        process_name: Name of the process (e.g., "train").
        log_dir: log files storage folder (default: "logs").
        use_json: If True, uses JSON formatting for logs (default: False).

    Yields:
        logging.Logger: Configured logger (also stored in _CURRENT_LOGGER).
    """

    global _CURRENT_LOGGER

    # Default settings in case config loader is missing or not yet in sys.path
    logging_level = "INFO"
    console_display = True
    use_json = False

    try:
        # Lazy import of ConfigLoader to avoid circular/missing dependencies during bootstrap
        from src.config.config_loader import ConfigLoader
        logging_cfg = ConfigLoader().get_value('logging', None)
        if logging_cfg:
            logging_level = logging_cfg.get('level', logging_level)
            console_display = logging_cfg.get('console_display', console_display)
            use_json = logging_cfg.get('use_json', use_json)
    except Exception:
        # Silent fallback to default console settings during initial bootstrap phases
        pass

    # 1. Get the current process ID
    pid = os.getpid()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 2. Create the logs directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 3. Log file name: {process_name}_{pid}_{timestamp}.log
    log_file = log_dir / f"{process_name}_{pid}_{timestamp}.log"

    # 4. Configure the logger
    logger = logging.getLogger(f"{process_name}_{pid}")
    logger.propagate = False  # Prevent logs from being propagated to the root logger

    if (logging_level == "DEBUG"):
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # 5. Clear any existing handlers to avoid duplicate logs
    if _CURRENT_LOGGER is not None:
        for handler in _CURRENT_LOGGER.handlers:
            handler.close()
        _CURRENT_LOGGER.handlers.clear()

    # 6. Formatter
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # 7. File handler (automatically closed at the end of the block)
    file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 8. Console handler
    if console_display:
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
        _CURRENT_LOGGER = None  # Reset the global variable


def get_current_logger() -> logging.Logger:
    """
    Retrieves the currently configured logger instance for the application.

    If no logger has been initialized via `setup_logger()`, this function
    returns a default root logger. This fallback prevents initialization
    errors during module imports, particularly during unit test discovery
    with Pytest.

    Returns:
        logging.Logger: The active logger instance or a default fallback logger.
    """
    global _CURRENT_LOGGER

    if _CURRENT_LOGGER is None:
        # Fallback to a default logger to allow Pytest discovery and
        # prevent RuntimeError during imports.
        _CURRENT_LOGGER = logging.getLogger("default")

    return _CURRENT_LOGGER


def get_current_log_file() -> Path | None:
    """
    Retrieves the absolute path of the current log file.

    Returns:
        Optional[Path]: The path to the log file if a FileHandler is active,
                        otherwise None.
    """
    logger = get_current_logger()

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename).resolve()

    return None


def log_method(logger_key: str = "logger") -> Callable:
    """
    Decorator that injects the current logger into a function's kwargs when appropriate.

    The decorator inspects the wrapped function's signature and will inject the current
    logger under `logger_key` only if:
      - the wrapped function declares a parameter with that name, and
      - the caller did not already provide that keyword argument.

    This prevents adding an unexpected keyword argument to functions that do not accept
    a logger parameter and avoids overwriting an explicitly supplied logger. The check
    is performed using `inspect.signature` so the decorator is safe to apply broadly.

    Args:
        logger_key: The name of the keyword parameter to inject
        the logger into (default: "logger").

    Returns:
        A decorator function that wraps the target function and injects
        the current logger when appropriate.
    """

    def decorator(func: Callable) -> Callable:
        sig = signature(func)
        accepts_logger = logger_key in sig.parameters

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the current logger
            if accepts_logger and logger_key not in kwargs:
                # Inject the logger into the keyword arguments
                kwargs[logger_key] = get_current_logger()

            # Call the original function
            return func(*args, **kwargs)

        return wrapper
    return decorator


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
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
