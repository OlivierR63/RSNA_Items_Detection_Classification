# coding: utf-8


import logging
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from src.core.utils.logger import (
    setup_logger,
    get_current_logger,
    log_method,
    JSONFormatter,
    _CURRENT_LOGGER
)


def test_setup_logger(caplog, tmp_path):
    """Test the creation of a logger with setup_logger."""
    
    log_dir = tmp_path / "logs"

    with setup_logger("test_process",
                      log_dir=str(log_dir),
                      use_json=False) as logger:

        assert logger is not None
        assert isinstance(logger, logging.Logger)
        logger.info("Test message")

        # Check that the log file was created
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) == 1

        # Check the log content
        with open(log_files[0], "r") as f:
            content = f.read()
            assert "Test message" in content


def test_json_formatter():
    """Teste le formatter JSON."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None
    )

    formatted = formatter.format(record)
    assert isinstance(formatted, str)

    # Check that it is valid JSON
    data = json.loads(formatted)
    assert data["message"] == "Test message"
    assert data["level"] == "INFO"


def test_log_method_decorator(caplog, tmp_path):
    """Test the log_method decorator."""

    log_dir = tmp_path / "logs"

    @log_method()
    def test_function(*, logger=None):
        logger.info("Test function called")

    # Test with automatic injection
    with setup_logger("test", log_dir=str(log_dir), use_json=False):
        test_function()
        log_files = list(log_dir.glob("*.log"))
        with open(log_files[0], "r") as f:
            content = f.read()
            assert "Test function called" in content

    # Test with explicit logger
    mock_logger = MagicMock()
    test_function(logger=mock_logger)
    mock_logger.info.assert_called_with("Test function called")


def test_get_current_logger(tmp_path):
    """Test retrieving the current logger."""
    log_dir = tmp_path / "logs"
    with setup_logger("test", log_dir=str(log_dir), use_json=False) as logger:
        current_logger = get_current_logger()
        assert current_logger is logger

    # Check that RuntimeError is raised when no logger is configured
    with pytest.raises(RuntimeError):
        get_current_logger()


def test_setup_logger_with_existing_handlers(caplog, tmp_path):
    """
    Test that setup_logger clears existing handlers when a logger is already set up.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)  # exist_ok = True to avoid error if log_dir already exists

    # First setup
    with setup_logger("test_process", log_dir=str(log_dir)) as logger1:
        assert logger1 is not None
        assert len(logger1.handlers) == 1  # Only file handler

        # Second setup (should clear existing handlers)
        with setup_logger("test_process", log_dir=str(log_dir)) as logger2:
            assert logger2 is not None
            assert len(logger2.handlers) == 1  # Only file handler (previous handlers cleared)


def test_setup_logger_with_console_display(caplog, tmp_path):
    """
    Test that setup_logger adds a console handler when console_display=True.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)  # exist_ok = True to avoid error if log_dir already exists

    with setup_logger("test_process", log_dir=str(log_dir), console_display=True) as logger:
        assert logger is not None
        assert len(logger.handlers) == 2  # File handler + console handler


def test_setup_logger_handlers_closed(tmp_path):
    """
    Test that setup_logger closes all handlers when exiting the context.
    """
    log_dir = tmp_path / "logs"

    # Use exist_ok = True to avoid an error if the directory already exists
    log_dir.mkdir(exist_ok=True)

    with setup_logger("test_process", log_dir=str(log_dir)) as logger:
        assert logger is not None
        assert len(logger.handlers) == 1  # File handler

    # After exiting the context, handlers should be closed and cleared
    assert logger.handlers == []


def test_log_method_decorator_without_logger_param(tmp_path):
    """
    Test that the log_method decorator does not inject a logger if the function does not accept it.
    """
    log_dir = tmp_path / "logs"

    @log_method()
    def test_function_without_logger():
        return True

    with setup_logger("test", log_dir=str(log_dir), use_json=False):
        result = test_function_without_logger()
        assert result is True


def test_log_method_decorator_with_logger_param(tmp_path):
    """
    Test that the log_method decorator injects the logger if the function accepts it.
    """
    log_dir = tmp_path / "logs"

    @log_method()
    def test_function_with_logger(logger=None):
        logger.info("Test message with logger")
        return logger is not None

    with setup_logger("test", log_dir=str(log_dir), use_json=False):
        result = test_function_with_logger()
        assert result is True
        log_files = list(log_dir.glob("*.log"))
        with open(log_files[0], "r") as f:
            content = f.read()
            assert "Test message with logger" in content


def test_setup_logger_with_json_formatter(tmp_path):
    """
    Test that setup_logger uses JSONFormatter when use_json=True.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)

    with setup_logger("test_process", log_dir=str(log_dir), use_json=True) as logger:
        assert logger is not None
        assert len(logger.handlers) == 1  # File handler
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)


def test_json_formatter_with_exception(tmp_path):
    """
    Test that JSONFormatter includes exception information when record.exc_info is set.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)

    with setup_logger("test_process", log_dir=str(log_dir), use_json=True) as logger:
        try:
            # Simulate an exception
            raise ValueError("Test exception")

        except ValueError:
            # Log the exception
            logger.exception("An error occurred")

        # Check that the log file contains the exception information
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) == 1
        with open(log_files[0], "r") as f:
            content = f.read()
            log_data = json.loads(content)
            assert "exception" in log_data
            assert "Test exception" in log_data["exception"]


