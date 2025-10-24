# tests/test_logger.py
import logging
import json
import pytest
from src.core.utils.logger import (
    setup_logger,
    get_current_logger,
    log_method,
    JSONFormatter
)


def test_setup_logger(caplog, tmp_path):
    """Teste la création d'un logger avec setup_logger."""
    log_dir = tmp_path / "logs"

    with setup_logger("test_process",
                      log_dir=str(log_dir),
                      use_json=False) as logger:
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        logger.info("Test message")

        # Vérifie que le fichier de log a été créé
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) == 1

        # Vérifie le contenu du log
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

    # Vérifie que c'est du JSON valide
    data = json.loads(formatted)
    assert data["message"] == "Test message"
    assert data["level"] == "INFO"


def test_log_method_decorator(caplog, mock_logger):
    """Teste le décorateur log_method."""

    @log_method()
    def test_function(*, logger=None):
        logger.info("Test function called")

    # Test avec injection automatique
    with setup_logger("test", use_json=False):
        test_function()
        assert "Test function called" in caplog.text

    # Test avec logger explicite
    test_function(logger=mock_logger)
    mock_logger.info.assert_called_with("Test function called")


def test_get_current_logger():
    """Teste la récupération du logger courant."""
    with setup_logger("test", use_json=False) as logger:
        current_logger = get_current_logger()
        assert current_logger is logger

    # Vérifie que RuntimeError est levé quand aucun logger n'est configuré
    with pytest.raises(RuntimeError):
        get_current_logger()
