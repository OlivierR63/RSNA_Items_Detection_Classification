# coding: utf-8

from pathlib import Path
from datetime import datetime, timedelta
from src.core.utils.clean_logs import clean_old_logs, main
import os
import sys


def test_clean_old_logs(tmp_path):
    """
    Test that clean_old_logs deletes files older than `days`
    and keep recent ones
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)  # Use exist_ok=True to avoid FileExistsError

    # Create a recent log file
    recent_log = log_dir / "recent.log"
    recent_log.touch()

    # Create an old log file
    old_log = log_dir / "old.log"
    old_log.touch()
    old_time = datetime.now() - timedelta(days=31)
    old_log_time = old_time.timestamp()
    os.utime(old_log, (old_log_time, old_log_time))

    # Call the function to clean old logs
    clean_old_logs(log_dir=str(log_dir), days=30)

    # Check that the old log file is deleted
    assert not old_log.exists()

    # Check that the recent log file still exists
    assert recent_log.exists()


def test_clean_old_logs_print_message(capfd, tmp_path):
    """
    Test that clean_old_logs prints a message when deleting a file.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create an old log file
    old_log = log_dir / "old.log"
    old_log.touch()
    old_time = datetime.now() - timedelta(days=31)
    old_log_time = old_time.timestamp()
    os.utime(old_log, (old_log_time, old_log_time))

    # Call the function to clean old logs
    clean_old_logs(log_dir=str(log_dir), days=30)

    # Check that the deletion message is printed
    out, err = capfd.readouterr()
    assert f"Deleting {old_log.name}" in out


def test_main_block(tmp_path, capfd):
    """
    Test that clean_old_logs is called when the script is run directly.
    """
    # Create a log directory and an old log file
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    old_log = log_dir / "old.log"
    old_log.touch()
    old_time = datetime.now() - timedelta(days=31)
    old_log_time = old_time.timestamp()
    os.utime(old_log, (old_log_time, old_log_time))

    # Path to the original script
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

    # Save the original sys.argv
    original_argv = sys.argv

    try:
        # Set sys.argv to simulate direct execution with arguments
        sys.argv = ['clean_logs.py', '--log-dir', str(log_dir), '--days', '30']

        # Call the main function directly
        main()

    finally:
        # Restore the original sys.argv
        sys.argv = original_argv
