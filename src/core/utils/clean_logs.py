# coding: utf-8

import argparse
from pathlib import Path
from datetime import datetime, timedelta

TIME_DELTA = 30


def clean_old_logs(log_dir: str = "config/logs", days: int = TIME_DELTA):
    """
    Deletes log files older than `days` days.

    Args:
        log_dir: Directory containing log files (default: "config/logs").
        days: Maximum age of log files to keep (default: TIME_DELTA).
    """
    log_dir = Path(log_dir)
    now = datetime.now()

    for log_file in log_dir.glob("*.log"):
        file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
        if now - file_time > timedelta(days=days):
            log_file.unlink()
            print(f"Deleting {log_file.name}")


def main():
    """
    Main function to handle command-line arguments.
    Parses the command-line arguments and calls the clean_old_logs function
    with the provided or default values.
    """

    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()

    # Add an optional argument for the log directory with a default value of "config/logs"
    parser.add_argument("--log-dir", default="config/logs")

    # Add an optional argument for the number of days with a default value of TIME_DELTA
    # The type=int ensures the argument is converted to an integer
    parser.add_argument("--days", type=int, default=TIME_DELTA)

    # Parse the command-line arguments and store them in the args object
    args = parser.parse_args()

    # Call the clean_old_logs function with the parsed arguments
    clean_old_logs(log_dir=args.log_dir, days=args.days)


if __name__ == "__main__":
    main()
